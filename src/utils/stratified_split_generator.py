import itertools
import math
import os
import re
from pathlib import Path
from typing import Optional, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fairlearn.metrics import (
    MetricFrame,
    count,
    equalized_odds_difference,
    equalized_odds_ratio,
    false_positive_rate,
    true_positive_rate,
)
from fairlearn.reductions import EqualizedOdds
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split


class StratifiedSplitGenerator:
    def __init__(
        self,
        passion_exp: str = "experiment_standard_split_conditions",
        eval_data_path="../../assets/evaluation",
        dataset_dir: Union[str, Path] = "../../data/PASSION/",
        meta_data_file: Union[str, Path] = "label.csv",
        split_file: Union[str, Path, None] = "PASSION_split.csv",
    ):
        self.eval_data_path = Path(eval_data_path)
        self.dataset_dir = Path(dataset_dir)
        self.out_dir = self.eval_data_path / passion_exp
        os.makedirs(self.out_dir, exist_ok=True)

        # TODO: only read once in the whole pipeline
        self.df_labels = pd.read_csv(self.dataset_dir / meta_data_file)
        self.df_split = pd.read_csv(self.dataset_dir / split_file)

    def _generate_age_group(self, df):
        bins = range(0, 101, 5)
        return pd.cut(
            df["age"],
            bins=bins,
            labels=[f"{i:02}-{i + 4:02}" for i in bins[:-1]],
            right=False,
        )

    def _create_split(
        self,
        original_df,
        stratify_cols: [str] = None,
        print_unknown_stratification_issues: bool = False,
        seed: int = 42,
    ):
        df = original_df.copy()

        # Filter the TRAIN data
        df_save_single_records = pd.DataFrame()
        if stratify_cols is None:
            stratify_str = "none" + f"__seed_{seed}"
        else:
            stratify_str = "_".join(stratify_cols) + f"__seed_{seed}"
            print(f"filtering stratification potential issues for: {stratify_str}")
            lonely_subject_ids = []
            if "fitzpatrick" in stratify_cols:
                lonely_subject_ids.extend(["AA00970059", "AA00971417", "AA00971384"])
            if "country" in stratify_cols:
                lonely_subject_ids.extend(["AA00970059"])
            if "country" in stratify_cols and "fitzpatrick" in stratify_cols:
                lonely_subject_ids.extend(
                    [
                        "AA00970095",
                        "AA00970651",
                        "AA00970828",
                        "AA00971108",
                        "AA00971141",
                        "AA00971251",
                        "AA00971417",
                        "AA00971972",
                    ]
                )
                if "sex" in stratify_cols:
                    lonely_subject_ids.extend(
                        [
                            "AA00970134",
                            "AA00970309",
                            "AA00970661",
                            "AA00970743",
                            "AA00970872",
                            "AA00971265",
                            "AA00971325",
                            "AA00971347",
                            "AA00971351",
                            "AA00971833",
                            "AA00972003",
                        ]
                    )
            if len(lonely_subject_ids) > 0:
                lonely_subject_mask = df["subject_id"].isin(lonely_subject_ids)
                df_save_single_records = df[lonely_subject_mask]
                df = df[~lonely_subject_mask]
            print(f"df_save_single_records: {df_save_single_records}")

        print(f"stratification ongoing with: {stratify_str}")

        train_df = df[df["Split"] == "TRAIN"].copy()
        test_df = df[df["Split"] == "TEST"].copy()

        if print_unknown_stratification_issues and stratify_cols:
            group_sizes = train_df.groupby(stratify_cols).size()
            rare_combinations = group_sizes[group_sizes < 2].reset_index()
            rare_rows = train_df.merge(rare_combinations, on=stratify_cols, how="inner")
            print("Records with too few samples for stratification:")
            print(rare_rows)

        stratify_vals = (
            train_df[stratify_cols].astype(str).agg("_".join, axis=1)
            if stratify_cols is not None
            else None
        )
        label_col = ["conditions_PASSION", "impetig"]
        X = train_df.drop(columns=label_col + ["Split"])
        y = train_df[label_col]

        # Perform the split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, stratify=stratify_vals, random_state=seed
        )

        # Label splits
        train_split = X_train[["subject_id"]].copy()
        if not df_save_single_records.empty:
            train_split = pd.concat(
                [train_split, df_save_single_records[["subject_id"]]]
            )
        train_split["Split"] = "TRAIN"

        val_split = X_val[["subject_id"]].copy()
        val_split["Split"] = "VALIDATION"

        test_split = test_df[["subject_id", "Split"]].copy()

        # Combine all
        final_df = pd.concat([train_split, val_split, test_split], axis=0).reset_index(
            drop=True
        )
        path = f"split_dataset__{stratify_str}.csv"
        final_df.to_csv(path, index=False)
        return final_df, stratify_str, path

    # TODO: extend to do it on the picture level, not only on subject level
    def run_split_distribution_evaluation(self, create_splits: bool = False):
        interesting_cols = [
            "country",
            "sex",
            "fitzpatrick",
            "impetig",
            "conditions_PASSION",
            "ageGroup",
        ]

        splits_to_evaluate = [(self.df_split, "PASSION_split", "igored")]
        if create_splits:
            splits_to_evaluate.extend(self._create_stratified_splits())

        # Ensure output directory
        output_dir = "distribution_outputs"
        os.makedirs(output_dir, exist_ok=True)

        split_names = []
        for split, split_name, _ in splits_to_evaluate:
            print(f"Split analysis of {split_name}")

            df = self.df_labels.copy().merge(split, on="subject_id", how="left")
            df.reset_index(drop=True, inplace=True)

            df["ageGroup"] = self._generate_age_group(df)

            cols_to_check = interesting_cols.copy()
            cols_to_check.append("Split")
            df_check = df[cols_to_check]

            self._analyze_distributions(cols_to_check, df, output_dir, split_name)

            df_train = df_check[df["Split"] == "TRAIN"]
            df_validation = df_check[df["Split"] == "VALIDATION"]
            df_test = df_check[df["Split"] == "TEST"]

            def print_distribution(df_split, name, cols):
                print(f"\n--- {name.upper()} SPLIT ---")
                if df_split.empty:
                    print(f"Warning: {name.upper()} SPLIT is empty. Skipping.")
                    return
                self._analyze_distributions(
                    cols, df_split, output_dir, f"{split_name}_{name}"
                )

            print_distribution(df_train, "train", cols_to_check)
            print_distribution(df_validation, "validation", cols_to_check)
            print_distribution(df_test, "test", cols_to_check)
            split_names.append(split_name)
        return split_names

    def create_stratified_splits(self):
        splits = self._create_stratified_splits()
        split_paths = [s[2] for s in splits]
        return split_paths

    def _create_stratified_splits(self):
        df = self.df_labels.copy().merge(self.df_split, on="subject_id", how="left")
        df.reset_index(drop=True, inplace=True)
        splits = [(self._create_split(df)), (self._create_split(df, seed=32))]
        # TODO: make splitting columns configurable
        stratify_columns_sets = [
            ["conditions_PASSION", "impetig"],
            ["conditions_PASSION", "impetig", "country"],
            ["conditions_PASSION", "impetig", "fitzpatrick"],
            ["conditions_PASSION", "impetig", "country", "fitzpatrick"],
            ["conditions_PASSION", "impetig", "country", "fitzpatrick", "sex"],
        ]
        for stratify_columns_set in stratify_columns_sets:
            splits.append((self._create_split(df, stratify_columns_set)))
            splits.append((self._create_split(df, stratify_columns_set, seed=32)))
        return splits

    def _analyze_distributions(self, cols_to_check, df, output_dir=None, name=""):
        results = []
        all_counts_dict = {}
        for col in cols_to_check:
            counts = df[col].value_counts(dropna=False).sort_index()
            percentages = (counts / counts.sum() * 100).round(2)
            print(f"\nDistribution for '{col}':")
            for value, count, percent in zip(
                counts.index, counts.values, percentages.values
            ):
                print(f"{value}: {count} ({percent}%)")
                if output_dir:
                    results.append(
                        {
                            "Split": name,
                            "Column": col,
                            "Value": value,
                            "Count": count,
                            "Percent": percent,
                        }
                    )

            if output_dir:
                self._save_analysis_plot(col, counts, name, output_dir)
                all_counts_dict[col] = counts

        if output_dir:
            df_out = pd.DataFrame(results)
            csv_path = os.path.join(output_dir, f"distribution_{name}.csv")
            df_out.to_csv(csv_path, index=False)

            self._save_combined_analysis_plot(all_counts_dict, name, output_dir)

    def _save_analysis_plot(self, col, counts, name, output_dir):
        plt.figure(figsize=(8, 4))
        ax = counts.plot(kind="bar", color="#add8e6")
        self._configure_plot_axes(ax, col, counts)
        plt.title(
            f"Distribution of {col} in {name} split", fontsize=13, fontweight="bold"
        )
        self._save_plot(col, name, output_dir)

    def _save_plot(self, col, name, output_dir):
        plt.tight_layout()
        plot_path = os.path.join(output_dir, f"{name}_{col}_distribution.png")
        plt.savefig(plot_path, bbox_inches="tight")
        plt.close()

    def _save_combined_analysis_plot(self, all_counts_dict, name, output_dir):
        num_plots = len(all_counts_dict)
        cols = 2
        rows = math.ceil(num_plots / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(8 * cols, 4 * rows))

        # In case axes is 1D, convert to 2D for uniform access
        if rows == 1:
            axes = np.array([axes])
        if cols == 1:
            axes = axes.reshape(-1, 1)

        axes = axes.flatten()  # Flatten for easy indexing

        for idx, (col, counts) in enumerate(all_counts_dict.items()):
            ax = axes[idx]
            counts.plot(kind="bar", ax=ax, color="#add8e6")
            self._configure_plot_axes(ax, col, counts)
            ax.set_title(f"{col}", fontsize=12, fontweight="bold")

        # # Hide unused subplots
        for i in range(len(all_counts_dict), len(axes)):
            fig.delaxes(axes[i])

        self._save_plot("overall", name, output_dir)

    def _configure_plot_axes(self, ax, col, counts):
        label_height = max(counts.values) * 0.1
        for i, val in enumerate(counts.values):
            ax.text(
                i,
                label_height,
                str(val),
                ha="center",
                va="center",
                fontsize=9,
                color="black",
                fontweight="bold",
            )
        ax.set_xlabel(col)
        ax.set_ylabel("Count")
        ax.set_xticklabels(
            [str(label) for label in counts.index], rotation=0, ha="center"
        )


if __name__ == "__main__":
    import sys

    class Tee:
        def __init__(self, filename, mode="a"):
            self.terminal = sys.stdout
            self.log = open(filename, mode, encoding="utf-8")

        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)

        def flush(self):
            self.terminal.flush()
            self.log.flush()

    sys.stdout = Tee("split_creation_n_evaluation.log", mode="w")
    sys.stderr = sys.stdout

    evaluator = StratifiedSplitGenerator(
        passion_exp=f"experiment_stratified_validation_split_conditions"
    )
    # to only run the split creation
    # created_splits = evaluator.create_stratified_splits()
    # to run the split creation and evaluation
    created_splits = evaluator.run_split_distribution_evaluation(create_splits=True)
    print(f"created_splits: {created_splits}")

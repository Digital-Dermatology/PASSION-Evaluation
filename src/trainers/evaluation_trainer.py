import copy
import pickle
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
import scipy
import torch
import wandb
from scipy.stats import permutation_test
from sklearn.model_selection import train_test_split
from torchvision import transforms
from tqdm import tqdm

from src.cleaner.robust_cleaner import RobustCleaner
from src.trainers.eval_types.fine_tuning import EvalFineTuning
from src.trainers.eval_types.knn import EvalKNN
from src.trainers.eval_types.lightgbm import EvalLightGBM
from src.trainers.eval_types.lin import EvalLin
from src.trainers.eval_types.random_forest import EvalRandomForest
from ssl_library.src.datasets.downstream_tasks.ddi_dataset import DDILabel
from ssl_library.src.datasets.downstream_tasks.derma_compass_dataset import (
    DermaCompassLabel,
)
from ssl_library.src.datasets.downstream_tasks.fitzpatrick17_dataset import (
    FitzpatrickLabel,
)
from ssl_library.src.datasets.helper import DatasetName, get_dataset
from ssl_library.src.pkg import Embedder, embed_dataset
from ssl_library.src.utils.utils import (
    fix_random_seeds,
    latex_median_quantile,
    p_value_stars,
)

eval_type_dict = {
    "fine_tuning": EvalFineTuning,
    "kNN": EvalKNN,
    "lin": EvalLin,
    "random_forest": EvalRandomForest,
    "lightgbm": EvalLightGBM,
}


class EvaluationTrainer(object):
    def __init__(
        self,
        dataset_name: DatasetName,
        config: dict,
        test_size: float = 0.15,
        ckp_path: Optional[str] = None,
        SSL_model: str = "dino",
        output_path: Union[Path, str] = "assets/cleaning_influence",
        cache_path: Union[Path, str] = "assets/cleaning_influence/cache",
        n_layers: int = 1,
        q: float = 0.05,
        alpha: float = 0.1,
        append_to_df: bool = False,
        initialize: bool = True,
        log_wandb: bool = False,
        debug: bool = False,
    ):
        # configurations
        self.dataset_name = dataset_name
        self.config = config
        self.test_size = test_size
        self.q = q
        self.alpha = alpha
        self.output_path = Path(output_path)
        self.cache_path = Path(cache_path)
        self.initialize = initialize
        self.append_to_df = append_to_df
        self.log_wandb = log_wandb
        self.debug = debug

        self.df_name = f"cleaning_influence_exp_{self.dataset_name.value}.csv"
        self.df_path = self.output_path / self.df_name

        # make sure the output and cache path exist
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.cache_path.mkdir(parents=True, exist_ok=True)

        # parse the config to get the eval types
        self.eval_types = []
        for k, v in self.config.items():
            if k in eval_type_dict.keys():
                self.eval_types.append((eval_type_dict.get(k), v))

        # save the results to the dataframe
        self.df = pd.DataFrame(
            [],
            columns=["Score", "Train_Cleaned", "Test_Cleaned", "Seed", "EvalType"],
        )

        if append_to_df:
            if not self.df_path.exists():
                print(f"Results for dataset: {self.dataset_name.value} not available.")
            else:
                print(f"Appending results to: {self.df_path}")
                self.df = pd.read_csv(self.df_path)

        # only used when purely plotting saved results
        if not self.initialize:
            return

        # load the correct model to use as initialization
        self.model, self.model_out_dim = self.load_model(
            ckp_path=ckp_path,
            SSL_model=SSL_model,
        )

        # load the dataset to evaluate on
        self.transform = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
        label_col = None
        if dataset_name == DatasetName.FITZPATRICK17K:
            label_col = FitzpatrickLabel.HIGH
        elif dataset_name == DatasetName.DDI:
            label_col = DDILabel.MALIGNANT
        else:
            label_col = DermaCompassLabel.SECONDARY

        data_config = copy.deepcopy(config["dataset"])
        data_path = data_config[dataset_name.value].pop("path")
        self.dataset, self.torch_dataset = get_dataset(
            dataset_name=dataset_name,
            dataset_path=Path(data_path),
            batch_size=128,
            transform=self.transform,
            label_col=label_col,
            high_quality=False,
            **data_config[dataset_name.value],
        )
        # check if the cache contains the embeddings already
        cache_file = self.cache_path / f"{dataset_name.value}.pickle"
        if cache_file.exists():
            print(f"Found cached file loading: {cache_file}")
            with open(cache_file, "rb") as file:
                cached_dict = pickle.load(file)
            self.emb_space = cached_dict["embedding_space"]
            self.labels = cached_dict["labels"]
            # make sure the `q` and `alpha` match what is specified
            if cached_dict["q"] == q and cached_dict["alpha"] == alpha:
                self.issues_dup = cached_dict["issues_duplicates"]
                self.issues_irr = cached_dict["issues_irrelevants"]
                self.issues_lbl = cached_dict["issues_labels"]
            del cached_dict
        else:
            self.emb_space, self.labels, _, _ = embed_dataset(
                torch_dataset=self.torch_dataset,
                model=self.model,
                n_layers=n_layers,
                memmap=False,
                normalize=False,
            )
            del _
            # detect the data quality issues using auto mode
            data_issues = self.get_dataset_errors(q=q, alpha=alpha)
            self.issues_dup, self.issues_irr, self.issues_lbl = data_issues
            # save the embeddings and issues to cache
            save_dict = {
                "embedding_space": self.emb_space,
                "labels": self.labels,
                "q": q,
                "alpha": alpha,
                "issues_duplicates": self.issues_dup,
                "issues_irrelevants": self.issues_irr,
                "issues_labels": self.issues_lbl,
            }
            with open(cache_file, "wb") as handle:
                pickle.dump(save_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load_model(self, SSL_model: str, ckp_path: Optional[str] = None):
        if ckp_path is not None:
            model, info, _ = Embedder.load_dino(
                ckp_path=ckp_path,
                return_info=True,
                n_head_layers=0,
            )
        else:
            model, info, _ = Embedder.load_pretrained(
                SSL_model,
                return_info=True,
                n_head_layers=0,
            )
        # set the model in evaluation mode
        model = model.eval()
        # move to correct device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        return model, info.out_dim

    def get_dataset_errors(
        self,
        q: float = 0.05,
        alpha: float = 0.1,
        chunk_size: int = 100,
    ):
        # apply SelfClean
        cleaner = RobustCleaner(
            emb_space=self.emb_space,
            labels=self.labels,
            class_labels=self.dataset.classes,
            memmap=False,
            global_leaves=False,
            chunk_size=chunk_size,
        )
        ret = cleaner.inspect_dataset(auto_cleaning=False)

        # perform automatic cleaning
        issues_dup = cleaner.fraction_cut(
            scores=ret["near_duplicates"]["scores"],
            alpha=alpha,
            q=q,
            plot_result=False,
            debug=True,
        )
        issues_ood = cleaner.fraction_cut(
            scores=ret["irrelevants"]["scores"],
            alpha=alpha,
            q=q,
            plot_result=False,
            debug=True,
        )
        issues_lbl = cleaner.fraction_cut(
            scores=ret["label_errors"]["scores"],
            alpha=alpha,
            q=q,
            plot_result=False,
            debug=True,
        )

        issues_dup = ret["near_duplicates"]["indices"][issues_dup]
        issues_ood = ret["irrelevants"]["indices"][issues_ood]
        issues_lbl = ret["label_errors"]["indices"][issues_lbl]

        issues_dup = issues_dup.astype(int)
        issues_ood = issues_ood.astype(int)
        issues_lbl = issues_lbl.astype(int)
        if self.debug:
            print(issues_dup.shape, issues_ood.shape, issues_lbl.shape)
        return issues_dup, issues_ood, issues_lbl

    def run_cleaning_influence(self):
        if self.df_path.exists() and not self.append_to_df:
            raise ValueError(
                f"Dataframe already exists, remove to start: {self.df_name}"
            )
        self.dataset.return_path = False

        for e_type, config in self.eval_types:
            e_name = e_type.name()
            n_repeats = config["n_repeats"]
            for seed in tqdm(np.arange(1, n_repeats + 1)):
                # make sure the seeds are fixed
                fix_random_seeds(seed)
                for clean_train in [False, True]:
                    for clean_test in [False, True]:
                        dataset_range = np.arange(len(self.emb_space))
                        train_range, test_range = train_test_split(
                            dataset_range,
                            test_size=self.test_size,
                            stratify=self.labels,
                            random_state=seed,
                        )
                        # make sure there is no introduced train/test leak of known duplicates!
                        train_range, test_range = self.clean_dataset_ranges(
                            train_range=train_range,
                            test_range=test_range,
                            clean_train=clean_train,
                            clean_test=clean_test,
                        )
                        # W&B configurations
                        if e_type is EvalFineTuning and self.log_wandb:
                            _config = copy.deepcopy(self.config)
                            _config["clean_train"] = clean_train
                            _config["clean_test"] = clean_test
                            _config["test_size"] = self.test_size
                            _config["cleaning_q"] = self.q
                            _config["cleaning_alpha"] = self.alpha
                            wandb.init(
                                config=_config,
                                project="SelfClean-CleaningInfluence",
                            )
                            wandb.run.name = (
                                f"{self.dataset_name.value}-{wandb.run.name}"
                            )
                            wandb.run.save()

                        # get train / test set
                        score = e_type.evaluate(
                            emb_space=self.emb_space,
                            labels=self.labels,
                            train_range=train_range,
                            test_range=test_range,
                            # only needed for fine-tuning
                            dataset=self.dataset,
                            model=self.model,
                            model_out_dim=self.model_out_dim,
                            log_wandb=self.log_wandb,
                            # rest of the method specific parameters set with kwargs
                            **config,
                        )
                        # finish the W&B run if needed
                        if e_type is EvalFineTuning and self.log_wandb:
                            wandb.finish()
                        # save the results to the overall dataframe
                        self.df.loc[len(self.df)] = [
                            score,
                            clean_train,
                            clean_test,
                            seed,
                            e_name,
                        ]
                # save the dataframe (will work even bug in latter datasets)
                self.df.to_csv(self.df_path, index=False)

    def clean_dataset_ranges(
        self,
        train_range: list,
        test_range: list,
        clean_train: bool,
        clean_test: bool,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if clean_train and clean_test:
            cleaning_str = "BOTH"
        elif clean_train:
            cleaning_str = "TRAIN"
        elif clean_test:
            cleaning_str = "TEST"
        else:
            cleaning_str = None

        if cleaning_str is not None:
            issues_ood_selection = [
                EvaluationTrainer.selection_on_split(
                    idx1=x[0],
                    idx2=x[1],
                    cleaning=cleaning_str,
                    train_range=train_range,
                    test_range=test_range,
                )
                for x in self.issues_dup
            ]

        if clean_train:
            if self.debug:
                print(f"(Train) Before cleaning: {len(train_range)}")
            train_range = np.asarray(
                [x for x in train_range if x not in issues_ood_selection]
            )
            train_range = np.asarray(
                [x for x in train_range if x not in self.issues_dup]
            )
            train_range = np.asarray(
                [x for x in train_range if x not in self.issues_lbl]
            )
            if self.debug:
                print(f"(Train) After cleaning: {len(train_range)}")

        if clean_test:
            if self.debug:
                print(f"(Test) Before cleaning: {len(test_range)}")
            test_range = np.asarray(
                [x for x in test_range if x not in issues_ood_selection]
            )
            test_range = np.asarray([x for x in test_range if x not in self.issues_dup])
            test_range = np.asarray([x for x in test_range if x not in self.issues_lbl])
            if self.debug:
                print(f"(Test) After cleaning: {len(test_range)}")
        return train_range, test_range

    def print_results(self):
        # Try to load the results
        if not self.df_path.exists():
            print(f"Results for dataset: {self.dataset_name.value} not available.")
        else:
            self.df = pd.read_csv(self.df_path)

        # print the values and perform significance test
        rng = np.random.default_rng(seed=42)
        print(f"{'-'*20} {self.dataset_name.value} {'-'*20}")
        for model_type in self.df["EvalType"].unique():
            print(f"{'*'*20} {model_type} {'*'*20}")
            l_diffs_test_cleaning = []
            l_diffs_train_cleaning = []
            for seed in self.df[self.df["EvalType"] == model_type]["Seed"].unique():
                train_dirty_test_dirty = self.df.loc[
                    (self.df["EvalType"] == model_type)
                    & (self.df["Seed"] == seed)
                    & (self.df["Train_Cleaned"] == False)
                    & (self.df["Test_Cleaned"] == False),
                    "Score",
                ]
                train_dirty_test_clean = self.df.loc[
                    (self.df["EvalType"] == model_type)
                    & (self.df["Seed"] == seed)
                    & (self.df["Train_Cleaned"] == False)
                    & (self.df["Test_Cleaned"] == True),
                    "Score",
                ]
                train_clean_test_clean = self.df.loc[
                    (self.df["EvalType"] == model_type)
                    & (self.df["Seed"] == seed)
                    & (self.df["Train_Cleaned"] == True)
                    & (self.df["Test_Cleaned"] == True),
                    "Score",
                ]
                l_diffs_test_cleaning.append(
                    train_dirty_test_clean.iloc[0] - train_dirty_test_dirty.iloc[0]
                )
                l_diffs_train_cleaning.append(
                    train_clean_test_clean.iloc[0] - train_dirty_test_clean.iloc[0]
                )
            l_diffs_test_cleaning = np.asarray(l_diffs_test_cleaning)
            l_diffs_train_cleaning = np.asarray(l_diffs_train_cleaning)

            train_dirty_test_dirty = self.df.loc[
                (self.df["EvalType"] == model_type)
                & (self.df["Train_Cleaned"] == False)
                & (self.df["Test_Cleaned"] == False),
                "Score",
            ]
            train_dirty_test_clean = self.df.loc[
                (self.df["EvalType"] == model_type)
                & (self.df["Train_Cleaned"] == False)
                & (self.df["Test_Cleaned"] == True),
                "Score",
            ]
            train_clean_test_clean = self.df.loc[
                (self.df["EvalType"] == model_type)
                & (self.df["Train_Cleaned"] == True)
                & (self.df["Test_Cleaned"] == True),
                "Score",
            ]

            p_value = scipy.stats.wilcoxon(
                train_dirty_test_dirty,
                train_dirty_test_clean,
            ).pvalue

            pvalue = permutation_test(
                (train_dirty_test_dirty, train_dirty_test_clean),
                EvaluationTrainer.test_statistic,
                vectorized=True,
                alternative="two-sided",
                random_state=rng,
                permutation_type="samples",
            ).pvalue
            print(
                f"Difference Test Cleaning: {latex_median_quantile(l_diffs_test_cleaning)} \ "
                f"^{p_value_stars(p_value)} (p_p: {round(pvalue, 4)}) (p_wc: {round(p_value, 4)})"
            )

            p_value = scipy.stats.wilcoxon(
                train_clean_test_clean,
                train_dirty_test_clean,
                alternative="greater",
            ).pvalue
            pvalue = permutation_test(
                (train_clean_test_clean, train_dirty_test_clean),
                EvaluationTrainer.test_statistic,
                vectorized=True,
                alternative="greater",
                random_state=rng,
                permutation_type="samples",
            ).pvalue
            print(
                f"Difference Train Cleaning: {latex_median_quantile(l_diffs_train_cleaning)} \ "
                f"^{p_value_stars(p_value)} (p_p: {round(pvalue, 4)}) (p_wc: {round(p_value, 4)})"
            )

            print(
                f"Cont+Cont: {latex_median_quantile(train_dirty_test_dirty.values)}, "
                f"Cont+Clean: {latex_median_quantile(train_dirty_test_clean.values)}, "
                f"Clean+Clean: {latex_median_quantile(train_clean_test_clean.values)}, "
            )
            print(
                f"${latex_median_quantile(train_dirty_test_dirty.values)}$ & "
                f"${latex_median_quantile(train_dirty_test_clean.values)}$ & "
                f"${latex_median_quantile(train_clean_test_clean.values)}$"
            )
            print()

    @staticmethod
    def train_or_test_split(idx: int, train_range: list, test_range: list):
        if idx in train_range:
            return "TRAIN"
        elif idx in test_range:
            return "TEST"
        else:
            return "ERROR"

    @staticmethod
    def selection_on_split(
        idx1: int, idx2: int, cleaning: str, train_range: list, test_range: list
    ):
        idx1_set = EvaluationTrainer.train_or_test_split(
            idx=idx1,
            train_range=train_range,
            test_range=test_range,
        )
        idx2_set = EvaluationTrainer.train_or_test_split(
            idx=idx2,
            train_range=train_range,
            test_range=test_range,
        )

        if idx1_set == idx2_set:
            return np.random.choice([idx1, idx2])
        elif idx1_set == cleaning:
            return idx1
        elif idx2_set == cleaning:
            return idx2
        else:
            return np.random.choice([idx1, idx2])

    @staticmethod
    def test_statistic(x, y, axis):
        r = scipy.stats.rankdata(x, axis=axis)
        d = x - y
        r_plus = np.sum((d > 0) * r, axis=axis)
        r_minus = np.sum((d < 0) * r, axis=axis)
        return r_plus

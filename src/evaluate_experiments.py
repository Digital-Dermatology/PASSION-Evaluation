import argparse
import copy
from pathlib import Path

import yaml

from src.datasets.helper import DatasetName
from src.trainers.experiment_age_group_generalization import (
    ExperimentAgeGroupGeneralization,
)
from src.trainers.experiment_center_generalization import ExperimentCenterGeneralization
from src.trainers.experiment_standard_split import ExperimentStandardSplit
from src.trainers.experiment_stratified_validation_split import (
    ExperimentStratifiedValidationSplit,
)
from src.utils.loader import Loader

my_parser = argparse.ArgumentParser(description="Experiments for the PASSION paper.")
my_parser.add_argument(
    "--config_path",
    type=str,
    required=True,
    help="Path to the config yaml.",
)
my_parser.add_argument(
    "--exp1",
    action="store_true",
    help="If the experiment 1, i.e. differential diagnosis, should be run.",
)
my_parser.add_argument(
    "--exp2",
    action="store_true",
    help="If the experiment 2, i.e. detecting impetigo, should be run.",
)
my_parser.add_argument(
    "--exp3",
    action="store_true",
    help="If the experiment 3, i.e. generalization collection centers, should be run.",
)
my_parser.add_argument(
    "--exp4",
    action="store_true",
    help="If the experiment 4, i.e. generalization age groups, should be run.",
)
my_parser.add_argument(
    "--exp5",
    action="store_true",
    help="If the experiment 5 should be run - differential diagnosis with different split files (no folds, using validation set instead of test set).",
)
my_parser.add_argument(
    "--exp6",
    action="store_true",
    help="If the experiment 6 should be run - differential diagnosis with different split files (5-fold cross-validation, using validation set instead of test set).",
)
my_parser.add_argument(
    "--exp7",
    action="store_true",
    help="If the experiment 7 should be run - evaluate differential diagnosis models trained with exp5/6 on the original test set",
)

my_parser.add_argument(
    "--append_results",
    action="store_true",
    help="If the results should be appended to the existing df (needs to be used with care!)",
)
args = my_parser.parse_args()

if __name__ == "__main__":
    # load config yaml
    args.config_path = Path(args.config_path)
    if not args.config_path.exists():
        raise ValueError(f"Unable to find config yaml file: {args.config_path}")
    config = yaml.load(open(args.config_path, "r"), Loader=Loader)
    # overall parameters used for all datasets
    log_wandb = config.pop("log_wandb")
    model = config.pop("model")
    if args.exp1:
        trainer = ExperimentStandardSplit(
            dataset_name=DatasetName.PASSION,
            config=config,
            SSL_model=model,
            append_to_df=args.append_results,
            log_wandb=log_wandb,
            add_info="conditions",
        )
        trainer.evaluate()

    if args.exp2:
        _config = copy.deepcopy(config)
        _config["dataset"]["passion"]["label_col"] = "IMPETIGO"
        trainer = ExperimentStandardSplit(
            dataset_name=DatasetName.PASSION,
            config=_config,
            SSL_model=model,
            append_to_df=args.append_results,
            log_wandb=log_wandb,
            add_info="impetigo",
        )
        trainer.evaluate()

    if args.exp3:
        trainer = ExperimentCenterGeneralization(
            dataset_name=DatasetName.PASSION,
            config=config,
            SSL_model=model,
            append_to_df=args.append_results,
            log_wandb=log_wandb,
        )
        trainer.evaluate()

    if args.exp4:
        trainer = ExperimentAgeGroupGeneralization(
            dataset_name=DatasetName.PASSION,
            config=config,
            SSL_model=model,
            append_to_df=args.append_results,
            log_wandb=log_wandb,
        )
        trainer.evaluate()

    if args.exp5 or args.exp6 or args.exp7:
        evaluator = StratifiedSplitGenerator(
            passion_exp=f"experiment_stratified_validation_split_conditions"
        )
        splits = evaluator.create_stratified_splits()
        print(f"splits: {splits}")

    if args.exp5:
        _config = copy.deepcopy(config)
        _config["fine_tuning"]["n_folds"] = None
        _config["fine_tuning"]["train"] = True

        for split_name in splits:
            _config["dataset"]["passion"]["split_file"] = f"{split_name}.csv"
            trainer = ExperimentStratifiedValidationSplit(
                dataset_name=DatasetName.PASSION,
                config=_config,
                SSL_model=model,
                append_to_df=args.append_results,
                log_wandb=log_wandb,
                add_info=f"conditions__{split_name}",
            )
            trainer.evaluate()

    if args.exp6:
        _config = copy.deepcopy(config)
        _config["fine_tuning"]["n_folds"] = 5
        _config["fine_tuning"]["train"] = True
        for split_name in splits:
            _config["dataset"]["passion"]["split_file"] = f"{split_name}.csv"
            trainer = ExperimentStratifiedValidationSplit(
                dataset_name=DatasetName.PASSION,
                config=_config,
                SSL_model=model,
                append_to_df=args.append_results,
                log_wandb=log_wandb,
                add_info=f"conditions__{split_name}",
            )
            trainer.evaluate()

    if args.exp7:
        _config = copy.deepcopy(config)
        _config["fine_tuning"]["n_folds"] = None
        _config["fine_tuning"]["train"] = False
        for split_name in splits:
            trainer = ExperimentStandardSplit(
                dataset_name=DatasetName.PASSION,
                config=_config,
                SSL_model=model,
                model_path=f"experiment_stratified_validation_split_conditions__{split_name}__passion__{model}",
                append_to_df=args.append_results,
                log_wandb=log_wandb,
                add_info=f"conditions__model_trained_with_{split_name}",
            )
            trainer.evaluate()

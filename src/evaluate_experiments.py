import argparse
import copy
from pathlib import Path

import yaml

from src.trainers.experiment_age_group_generalization import (
    ExperimentAgeGroupGeneralization,
)
from src.trainers.experiment_center_generalization import ExperimentCenterGeneralization
from src.trainers.experiment_standard_split import ExperimentStandardSplit
from ssl_library.src.datasets.helper import DatasetName
from ssl_library.src.utils.loader import Loader

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

    if args.exp1:
        trainer = ExperimentStandardSplit(
            dataset_name=DatasetName.PASSION,
            config=config,
            SSL_model="imagenet",
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
            SSL_model="imagenet",
            append_to_df=args.append_results,
            log_wandb=log_wandb,
            add_info="impetigo",
        )
        trainer.evaluate()

    if args.exp3:
        trainer = ExperimentCenterGeneralization(
            dataset_name=DatasetName.PASSION,
            config=config,
            SSL_model="imagenet",
            append_to_df=args.append_results,
            log_wandb=log_wandb,
        )
        trainer.evaluate()

    if args.exp4:
        trainer = ExperimentAgeGroupGeneralization(
            dataset_name=DatasetName.PASSION,
            config=config,
            SSL_model="imagenet",
            append_to_df=args.append_results,
            log_wandb=log_wandb,
        )
        trainer.evaluate()

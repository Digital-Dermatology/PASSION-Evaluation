import argparse
from pathlib import Path

import yaml

from src.trainers.evaluation_trainer import EvaluationTrainer
from ssl_library.src.datasets.helper import DatasetName
from ssl_library.src.utils.loader import Loader

my_parser = argparse.ArgumentParser(
    description="Investigates the influence of dataset cleaning on finetuned performance."
)
my_parser.add_argument(
    "--config_path",
    type=str,
    required=True,
    help="Path to the config yaml.",
)
my_parser.add_argument(
    "--print_results_only",
    action="store_true",
    help="If the results should only be printed and no training is performed.",
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
    # initialize the trainer
    trainer = EvaluationTrainer(
        dataset_name=DatasetName.PASSION,
        config=config,
        SSL_model="imagenet",
        append_to_df=args.append_results,
        initialize=False if args.print_results_only else True,
        log_wandb=log_wandb,
    )
    if not args.print_results_only:
        trainer.evaluate()
    trainer.print_results()

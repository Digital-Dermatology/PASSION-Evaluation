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
    "--datasets",
    nargs="+",
    default=[
        "DDI",
        "ham10000",
        "fitzpatrick17k",
        "FOOD_101",
        "ImageNet-1k",
    ],
    help="Name of the datasets to evaluate on.",
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
    test_size = config.pop("test_size")
    log_wandb = config.pop("log_wandb")
    # cleaning parameters used for all datasets
    q = config.pop("q")
    alpha = config.pop("alpha")
    # define the models for each dataset
    model_dict = {
        DatasetName.DDI: "models/DDI/DINO/checkpoint-epoch500.pth",
        DatasetName.HAM10000: "models/HAM10000/DINO/checkpoint-epoch500.pth",
        DatasetName.FITZPATRICK17K: "models/fitzpatrick17k/DINO/checkpoint-epoch500.pth",
        DatasetName.FOOD_101: "models/Food101N/DINO/checkpoint-epoch100.pth",
        DatasetName.IMAGENET_1K: "models/ImageNet-1k/DINO/checkpoint-epoch100.pth",
    }
    # loop over all given datasets
    for dataset_name in args.datasets:
        # initialize the trainer
        dataset_name = DatasetName(dataset_name)
        trainer = EvaluationTrainer(
            dataset_name=dataset_name,
            config=config,
            test_size=test_size,
            ckp_path=model_dict.get(dataset_name),
            q=q,
            alpha=alpha,
            append_to_df=args.append_results,
            initialize=False if args.print_results_only else True,
            log_wandb=log_wandb,
        )
        if not args.print_results_only:
            trainer.run_cleaning_influence()
        trainer.print_results()

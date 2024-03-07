from pathlib import Path
from typing import Iterator, Optional, Tuple, Union

import numpy as np

from src.trainers.evaluation_trainer import EvaluationTrainer
from ssl_library.src.datasets.helper import DatasetName


class ExperimentAgeGroupGeneralization(EvaluationTrainer):
    def __init__(
        self,
        dataset_name: DatasetName,
        config: dict,
        ckp_path: Optional[str] = None,
        SSL_model: str = "imagenet",
        output_path: Union[Path, str] = "assets/evaluation",
        cache_path: Union[Path, str] = "assets/evaluation/cache",
        n_layers: int = 1,
        append_to_df: bool = False,
        wandb_project_name: str = "PASSION-Evaluation",
        log_wandb: bool = False,
    ):
        super().__init__(
            dataset_name=dataset_name,
            config=config,
            ckp_path=ckp_path,
            SSL_model=SSL_model,
            output_path=output_path,
            cache_path=cache_path,
            n_layers=n_layers,
            append_to_df=append_to_df,
            wandb_project_name=wandb_project_name,
            log_wandb=log_wandb,
        )

    @property
    def experiment_name(self) -> str:
        return "experiment_age"

    def split_dataframe_iterator(self) -> Iterator[Tuple[np.ndarray, np.ndarray, str]]:
        train_valid_pediatric_range = self.dataset.meta_data[
            (self.dataset.meta_data["Split"] == "TRAIN")
            & (self.dataset.meta_data["age"] <= 16)
        ].index.values
        train_valid_full_pediatric_range = (
            self.dataset.meta_data[self.dataset.meta_data["Split"] == "TRAIN"]
            .sample(n=len(train_valid_pediatric_range), random_state=self.seed)
            .index.values
        )

        train_valid_adolescent_range = self.dataset.meta_data[
            (self.dataset.meta_data["Split"] == "TRAIN")
            & (self.dataset.meta_data["age"] > 16)
        ].index.values
        train_valid_full_adolescent_range = (
            self.dataset.meta_data[self.dataset.meta_data["Split"] == "TRAIN"]
            .sample(n=len(train_valid_adolescent_range), random_state=self.seed)
            .index.values
        )

        test_range = self.dataset.meta_data[
            self.dataset.meta_data["Split"] == "TEST"
        ].index.values

        yield train_valid_pediatric_range, test_range, "TRAIN: Pediatric(<=16), TEST: Standard"
        yield train_valid_full_pediatric_range, test_range, "TRAIN: Standard (resampled: Pediatric), TEST: Standard"

        yield train_valid_adolescent_range, test_range, "TRAIN: Adolescent (>16), TEST: Standard"
        yield train_valid_full_adolescent_range, test_range, "TRAIN: Standard (resampled: Adolescent), TEST: Standard"

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
        initialize: bool = True,
        wandb_project_name: str = "PASSION-Evaluation",
        log_wandb: bool = False,
        debug: bool = False,
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
            initialize=initialize,
            wandb_project_name=wandb_project_name,
            log_wandb=log_wandb,
            debug=debug,
        )

    @property
    def experiment_name(self) -> str:
        return "experiment_3"

    def split_dataframe_iterator(self) -> Iterator[Tuple[np.ndarray, np.ndarray, str]]:
        pediatric_indices = self.dataset.meta_data[
            self.dataset.meta_data["age"] <= 16
        ].index.values
        adolescent_indices = self.dataset.meta_data[
            self.dataset.meta_data["age"] > 16
        ].index.values
        yield pediatric_indices, adolescent_indices, "TRAIN: Pediatric(<=16), TEST: Adolescent (>16)"
        yield adolescent_indices, pediatric_indices, "TRAIN: Adolescent (>16), TEST: Pediatric(<=16)"

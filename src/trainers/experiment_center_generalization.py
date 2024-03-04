from itertools import combinations
from pathlib import Path
from typing import Iterator, Optional, Tuple, Union

import numpy as np

from src.trainers.evaluation_trainer import EvaluationTrainer
from ssl_library.src.datasets.helper import DatasetName


class ExperimentCenterGeneralization(EvaluationTrainer):
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
        return "experiment_2"

    def split_dataframe_iterator(self) -> Iterator[Tuple[np.ndarray, np.ndarray, str]]:
        l_countries = set(self.dataset.meta_data["country"].unique())
        data_combinations = [
            {"train": list(x), "test": list(set(x) ^ l_countries)}
            for x in list(combinations(l_countries, 2))
        ]

        for split_dict in data_combinations:
            train_valid_range = self.dataset.meta_data[
                self.dataset.meta_data["country"].isin(split_dict["train"])
            ].index.values
            test_range = self.dataset.meta_data[
                self.dataset.meta_data["country"].isin(split_dict["test"])
            ].index.values
            split_name = f"TRAIN: {'_'.join(split_dict['train'])}, TEST: {'_'.join(split_dict['test'])}"
            yield train_valid_range, test_range, split_name

import copy
import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import wandb
from sklearn.model_selection import StratifiedGroupKFold
from torchvision import transforms
from tqdm import tqdm

from src.trainers.eval_types.base import BaseEvalType
from src.trainers.eval_types.fine_tuning import EvalFineTuning
from src.trainers.eval_types.knn import EvalKNN
from src.trainers.eval_types.lin import EvalLin
from ssl_library.src.datasets.helper import DatasetName, get_dataset
from ssl_library.src.pkg import Embedder, embed_dataset
from ssl_library.src.utils.utils import fix_random_seeds

eval_type_dict = {
    "fine_tuning": EvalFineTuning,
    "kNN": EvalKNN,
    "lin": EvalLin,
}


class EvaluationTrainer(ABC, object):
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
        debug: bool = False,
    ):
        self.dataset_name = dataset_name
        self.config = config
        self.output_path = Path(output_path)
        self.cache_path = Path(cache_path)
        self.append_to_df = append_to_df
        self.wandb_project_name = wandb_project_name
        self.log_wandb = log_wandb
        self.debug = debug
        self.seed = config["seed"]
        fix_random_seeds(self.seed)

        self.df_name = f"passion_{self.experiment_name}_{self.dataset_name.value}.csv"
        self.df_path = self.output_path / self.df_name
        self.model_path = self.output_path / self.experiment_name

        # make sure the output and cache path exist
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.cache_path.mkdir(parents=True, exist_ok=True)
        self.model_path.mkdir(parents=True, exist_ok=True)

        # parse the config to get the eval types
        self.eval_types = []
        for k, v in self.config.items():
            if k in eval_type_dict.keys():
                self.eval_types.append((eval_type_dict.get(k), v))

        # save the results to the dataframe
        self.df = pd.DataFrame(
            [],
            columns=[
                "Score",
                "EvalTargets",
                "EvalPredictions",
                "EvalType",
                "AdditionalRunInfo",
                "SplitName",
            ],
        )
        if append_to_df:
            if not self.df_path.exists():
                print(f"Results for dataset: {self.dataset_name.value} not available.")
            else:
                print(f"Appending results to: {self.df_path}")
                self.df = pd.read_csv(self.df_path)

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
        data_config = copy.deepcopy(config["dataset"])
        data_path = data_config[dataset_name.value].pop("path")
        self.dataset, self.torch_dataset = get_dataset(
            dataset_name=dataset_name,
            dataset_path=Path(data_path),
            batch_size=config.get("batch_size", 128),
            transform=self.transform,
            **data_config[dataset_name.value],
        )
        # check if the cache contains the embeddings already
        cache_file = (
            self.cache_path / f"{dataset_name.value}_{self.experiment_name}.pickle"
        )
        if cache_file.exists():
            print(f"Found cached file loading: {cache_file}")
            with open(cache_file, "rb") as file:
                cached_dict = pickle.load(file)
            self.emb_space = cached_dict["embedding_space"]
            self.labels = cached_dict["labels"]
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
            # save the embeddings and issues to cache
            save_dict = {
                "embedding_space": self.emb_space,
                "labels": self.labels,
            }
            with open(cache_file, "wb") as handle:
                pickle.dump(save_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @property
    @abstractmethod
    def experiment_name(self) -> str:
        pass

    @abstractmethod
    def split_dataframe_iterator(self) -> Iterator[Tuple[np.ndarray, np.ndarray, str]]:
        pass

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

    def evaluate(self):
        if self.df_path.exists() and not self.append_to_df:
            raise ValueError(
                f"Dataframe already exists, remove to start: {self.df_path}"
            )
        self.dataset.return_path = False

        for e_type, config in self.eval_types:
            for (
                train_valid_range,
                test_range,
                split_name,
            ) in self.split_dataframe_iterator():
                if config.get("n_folds", None) is not None:
                    k_fold = StratifiedGroupKFold(
                        n_splits=config["n_folds"],
                        random_state=self.seed,
                        shuffle=True,
                    )
                    labels = self.dataset.meta_data.loc[
                        train_valid_range, self.dataset.LBL_COL
                    ].values
                    groups = self.dataset.meta_data.loc[
                        train_valid_range, "subject_id"
                    ].values
                    fold_generator = k_fold.split(train_valid_range, labels, groups)
                    for i_fold, (train_range, valid_range) in tqdm(
                        enumerate(fold_generator),
                        total=config["n_folds"],
                        desc="K-Folds",
                    ):
                        self._run_evaluation_on_range(
                            e_type=e_type,
                            train_range=train_range,
                            eval_range=valid_range,
                            config=config,
                            add_run_info=f"Fold-{i_fold}",
                            split_name=split_name,
                            saved_model_path=None,
                        )
                if config["eval_test_performance"]:
                    # TODO: Non-informed baselines?
                    self._run_evaluation_on_range(
                        e_type=e_type,
                        train_range=train_valid_range,
                        eval_range=test_range,
                        config=config,
                        add_run_info="Test",
                        split_name=split_name,
                        saved_model_path=self.model_path,
                    )

    def _run_evaluation_on_range(
        self,
        e_type: BaseEvalType,
        train_range: np.ndarray,
        eval_range: np.ndarray,
        config: dict,
        add_run_info: Optional[str] = None,
        split_name: Optional[str] = None,
        saved_model_path: Union[Path, str, None] = None,
    ):
        # W&B configurations
        if e_type is EvalFineTuning and self.log_wandb:
            _config = copy.deepcopy(self.config)
            if split_name is not None:
                _config["split_name"] = split_name
            wandb.init(
                config=_config,
                project=self.wandb_project_name,
            )
            wandb_run_name = f"{self.experiment_name}-{wandb.run.name}"
            if add_run_info is not None:
                wandb_run_name += f"-{add_run_info}"
            wandb.run.name = wandb_run_name
            wandb.run.save()
        # get train / test set
        score_dict = e_type.evaluate(
            emb_space=self.emb_space,
            labels=self.labels,
            train_range=train_range,
            evaluation_range=eval_range,
            # only needed for fine-tuning
            dataset=self.dataset,
            model=self.model,
            model_out_dim=self.model_out_dim,
            log_wandb=self.log_wandb,
            saved_model_path=saved_model_path,
            # rest of the method specific parameters set with kwargs
            **config,
        )
        # finish the W&B run if needed
        if e_type is EvalFineTuning and self.log_wandb:
            wandb.finish()
        # save the results to the overall dataframe + save df
        self.df.loc[len(self.df)] = list(score_dict.values()) + [
            split_name,
            add_run_info,
            e_type.name(),
        ]
        self.df.to_csv(self.df_path, index=False)

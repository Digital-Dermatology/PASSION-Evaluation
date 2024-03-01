import copy
import pickle
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import torch
import wandb
from sklearn.model_selection import KFold
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


class EvaluationTrainer(object):
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
        log_wandb: bool = False,
        debug: bool = False,
    ):
        # configurations
        self.dataset_name = dataset_name
        self.config = config
        self.output_path = Path(output_path)
        self.cache_path = Path(cache_path)
        self.initialize = initialize
        self.append_to_df = append_to_df
        self.log_wandb = log_wandb
        self.debug = debug

        self.seed = config["seed"]
        fix_random_seeds(self.seed)

        self.df_name = f"passion_exp_{self.dataset_name.value}.csv"
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
        self.df = pd.DataFrame([], columns=["Score", "Seed", "EvalType"])
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
        data_config = copy.deepcopy(config["dataset"])
        data_path = data_config[dataset_name.value].pop("path")
        self.dataset, self.torch_dataset = get_dataset(
            dataset_name=dataset_name,
            dataset_path=Path(data_path),
            batch_size=128,
            transform=self.transform,
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
            train_valid_range = self.dataset.meta_data[
                self.dataset.meta_data["Split"] == "TRAIN"
            ].index.values
            test_range = self.dataset.meta_data[
                self.dataset.meta_data["Split"] == "TEST"
            ].index.values

            k_fold = KFold(
                n_splits=config["n_folds"],
                random_state=self.seed,
                shuffle=True,
            )
            fold_generator = k_fold.split(train_valid_range)
            for i_fold, (train_range, valid_range) in tqdm(
                enumerate(fold_generator),
                total=config["n_folds"],
            ):
                self._run_evaluation_on_range(
                    e_type=e_type,
                    train_range=train_range,
                    eval_range=valid_range,
                    config=config,
                    add_run_info=f"Fold-{i_fold}",
                )

            if config["eval_test_performance"]:
                self._run_evaluation_on_range(
                    e_type=e_type,
                    train_range=train_valid_range,
                    eval_range=test_range,
                    config=config,
                    add_run_info="Test",
                )

    def _run_evaluation_on_range(
        self,
        e_type: BaseEvalType,
        train_range: np.ndarray,
        eval_range: np.ndarray,
        config: dict,
        add_run_info: Optional[str] = None,
    ):
        # W&B configurations
        if e_type is EvalFineTuning and self.log_wandb:
            wandb.init(
                config=self.config,
                project="PASSION-Evaluation",
            )
            wandb_run_name = f"{self.dataset_name.value}-{wandb.run.name}"
            if add_run_info is not None:
                wandb_run_name += f"-{add_run_info}"
            wandb.run.name = wandb_run_name
            wandb.run.save()

        # get train / test set
        score = e_type.evaluate(
            emb_space=self.emb_space,
            labels=self.labels,
            train_range=train_range,
            evaluation_range=eval_range,
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
            add_run_info,
            e_type.name(),
        ]
        # save the dataframe (will work even bug in latter datasets)
        self.df.to_csv(self.df_path, index=False)

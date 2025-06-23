import re
from enum import Enum
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd
import torch
from loguru import logger
from PIL import Image, ImageFile

from src.datasets.generic_image_dataset import GenericImageDataset

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True


class PASSIONLabel(Enum):
    CONDITIONS = "conditions_PASSION"
    IMPETIGO = "impetig"


def extract_subject_id(path: str):
    pattern = r"([A-Za-z]+[0-9]+)"
    match = re.search(pattern, path)
    if match:
        return str(match.group(1)).strip()
    else:
        return np.nan


class PASSIONDataset(GenericImageDataset):
    """PASSION dataset."""

    IMG_COL = "img_path"
    LBL_COL = None

    def __init__(
        self,
        dataset_dir: Union[str, Path] = "data/PASSION/",
        meta_data_file: Union[str, Path] = "label.csv",
        split_file: Union[str, Path, None] = None,
        transform=None,
        val_transform=None,
        label_col: Union[PASSIONLabel, str] = PASSIONLabel.CONDITIONS,
        image_extensions: Sequence = (
            "*.jpeg",
            "*.jpg",
            "*.JPG",
            "*.JPEG",
            "*.PNG",
            "*.png",
        ),
        pre_computed_embeddings_path: Optional[Union[str, Path]] = None,
        return_embedding: bool = False,
        **kwargs,
    ):
        """
        Initialize the dataset.

        Sets the correct path for the needed arguments.

        Parameters
        ----------
        dataset_dir : str
            Directory with all the images.
        transform : Union[callable, optional]
            Optional transform to be applied to the images.
        val_transform : Union[callable, optional]
            Optional transform to be applied to the images when in validation mode.
        """
        if isinstance(label_col, str):
            label_col = PASSIONLabel[label_col]
        self.LBL_COL = label_col.value
        super().__init__(
            dataset_dir=dataset_dir,
            transform=transform,
            val_transform=val_transform,
            image_extensions=image_extensions,
            **kwargs,
        )
        meta_data_file = self.check_path(self.dataset_dir / meta_data_file)
        self.meta_data["subject_id"] = self.meta_data.img_path.apply(extract_subject_id)
        # get the labels of meta-data `PASSION_Files`
        passion_meta_data = pd.read_csv(meta_data_file, index_col=0)
        self.LBL_COL = self.LBL_COL.replace("lbl_", "")
        self.meta_data = self.meta_data.drop(
            columns=[self.LBL_COL, f"lbl_{self.LBL_COL}"], axis=1
        ).merge(passion_meta_data, on=["subject_id"], how="inner")
        # fill the `impetig` column
        impetigo_mapper = {0.0: "not impetiginized", 1.0: "impetiginized"}
        self.meta_data["impetig"] = self.meta_data["impetig"].fillna(value=0.0)
        self.meta_data["impetig"] = self.meta_data["impetig"].apply(impetigo_mapper.get)
        # get the splitting type
        if split_file is not None:
            split_file = self.check_path(self.dataset_dir / split_file)
            df_split = pd.read_csv(split_file)
            self.meta_data = self.meta_data.merge(df_split, on="subject_id", how="left")
            self.meta_data.reset_index(drop=True, inplace=True)
            del df_split

        self.meta_data.reset_index(drop=True, inplace=True)
        int_lbl, lbl_mapping = pd.factorize(self.meta_data[self.LBL_COL], sort=True)
        self.LBL_COL = f"lbl_{self.LBL_COL}"
        self.meta_data[self.LBL_COL] = int_lbl
        # precomputed embeddings
        self.pre_computed_embeddings = None
        if pre_computed_embeddings_path is not None:
            import pickle

            with open(pre_computed_embeddings_path, "rb") as file:
                self.pre_computed_embeddings = pickle.load(file)
            self.pre_computed_embeddings = {
                str(dataset_dir.parent / k): v
                for k, v in self.pre_computed_embeddings.items()
            }
            self.meta_data["embedding"] = self.meta_data["img_path"].apply(
                lambda x: self.pre_computed_embeddings.get(x)
            )

            self.meta_data = self.meta_data[self.meta_data["embedding"].notna()]
            self.meta_data.reset_index(drop=True, inplace=True)

        self.return_embedding = return_embedding
        self.classes = list(lbl_mapping)
        self.n_classes = len(self.classes)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        img_name = self.meta_data.loc[self.meta_data.index[index], self.IMG_COL]
        diagnosis = self.meta_data.loc[self.meta_data.index[index], self.LBL_COL]
        if self.return_embedding:
            embedding = self.meta_data.loc[self.meta_data.index[index], "embedding"]
            embedding = torch.Tensor(embedding)

            logger.debug(f"returned embedding")
            return embedding, img_name, int(diagnosis)

        image = Image.open(img_name)
        image = image.convert("RGB")
        if self.transform and self.training:
            image = self.transform(image)
        elif self.val_transform and not self.training:
            image = self.val_transform(image)

        if self.training:
            return image, int(diagnosis)
        return image, img_name, int(diagnosis), index

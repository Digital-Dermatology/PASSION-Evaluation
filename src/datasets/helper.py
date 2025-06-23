from enum import Enum
from pathlib import Path

from loguru import logger
from torch.utils.data import DataLoader

from src.datasets.base_dataset import BaseDataset
from src.datasets.passion_dataset import PASSIONDataset


class DatasetName(Enum):
    PASSION = "passion"


def get_dataset(
    dataset_name: DatasetName,
    dataset_path: Path = Path("../data/"),
    batch_size: int = 128,
    transform=None,
    **kwargs,
) -> BaseDataset:
    if dataset_name == DatasetName.PASSION:
        dataset = PASSIONDataset(
            dataset_dir=dataset_path,
            transform=transform,
            **kwargs,
        )
    else:
        raise ValueError(f"Unknown dataset: {str(dataset_name)}")

    torch_dataset = DataLoader(
        dataset,
        batch_size=batch_size,
        drop_last=False,
        shuffle=False,
        collate_fn=(
            dataset.__class__.collate_fn
            if hasattr(dataset.__class__, "collate_fn")
            else None
        ),
    )
    logger.debug(
        f"Loaded `{dataset_name.value}` which contains {len(torch_dataset)} "
        f"batches with a batch size of {batch_size}."
    )
    return dataset, torch_dataset

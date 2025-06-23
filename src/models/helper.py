import tempfile
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import torch
from loguru import logger
from tqdm.auto import tqdm

ARR_TYPE = Union[np.ndarray, np.memmap, torch.Tensor]


def embed_dataset(
    torch_dataset: torch.utils.data.DataLoader,
    model: Optional[torch.nn.Sequential],
    n_layers: int,
    normalize: bool = False,
    tqdm_desc: Optional[str] = None,
) -> Union[Tuple[ARR_TYPE, ARR_TYPE, ARR_TYPE, ARR_TYPE], Tuple[ARR_TYPE, ARR_TYPE]]:
    labels = []
    paths = []
    indices = []
    batch_size = torch_dataset.batch_size
    iterator = tqdm(
        enumerate(torch_dataset),
        position=0,
        leave=True,
        total=len(torch_dataset),
        desc=tqdm_desc,
    )
    # calculate the embedding dimension for memmap array
    _batch = torch_dataset.dataset[0][0][None, ...]
    if model is not None:
        _batch = _batch.to(model.device)
    batch_dim = tuple(_batch.shape)[1:]
    if (
        type(model) is torch.jit._script.RecursiveScriptModule
        or type(model) is torch.nn.Sequential
    ):
        emb_dim = model(_batch).squeeze().shape[0]
    elif model is None:
        emb_dim = _batch.squeeze().shape[0]
    else:
        emb_dim = model(_batch, n_layers=n_layers).squeeze().shape[0]

    emb_space = np.zeros(shape=(len(torch_dataset.dataset), emb_dim))
    images = np.zeros(shape=(len(torch_dataset.dataset), *batch_dim))

    del emb_dim, batch_dim, _batch
    # embed the dataset
    if (
        type(model) is torch.jit._script.RecursiveScriptModule
        or type(model) is torch.nn.Sequential
    ):
        model_has_layers = False
    else:
        model_has_layers = True

    for i, batch_tup in iterator:
        if len(batch_tup) == 4:
            batch, path, label, index = batch_tup
        else:
            raise ValueError(f"Unknown batch tuple: {batch_tup}")

        with torch.no_grad():
            if model:
                batch = batch.to(model.device)
                if not model_has_layers:
                    emb = model(batch)
                else:
                    emb = model(batch, n_layers=n_layers)
            else:
                emb = batch
            emb = emb.squeeze()

            if normalize:
                emb = torch.nn.functional.normalize(emb, dim=-1, p=2)
            emb_space[batch_size * i : batch_size * (i + 1), :] = emb.cpu()
            labels.append(label.cpu())
            images[batch_size * i : batch_size * (i + 1), :] = batch.cpu()
            if path is not None:
                paths += path
            if index is not None:
                indices += index
    labels = torch.concat(labels).cpu()

    paths = np.array(paths)
    indices = np.array(indices)

    return emb_space, labels, images, paths, indices

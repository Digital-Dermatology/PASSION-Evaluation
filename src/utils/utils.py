import os

import numpy as np
import torch
import torch.distributed as dist
from loguru import logger


def fix_random_seeds(seed=42):
    """
    Fix random seeds.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val


def compare_models(model_1, model_2, log=False):
    models_differ = 0
    for key_item_1, key_item_2 in zip(
        model_1.state_dict().items(), model_2.state_dict().items()
    ):
        if torch.equal(key_item_1[1], key_item_2[1]):
            pass
        else:
            models_differ += 1
            if log and (key_item_1[0] == key_item_2[0]):
                logger.error("Mismatch found at", key_item_1[0])
    return models_differ


def restart_from_checkpoint(ckp_path, run_variables=None, **kwargs):
    if not os.path.isfile(ckp_path):
        logger.info("Pre-trained weights not found. Training from scratch.")
        return
    logger.info("Found checkpoint at {}".format(ckp_path))

    # open checkpoint file
    # TODO: fix the third param when the checkpoint loading was corrected, potential security issue
    checkpoint = torch.load(ckp_path, map_location="cpu", weights_only=False)

    # key is what to look for in the checkpoint file
    # value is the object to load
    # example: {'state_dict': model}
    for key, value in kwargs.items():
        if key in checkpoint and value is not None:
            try:
                msg = value.load_state_dict(checkpoint[key], strict=False)
                if len(msg.missing_keys) > 0:
                    k = next(iter(checkpoint[key]))
                    if "module." in k:
                        logger.debug(
                            f"=> Found `module` in {key}, trying to transform."
                        )
                        transf_state_dict = OrderedDict()
                        for k, v in checkpoint[key].items():
                            # remove the module from the key
                            # this is caused by the distributed training
                            k = k.replace("module.", "")
                            transf_state_dict[k] = v
                        msg = value.load_state_dict(transf_state_dict, strict=False)
                logger.debug(
                    "=> loaded '{}' from checkpoint '{}' with msg {}".format(
                        key, ckp_path, msg
                    )
                )
            except TypeError:
                try:
                    msg = value.load_state_dict(checkpoint[key])
                    logger.debug(
                        "=> loaded '{}' from checkpoint: '{}'".format(key, ckp_path)
                    )
                except ValueError:
                    logger.error(
                        "=> failed to load '{}' from checkpoint: '{}'".format(
                            key, ckp_path
                        )
                    )
        else:
            logger.error(
                "=> key '{}' not found in checkpoint: '{}'".format(key, ckp_path)
            )

    # reload variable important for the run
    if run_variables is not None:
        for var_name in run_variables:
            if var_name in checkpoint:
                run_variables[var_name] = checkpoint[var_name]


def save_checkpoint(run_dir, save_dict, epoch, save_best=False):
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    filename = str(run_dir / "checkpoints" / "checkpoint-epoch{}.pth".format(epoch))
    torch.save(save_dict, filename)
    logger.info("Saving checkpoint: {} ...".format(filename))
    if save_best:
        best_path = str(run_dir / "checkpoints" / "model_best.pth")
        torch.save(save_dict, best_path)
        logger.info("Saving current best: model_best.pth ...")


class EarlyStopping:
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """

    def __init__(self, patience=5, min_delta=0, log_messages: bool = True):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.log_messages = log_messages
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss == None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            # reset counter if validation loss improves
            self.counter = 0
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            if self.log_messages:
                logger.info(f"Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                if self.log_messages:
                    logger.info("Early stopping")
                self.early_stop = True

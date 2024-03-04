import copy
from collections import OrderedDict
from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchmetrics
import wandb
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch_lr_finder import LRFinder
from torchinfo import summary
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from tqdm import tqdm

from src.trainers.eval_types.base import BaseEvalType
from ssl_library.src.models.fine_tuning.classifiers import LinearClassifier
from ssl_library.src.optimizers.utils import get_optimizer_type
from ssl_library.src.utils.utils import (
    EarlyStopping,
    restart_from_checkpoint,
    save_checkpoint,
    set_requires_grad,
)


class EvalFineTuning(BaseEvalType):
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(90),
            # Experiments showed color jitter hinders performance,
            # but check again if problems with models and datasets arise
            # transforms.ColorJitter(brightness=0.3, contrast=0.3, hue=0.3),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    @staticmethod
    def name() -> str:
        return "finetuning"

    @classmethod
    def evaluate(
        cls,
        train_range: np.ndarray,
        evaluation_range: np.ndarray,
        dataset: torch.utils.data.Dataset,
        model: torch.nn.Module,
        model_out_dim: int,
        learning_rate: float,
        batch_size: int,
        train_epochs: int,
        warmup_epochs: int,
        early_stopping_patience: int,
        use_bn_in_head: bool,
        dropout_in_head: float,
        num_workers: int,
        saved_model_path: Union[Path, str, None] = None,
        find_optimal_lr: bool = False,
        use_lr_scheduler: bool = False,
        log_wandb: bool = False,
        debug: bool = False,
        **kwargs,
    ) -> dict:
        model = copy.deepcopy(model)
        # get dataloader for batched compute
        train_loader, eval_loader = cls.get_train_eval_loaders(
            dataset=dataset,
            train_range=train_range,
            evaluation_range=evaluation_range,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        # create the classifier
        device = model.device
        classifier = torch.nn.Sequential(
            OrderedDict(
                [
                    ("backbone", model),
                    ("flatten", torch.nn.Flatten()),
                    (
                        "fc",
                        LinearClassifier(
                            model_out_dim,
                            dataset.n_classes,
                            large_head=False,
                            use_bn=use_bn_in_head,
                            dropout_rate=dropout_in_head,
                        ),
                    ),
                ]
            )
        )
        classifier.to(device)
        # make sure the classifier can get trained
        set_requires_grad(classifier, True)
        if debug:
            try:
                summary(classifier, input_size=(1, 3, 224, 224))
            except RuntimeError:
                print("Summary can not be displayed for a Huggingface model.")
                print(
                    f"Number of parameters backbone: {classifier.backbone.model.num_parameters():,}"
                )
        if log_wandb:
            wandb.watch(classifier, log="all", log_freq=len(train_loader))
        # loss function, optimizer, scores
        criterion = torch.nn.CrossEntropyLoss(
            weight=train_loader.dataset.get_class_weights(),
        )
        criterion = criterion.to(device)
        optimizer_cls = get_optimizer_type(optimizer_name="adam")
        optimizer = optimizer_cls(
            params=classifier.parameters(),
            lr=learning_rate,
        )
        if find_optimal_lr:
            # automatic learning rate finder
            lr_finder = LRFinder(classifier, optimizer, criterion, device=device)
            lr_finder.range_test(train_loader, end_lr=100, num_iter=100)
            lrs = lr_finder.history["lr"]
            losses = lr_finder.history["loss"]
            # log the LRFinder plot
            fig, ax = plt.subplots()
            lr_finder.plot(ax=ax)
            wandb.log({"LRFinder_Plot": wandb.Image(fig)})
            # to reset the model and optimizer to their initial state
            lr_finder.reset()
            try:
                min_grad_idx = (np.gradient(np.array(losses))).argmin()
                best_lr = lrs[min_grad_idx]
                optimizer = optimizer_cls(
                    params=classifier.parameters(),
                    lr=best_lr,
                )
            except ValueError:
                print("Failed to compute the gradients. Relying on default lr.")

        # we use early stopping to speed up the training
        early_stopping = EarlyStopping(
            patience=early_stopping_patience,
            log_messages=debug,
        )

        if use_lr_scheduler:
            # define the learning rate scheduler
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer=optimizer,
                T_max=train_epochs,
                eta_min=0,
            )

        # load the model from checkpoint if provided
        to_restore = {"epoch": 0}
        # TODO: fix this here
        if False:
            if saved_model_path is not None:
                restart_from_checkpoint(
                    Path(saved_model_path) / "checkpoints" / "model_best.pth",
                    run_variables=to_restore,
                    classifier=classifier,
                    optimizer=optimizer,
                    loss=criterion,
                )
        start_epoch = to_restore["epoch"]

        # define metrics
        loss_metric_train = torchmetrics.MeanMetric().to(device)
        f1_score_train = torchmetrics.F1Score(
            task="multiclass",
            num_classes=classifier.fc.num_labels,
            average="macro",
        ).to(device)

        loss_metric_val = torchmetrics.MeanMetric().to(device)
        f1_score_val = torchmetrics.F1Score(
            task="multiclass",
            num_classes=classifier.fc.num_labels,
            average="macro",
        ).to(device)
        precision_val = torchmetrics.Precision(
            task="multiclass",
            num_classes=classifier.fc.num_labels,
            average="macro",
        ).to(device)
        recall_val = torchmetrics.Recall(
            task="multiclass",
            num_classes=classifier.fc.num_labels,
            average="macro",
        ).to(device)
        auroc_val = torchmetrics.AUROC(
            task="multiclass",
            num_classes=classifier.fc.num_labels,
        ).to(device)

        # start training
        epoch, step = start_epoch, 0
        eval_scores_dict = {
            "f1": {
                "metric": f1_score_val,
                "scores": [],
            },
            "precision": {
                "metric": precision_val,
                "scores": [],
            },
            "recall": {
                "metric": recall_val,
                "scores": [],
            },
            "auroc": {
                "metric": auroc_val,
                "scores": [],
            },
        }
        l_loss_val = []
        best_val_loss = np.inf
        best_model_wts = copy.deepcopy(classifier.state_dict())
        for epoch in tqdm(
            range(epoch, train_epochs),
            total=train_epochs,
            desc="Model Training",
        ):
            if epoch >= warmup_epochs:
                # make sure the classifier and backbone get trained
                set_requires_grad(classifier, True)
            else:
                # freeze the backbone and let only the classifier be trained
                set_requires_grad(classifier, True)
                set_requires_grad(classifier.backbone, False)

            # training
            classifier.train()
            for img, target in train_loader:
                img = img.to(device)
                target = target.to(device)

                optimizer.zero_grad()

                pred = classifier(img)
                loss = criterion(pred, target)

                loss.backward()
                optimizer.step()
                if use_lr_scheduler:
                    scheduler.step()

                # W&B logging if needed
                if log_wandb:
                    log_dict = {
                        "train_loss": loss.item(),
                        # TODO: check if this needs a LogSoftmax
                        "train_f1": f1_score_train(pred, target),
                        "learning_rate": optimizer.param_groups[0]["lr"],
                        "weight_decay": optimizer.param_groups[0]["weight_decay"],
                        "epoch": epoch,
                        "step": step,
                    }
                    wandb.log(log_dict)
                # add to overall metrics
                loss_metric_train.update(loss.detach())
                f1_score_train.update(pred, target)
                step += 1

            # Evaluation
            classifier.eval()
            for img, target in eval_loader:
                img = img.to(device)
                target = target.to(device)
                with torch.no_grad():
                    pred = classifier(img)
                    loss = criterion(pred, target)
                loss_metric_val.update(loss)
                # TODO: check if this needs a LogSoftmax
                for _score_dict in eval_scores_dict.values():
                    _score_dict["metric"].update(pred, target)
            l_loss_val.append(loss_metric_val.compute())
            for _score_dict in eval_scores_dict.values():
                _score_dict["scores"].append(_score_dict["metric"].compute())
            # check if we have new best model
            if l_loss_val[-1] < best_val_loss:
                best_val_loss = l_loss_val[-1]
                best_model_wts = copy.deepcopy(classifier.state_dict())
            # check early stopping
            early_stopping(loss_metric_val.compute())
            if early_stopping.early_stop:
                if debug:
                    print("EarlyStopping, evaluation did not decrease.")
                break
            # W&B logging if needed
            if log_wandb:
                log_dict = {
                    "eval_loss": loss_metric_val.compute(),
                    "epoch": epoch,
                    "step": step,
                }
                for score_name, _score_dict in eval_scores_dict.items():
                    log_dict[f"eval_{score_name}"] = _score_dict["scores"][-1]
                wandb.log(log_dict)

        # get the best epoch in terms of F1 score
        wandb.unwatch()
        best_epoch = torch.Tensor(eval_scores_dict["f1"]["scores"]).argmax()
        if log_wandb:
            log_dict = {
                "best_eval_epoch": best_epoch,
                "best_eval_loss": l_loss_val[best_epoch],
                "epoch": epoch,
                "step": step,
            }
            for score_name, _score_dict in eval_scores_dict.items():
                log_dict[f"best_eval_{score_name}"] = _score_dict["scores"][best_epoch]
            wandb.log(log_dict)
        classifier.load_state_dict(best_model_wts)
        if saved_model_path is not None:
            save_dict = {
                "arch": type(classifier).__name__,
                "epoch": epoch,
                "classifier": classifier,
                "optimizer": optimizer.state_dict(),
                "loss": criterion.state_dict(),
            }
            save_checkpoint(
                run_dir=saved_model_path,
                save_dict=save_dict,
                epoch=epoch,
                save_best=True,
            )
        # create eval predictions for saving
        targets, predictions = [], []
        classifier.eval()
        for img, target in eval_loader:
            img = img.to(device)
            target = target.to(device)
            with torch.no_grad():
                pred = classifier(img)
            # TODO: LogSoftmax or only Softmax?
            # pred = nn.LogSoftmax(dim=1)(pred)
            targets.append(target.cpu().argmax(dim=-1))
            predictions.append(pred.cpu())
        targets = torch.concat(targets).cpu().numpy()
        predictions = torch.concat(predictions).cpu().numpy()
        return {
            "score": float(eval_scores_dict["f1"]["scores"][best_epoch] * 100),
            "targets": targets,
            "predictions": predictions,
        }

    @classmethod
    def get_train_eval_loaders(
        cls,
        dataset: torch.utils.data.Dataset,
        train_range: np.ndarray,
        evaluation_range: np.ndarray,
        batch_size: int,
        num_workers: int,
    ):
        train_dataset = copy.deepcopy(dataset)
        train_dataset.transform = cls.train_transform
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=SubsetRandomSampler(train_range),
            num_workers=num_workers,
            drop_last=True,
            shuffle=False,
        )
        del train_dataset

        eval_dataset = copy.deepcopy(dataset)
        eval_dataset.transform = cls.val_transform
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=batch_size,
            sampler=SubsetRandomSampler(evaluation_range),
            num_workers=num_workers,
            drop_last=False,
            shuffle=False,
        )
        del eval_dataset
        return train_loader, eval_loader

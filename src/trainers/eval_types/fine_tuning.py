import copy
from collections import OrderedDict

import numpy as np
import torch
import torchmetrics
import wandb
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch_lr_finder import LRFinder
from torchinfo import summary
from torchvision import transforms

from src.trainers.eval_types.base import BaseEvalType
from ssl_library.src.models.fine_tuning.classifiers import LinearClassifier
from ssl_library.src.optimizers.utils import get_optimizer_type
from ssl_library.src.utils.utils import EarlyStopping, set_requires_grad


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
            transforms.Resize(224, interpolation=3),
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
        test_range: np.ndarray,
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
        find_optimal_lr: bool = False,
        log_wandb: bool = False,
        debug: bool = False,
        **kwargs,
    ) -> float:
        # get dataloader for batched compute
        train_loader, eval_loader = cls.get_train_eval_loaders(
            dataset=dataset,
            train_range=train_range,
            test_range=test_range,
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

        # define the learning rate scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            T_max=train_epochs,
            eta_min=0,
        )

        # define metrics
        loss_metric_train = torchmetrics.MeanMetric()
        loss_metric_train = loss_metric_train.to(device)
        f1_score_train = torchmetrics.F1Score(
            task="multiclass",
            num_classes=classifier.fc.num_labels,
            average="macro",
        )
        f1_score_train = f1_score_train.to(device)

        loss_metric_val = torchmetrics.MeanMetric()
        loss_metric_val = loss_metric_val.to(device)
        f1_score_val = torchmetrics.F1Score(
            task="multiclass",
            num_classes=classifier.fc.num_labels,
            average="macro",
        )
        f1_score_val = f1_score_val.to(device)

        # start training
        step = 0
        l_f1_val = []
        for epoch in range(train_epochs):
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
                scheduler.step()

                # W&B logging if needed
                if log_wandb:
                    log_dict = {
                        "train_loss": loss.item(),
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

            # Validation
            classifier.eval()
            for img, target in eval_loader:
                # move batch to device
                img = img.to(device)
                target = target.to(device)

                with torch.no_grad():
                    pred = classifier(img)
                    loss = criterion(pred, target)
                # add to overall metrics
                loss_metric_val.update(loss)
                f1_score_val.update(pred, target)
            l_f1_val.append(f1_score_val.compute())
            if debug:
                print(
                    f"Epoch: {epoch}, "
                    f"Train Loss: {loss_metric_train.compute()}, "
                    f"Train F1: {f1_score_train.compute()}, "
                    f"Valid Loss: {loss_metric_val.compute()}, "
                    f"Valid F1: {f1_score_val.compute()}"
                )
            # check early stopping
            early_stopping(loss_metric_val.compute())
            if early_stopping.early_stop:
                if debug:
                    print("EarlyStopping, validation did not decrease.")
                break
            # W&B logging if needed
            if log_wandb:
                log_dict = {
                    "valid_loss": loss_metric_val.compute(),
                    "valid_f1": l_f1_val[-1],
                    "epoch": epoch,
                    "step": step,
                }
                wandb.log(log_dict)
        return float(torch.Tensor(l_f1_val).max() * 100)

    @classmethod
    def get_train_eval_loaders(
        cls,
        dataset: torch.utils.data.Dataset,
        train_range: np.ndarray,
        test_range: np.ndarray,
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
            sampler=SubsetRandomSampler(test_range),
            num_workers=num_workers,
            drop_last=False,
            shuffle=False,
        )
        del eval_dataset
        return train_loader, eval_loader

from abc import ABC, abstractmethod

import numpy as np


class BaseEvalType(ABC):
    @staticmethod
    @abstractmethod
    def name() -> str:
        raise NotImplementedError

    @staticmethod
    def split_data(
        emb_space: np.ndarray,
        labels: np.ndarray,
        train_range: np.ndarray,
        test_range: np.ndarray,
    ):
        X_train, y_train = (
            emb_space[train_range],
            labels[train_range],
        )
        X_test, y_test = emb_space[test_range], labels[test_range]
        return (X_train, y_train), (X_test, y_test)

    @classmethod
    @abstractmethod
    def evaluate(
        cls,
        emb_space: np.ndarray,
        labels: np.ndarray,
        train_range: np.ndarray,
        test_range: np.ndarray,
        **kwargs,
    ) -> float:
        raise NotImplementedError

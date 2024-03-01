import numpy as np
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier

from src.trainers.eval_types.base import BaseEvalType


class EvalKNN(BaseEvalType):
    @staticmethod
    def name() -> str:
        return "kNN"

    @classmethod
    def evaluate(
        cls,
        emb_space: np.ndarray,
        labels: np.ndarray,
        train_range: np.ndarray,
        test_range: np.ndarray,
        k: int = 10,
        **kwargs,
    ) -> float:
        train, test = cls.split_data(
            emb_space=emb_space,
            labels=labels,
            train_range=train_range,
            test_range=test_range,
        )
        X_train, y_train = train
        X_test, y_test = test
        del train, test

        knn = KNeighborsClassifier(
            n_neighbors=k,
            metric="cosine",
        )
        knn.fit(X_train, y_train)
        f1 = float(
            f1_score(
                y_test,
                knn.predict(X_test),
                average="macro",
            )
            * 100
        )
        return f1

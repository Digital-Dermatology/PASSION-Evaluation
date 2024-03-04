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
        evaluation_range: np.ndarray,
        k: int = 10,
        **kwargs,
    ) -> dict:
        train, evaluation = cls.split_data(
            emb_space=emb_space,
            labels=labels,
            train_range=train_range,
            evaluation_range=evaluation_range,
        )
        X_train, y_train = train
        X_eval, y_eval = evaluation
        del train, evaluation

        knn = KNeighborsClassifier(
            n_neighbors=k,
            metric="cosine",
        )
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_eval)

        f1 = float(
            f1_score(
                y_eval,
                y_pred,
                average="macro",
            )
            * 100
        )
        return {
            "score": f1,
            "targets": y_eval,
            "predictions": y_pred,
        }

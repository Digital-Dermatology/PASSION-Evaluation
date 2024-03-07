import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.metrics import f1_score

from src.trainers.eval_types.base import BaseEvalType


class EvalDummy(BaseEvalType):
    @staticmethod
    def name() -> str:
        return "DummyClassifier"

    @classmethod
    def evaluate(
        cls,
        emb_space: np.ndarray,
        labels: np.ndarray,
        train_range: np.ndarray,
        evaluation_range: np.ndarray,
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

        knn = DummyClassifier(
            strategy=kwargs.get("strategy", "most_frequent"),
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


class EvalDummyMostFrequent(EvalDummy):
    @staticmethod
    def name() -> str:
        return "DummyClassifier-MostFrequent"

    @classmethod
    def evaluate(
        cls,
        emb_space: np.ndarray,
        labels: np.ndarray,
        train_range: np.ndarray,
        evaluation_range: np.ndarray,
        **kwargs,
    ) -> dict:
        return super().evaluate(
            emb_space=emb_space,
            labels=labels,
            train_range=train_range,
            evaluation_range=evaluation_range,
            strategy="most_frequent",
            **kwargs,
        )


class EvalDummyConstant(EvalDummy):
    @staticmethod
    def name() -> str:
        return "DummyClassifier-Constant"

    @classmethod
    def evaluate(
        cls,
        emb_space: np.ndarray,
        labels: np.ndarray,
        train_range: np.ndarray,
        evaluation_range: np.ndarray,
        constant: int = 1,
        **kwargs,
    ) -> dict:
        return super().evaluate(
            emb_space=emb_space,
            labels=labels,
            train_range=train_range,
            evaluation_range=evaluation_range,
            strategy="constant",
            constant=constant,
            **kwargs,
        )


class EvalDummyUniform(EvalDummy):
    @staticmethod
    def name() -> str:
        return "DummyClassifier-Uniform"

    @classmethod
    def evaluate(
        cls,
        emb_space: np.ndarray,
        labels: np.ndarray,
        train_range: np.ndarray,
        evaluation_range: np.ndarray,
        **kwargs,
    ) -> dict:
        return super().evaluate(
            emb_space=emb_space,
            labels=labels,
            train_range=train_range,
            evaluation_range=evaluation_range,
            strategy="uniform",
            **kwargs,
        )

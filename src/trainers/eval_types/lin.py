import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from src.trainers.eval_types.base import BaseEvalType


class EvalLin(BaseEvalType):
    @staticmethod
    def name() -> str:
        return "lin"

    @classmethod
    def evaluate(
        cls,
        emb_space: np.ndarray,
        labels: np.ndarray,
        train_range: np.ndarray,
        test_range: np.ndarray,
        solver: str = "sag",
        tol: float = 0.1,
        max_iter: int = 100,
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

        lin = LogisticRegression(
            solver=solver,
            n_jobs=-1,
            tol=tol,
            max_iter=max_iter,
        )
        lin.fit(X_train, y_train)
        f1 = float(
            f1_score(
                y_test,
                lin.predict(X_test),
                average="macro",
            )
            * 100
        )
        return f1

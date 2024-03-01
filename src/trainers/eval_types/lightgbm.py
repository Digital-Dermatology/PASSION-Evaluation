import lightgbm as lgb
import numpy as np
from sklearn.metrics import f1_score

from src.trainers.eval_types.base import BaseEvalType


class EvalLightGBM(BaseEvalType):
    @staticmethod
    def name() -> str:
        return "LightGBM"

    @classmethod
    def evaluate(
        cls,
        emb_space: np.ndarray,
        labels: np.ndarray,
        train_range: np.ndarray,
        test_range: np.ndarray,
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

        estimator = lgb.LGBMClassifier(
            n_jobs=-1,
            seed=42,
            device="gpu",
            verbosity=-1,
        )
        estimator.fit(X_train, y_train)
        f1 = float(
            f1_score(
                y_test,
                estimator.predict(X_test),
                average="macro",
            )
            * 100
        )
        return f1

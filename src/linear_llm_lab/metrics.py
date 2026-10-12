"""Numerically robust representation-prediction metrics."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


@dataclass(frozen=True)
class RegressionMetrics:
    mse: float
    cosine: float
    r2: float

    def as_dict(self) -> dict[str, float]:
        return {"mse": self.mse, "cosine": self.cosine, "r2": self.r2}


def mean_cosine(y_true: Array, y_pred: Array, eps: float = 1e-12) -> float:
    numerator = np.sum(y_true * y_pred, axis=-1)
    denominator = np.linalg.norm(y_true, axis=-1) * np.linalg.norm(y_pred, axis=-1)
    return float(np.mean(numerator / np.maximum(denominator, eps)))


def explained_variance_r2(y_true: Array, y_pred: Array) -> float:
    residual = float(np.sum((y_true - y_pred) ** 2))
    total = float(np.sum((y_true - y_true.mean(axis=0)) ** 2))
    return 1.0 - residual / total if total > 0 else float(residual == 0)


def regression_metrics(y_true: Array, y_pred: Array) -> RegressionMetrics:
    return RegressionMetrics(
        mse=float(np.mean((y_true - y_pred) ** 2)),
        cosine=mean_cosine(y_true, y_pred),
        r2=explained_variance_r2(y_true, y_pred),
    )


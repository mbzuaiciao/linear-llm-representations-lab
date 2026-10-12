"""Small linear probes with explicit train/validation reporting."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression

from .linear_maps import Array


@dataclass
class ProbeResult:
    model: LogisticRegression
    train_accuracy: float
    validation_accuracy: float


def train_probe(
    x_train: Array,
    y_train: NDArray[np.int64],
    x_validation: Array,
    y_validation: NDArray[np.int64],
    seed: int = 0,
) -> ProbeResult:
    model = LogisticRegression(max_iter=1_000, random_state=seed)
    model.fit(x_train, y_train)
    return ProbeResult(
        model=model,
        train_accuracy=float(model.score(x_train, y_train)),
        validation_accuracy=float(model.score(x_validation, y_validation)),
    )


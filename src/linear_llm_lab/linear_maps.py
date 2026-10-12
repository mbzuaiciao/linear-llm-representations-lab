"""Explicit least-squares affine maps."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


@dataclass
class AffineMap:
    matrix: Array
    bias: Array

    def predict(self, x: Array) -> Array:
        return x @ self.matrix + self.bias

    @property
    def parameter_count(self) -> int:
        return int(self.matrix.size + self.bias.size)


def fit_affine(x: Array, y: Array, ridge: float = 1e-8) -> AffineMap:
    """Fit y ~= x A + b, leaving the intercept unregularized."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    design = np.column_stack([x, np.ones(len(x))])
    gram = design.T @ design
    penalty = ridge * np.eye(gram.shape[0])
    penalty[-1, -1] = 0.0
    weights = np.linalg.solve(gram + penalty, design.T @ y)
    return AffineMap(weights[:-1], weights[-1])


def identity_prediction(x: Array, output_dim: int | None = None) -> Array:
    resolved_output_dim = x.shape[1] if output_dim is None else output_dim
    prediction = np.zeros((len(x), resolved_output_dim), dtype=np.float64)
    shared = min(x.shape[1], resolved_output_dim)
    prediction[:, :shared] = x[:, :shared]
    return prediction

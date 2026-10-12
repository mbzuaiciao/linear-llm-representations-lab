"""Model comparisons used throughout the experiments and capstone."""

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from sklearn.neural_network import MLPRegressor

from .linear_maps import Array, fit_affine, identity_prediction
from .low_rank import fit_low_rank
from .metrics import regression_metrics
from .regimes import fit_regimes


class Predictor(Protocol):
    @property
    def parameter_count(self) -> int: ...

    def predict(self, x: Array) -> Array: ...


@dataclass
class MeanPredictor:
    mean: Array

    def predict(self, x: Array) -> Array:
        return np.repeat(self.mean[None], len(x), axis=0)

    @property
    def parameter_count(self) -> int:
        return int(self.mean.size)


@dataclass
class IdentityPredictor:
    output_dim: int

    def predict(self, x: Array) -> Array:
        return identity_prediction(x, self.output_dim)

    @property
    def parameter_count(self) -> int:
        return 0


@dataclass
class MLPPredictor:
    model: MLPRegressor

    def predict(self, x: Array) -> Array:
        return np.asarray(self.model.predict(x), dtype=np.float64)

    @property
    def parameter_count(self) -> int:
        return int(sum(weights.size for weights in self.model.coefs_) + sum(bias.size for bias in self.model.intercepts_))


def fit_mlp(x: Array, y: Array, seed: int = 0, hidden: int = 32) -> MLPPredictor:
    model = MLPRegressor(
        hidden_layer_sizes=(hidden,),
        activation="tanh",
        max_iter=1_000,
        early_stopping=True,
        random_state=seed,
    ).fit(x, y)
    return MLPPredictor(model)


def fit_candidate_models(x: Array, y: Array, seed: int = 0, rank: int = 4, regimes: int = 2) -> dict[str, Predictor]:
    return {
        "identity": IdentityPredictor(y.shape[1]),
        "mean": MeanPredictor(y.mean(axis=0)),
        "affine": fit_affine(x, y, ridge=1e-6),
        f"rank_{rank}": fit_low_rank(x, y, rank=rank, ridge=1e-6),
        f"local_{regimes}": fit_regimes(x, y, n_regimes=regimes, seed=seed),
        "mlp": fit_mlp(x, y, seed=seed),
    }


def evaluate_models(models: dict[str, Predictor], x: Array, y: Array) -> dict[str, dict[str, float | int]]:
    output: dict[str, dict[str, float | int]] = {}
    for name, model in models.items():
        metrics = regression_metrics(y, model.predict(x)).as_dict()
        count = model.parameter_count
        output[name] = {
            **metrics,
            "parameters": count,
            "r2_per_1000_parameters": metrics["r2"] / max(count, 1) * 1_000,
        }
    return output


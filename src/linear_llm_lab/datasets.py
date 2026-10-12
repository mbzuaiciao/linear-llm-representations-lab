"""Controlled prompts and synthetic activation-like dynamics."""

from dataclasses import dataclass

import numpy as np

from .linear_maps import Array

POSITIVE_WORDS = ["wonderful", "delightful", "excellent", "kind", "bright", "pleasant"]
NEGATIVE_WORDS = ["terrible", "awful", "dreadful", "cruel", "bleak", "unpleasant"]
NOUNS = ["movie", "book", "meal", "journey", "concert", "story"]


def sentiment_prompts(repeats: int = 4) -> tuple[list[str], np.ndarray]:
    prompts: list[str] = []
    labels: list[int] = []
    for repeat in range(repeats):
        for noun, positive, negative in zip(NOUNS, POSITIVE_WORDS, NEGATIVE_WORDS, strict=True):
            suffix = "!" * (repeat % 2) + "."
            prompts.extend([f"The {noun} was {positive}{suffix}", f"The {noun} was {negative}{suffix}"])
            labels.extend([1, 0])
    return prompts, np.asarray(labels, dtype=np.int64)


def prompt_domains(n: int = 64) -> tuple[list[str], list[str]]:
    declaratives = [f"The careful researcher records observation {i}." for i in range(n)]
    questions = [f"Why did the researcher record observation {i}?" for i in range(n)]
    return declaratives, questions


@dataclass(frozen=True)
class DynamicsData:
    x_train: Array
    y_train: Array
    x_test: Array
    y_test: Array
    x_shift: Array
    y_shift: Array


def synthetic_dynamics(seed: int = 0, n: int = 600, dimension: int = 16) -> DynamicsData:
    """Activation-like data with low-rank global and regime-specific dynamics."""
    rng = np.random.default_rng(seed)
    latent_dim = min(5, dimension)
    basis, _ = np.linalg.qr(rng.normal(size=(dimension, latent_dim)))
    global_matrix = 0.72 * np.eye(dimension) + 0.16 * (basis @ basis.T)
    direction = rng.normal(size=dimension)
    direction /= np.linalg.norm(direction)
    regime_update = 0.20 * np.outer(direction, direction)

    def sample(count: int, shift: float) -> tuple[Array, Array]:
        latent = rng.normal(loc=shift, size=(count, latent_dim))
        x = latent @ basis.T + 0.04 * rng.normal(size=(count, dimension))
        signs = np.where(x[:, 0] >= 0, 1.0, -1.0)[:, None]
        y = x @ global_matrix + signs * (x @ regime_update)
        y += 0.01 * rng.normal(size=y.shape)
        return x.astype(np.float64), y.astype(np.float64)

    x_train, y_train = sample(n, 0.0)
    x_test, y_test = sample(max(n // 3, 100), 0.0)
    x_shift, y_shift = sample(max(n // 3, 100), 1.2)
    return DynamicsData(x_train, y_train, x_test, y_test, x_shift, y_shift)


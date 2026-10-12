"""Reduced-rank regression implemented with a truncated SVD."""

from dataclasses import dataclass

import numpy as np

from .linear_maps import AffineMap, Array, fit_affine


@dataclass
class LowRankAffineMap(AffineMap):
    rank: int

    @property
    def parameter_count(self) -> int:
        input_dim, output_dim = self.matrix.shape
        return int(self.rank * (input_dim + output_dim) + output_dim)


def fit_low_rank(x: Array, y: Array, rank: int, ridge: float = 1e-8) -> LowRankAffineMap:
    """Fit affine regression, then take its best rank-r matrix approximation."""
    full = fit_affine(x, y, ridge=ridge)
    u, singular_values, vt = np.linalg.svd(full.matrix, full_matrices=False)
    effective_rank = min(rank, len(singular_values))
    matrix = (u[:, :effective_rank] * singular_values[:effective_rank]) @ vt[:effective_rank]
    bias = np.mean(y - x @ matrix, axis=0)
    return LowRankAffineMap(matrix, bias, effective_rank)


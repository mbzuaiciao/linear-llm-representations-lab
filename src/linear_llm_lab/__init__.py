"""Tools for testing linear structure in learned representations."""

from .linear_maps import AffineMap, fit_affine
from .low_rank import LowRankAffineMap, fit_low_rank

__all__ = ["AffineMap", "LowRankAffineMap", "fit_affine", "fit_low_rank"]


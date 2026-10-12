"""Piecewise-affine prediction using k-means regimes."""

from dataclasses import dataclass
from typing import cast

import numpy as np
from sklearn.cluster import KMeans

from .linear_maps import AffineMap, Array, fit_affine


@dataclass
class RegimeMap:
    clusterer: KMeans
    maps: list[AffineMap]

    def assign(self, x: Array) -> np.ndarray:
        return cast(np.ndarray, self.clusterer.predict(x))

    def predict(self, x: Array) -> Array:
        assignments = self.assign(x)
        output_dim = self.maps[0].bias.size
        output = np.empty((len(x), output_dim), dtype=np.float64)
        for index, model in enumerate(self.maps):
            mask = assignments == index
            if np.any(mask):
                output[mask] = model.predict(x[mask])
        return output

    @property
    def parameter_count(self) -> int:
        centers = int(self.clusterer.cluster_centers_.size)
        return centers + sum(model.parameter_count for model in self.maps)


def fit_regimes(x: Array, y: Array, n_regimes: int, seed: int = 0) -> RegimeMap:
    clusterer = KMeans(n_clusters=n_regimes, n_init="auto", random_state=seed).fit(x)
    maps: list[AffineMap] = []
    fallback = fit_affine(x, y, ridge=1e-6)
    for index in range(n_regimes):
        mask = clusterer.labels_ == index
        maps.append(fit_affine(x[mask], y[mask], ridge=1e-6) if np.any(mask) else fallback)
    return RegimeMap(clusterer, maps)

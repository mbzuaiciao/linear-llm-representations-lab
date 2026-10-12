"""Geometry statistics for sequences of hidden states."""

from dataclasses import dataclass

import numpy as np

from .linear_maps import Array


@dataclass(frozen=True)
class TrajectoryStats:
    step_lengths: Array
    turning_angles: Array
    displacement_cosines: Array


def trajectory_statistics(states: Array, eps: float = 1e-12) -> TrajectoryStats:
    if len(states) < 2:
        raise ValueError("A trajectory needs at least two states")
    displacements = np.diff(states, axis=0)
    lengths = np.linalg.norm(displacements, axis=1)
    if len(displacements) < 2:
        empty = np.empty(0, dtype=np.float64)
        return TrajectoryStats(lengths, empty, empty)
    dot = np.sum(displacements[:-1] * displacements[1:], axis=1)
    scale = lengths[:-1] * lengths[1:]
    cosines = np.clip(dot / np.maximum(scale, eps), -1.0, 1.0)
    return TrajectoryStats(lengths, np.arccos(cosines), cosines)


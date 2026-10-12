import numpy as np

from linear_llm_lab.trajectories import trajectory_statistics


def test_straight_trajectory_has_zero_turning_angle() -> None:
    states = np.column_stack([np.arange(6), np.zeros(6)]).astype(np.float64)
    stats = trajectory_statistics(states)
    np.testing.assert_allclose(stats.step_lengths, 1)
    np.testing.assert_allclose(stats.displacement_cosines, 1)
    np.testing.assert_allclose(stats.turning_angles, 0)


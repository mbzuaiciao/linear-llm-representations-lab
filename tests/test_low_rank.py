import numpy as np

from linear_llm_lab.low_rank import fit_low_rank


def test_rank_r_system_is_recovered() -> None:
    rng = np.random.default_rng(2)
    x = rng.normal(size=(400, 8))
    matrix = rng.normal(size=(8, 3)) @ rng.normal(size=(3, 7))
    bias = rng.normal(size=7)
    y = x @ matrix + bias
    model = fit_low_rank(x, y, rank=3, ridge=0.0)
    np.testing.assert_allclose(model.predict(x), y, atol=1e-9)
    assert np.linalg.matrix_rank(model.matrix, tol=1e-8) == 3


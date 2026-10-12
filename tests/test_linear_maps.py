import numpy as np

from linear_llm_lab.linear_maps import fit_affine
from linear_llm_lab.metrics import regression_metrics


def test_exact_affine_system_is_recovered() -> None:
    rng = np.random.default_rng(1)
    x = rng.normal(size=(200, 5))
    matrix = rng.normal(size=(5, 3))
    bias = rng.normal(size=3)
    model = fit_affine(x, x @ matrix + bias, ridge=0.0)
    np.testing.assert_allclose(model.matrix, matrix, atol=1e-10)
    np.testing.assert_allclose(model.bias, bias, atol=1e-10)


def test_metrics_are_exact_for_perfect_prediction() -> None:
    y = np.arange(12, dtype=np.float64).reshape(4, 3) + 1
    result = regression_metrics(y, y)
    assert result.mse == 0
    assert result.r2 == 1
    assert np.isclose(result.cosine, 1)


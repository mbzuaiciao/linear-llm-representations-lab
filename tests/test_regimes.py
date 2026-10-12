import numpy as np

from linear_llm_lab.linear_maps import fit_affine
from linear_llm_lab.metrics import regression_metrics
from linear_llm_lab.regimes import fit_regimes


def test_piecewise_system_prefers_local_regimes() -> None:
    rng = np.random.default_rng(4)
    x = np.concatenate([rng.normal(loc=-2, scale=0.3, size=(200, 2)), rng.normal(loc=2, scale=0.3, size=(200, 2))])
    y = np.empty_like(x)
    y[:200] = x[:200] @ np.array([[2.0, 0.0], [0.0, 0.5]])
    y[200:] = x[200:] @ np.array([[-1.0, 0.5], [0.5, -1.0]])
    order = rng.permutation(len(x))
    train, test = order[:300], order[300:]
    global_model = fit_affine(x[train], y[train])
    local_model = fit_regimes(x[train], y[train], 2, seed=4)
    global_mse = regression_metrics(y[test], global_model.predict(x[test])).mse
    local_mse = regression_metrics(y[test], local_model.predict(x[test])).mse
    assert local_mse < global_mse * 0.05
    assert set(local_model.assign(x[test])) == {0, 1}


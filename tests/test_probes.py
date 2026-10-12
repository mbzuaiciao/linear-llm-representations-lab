import numpy as np

from linear_llm_lab.probes import train_probe


def test_probe_learns_linearly_separable_feature() -> None:
    rng = np.random.default_rng(3)
    x = rng.normal(size=(200, 5))
    y = (x[:, 0] > 0).astype(np.int64)
    result = train_probe(x[:150], y[:150], x[150:], y[150:], seed=3)
    assert result.train_accuracy > 0.95
    assert result.validation_accuracy > 0.9


import torch

from linear_llm_lab.jacobians import relative_taylor_error


def test_taylor_error_decreases_with_perturbation_size() -> None:
    point = torch.tensor([0.2, -0.3], dtype=torch.float64)
    direction = torch.tensor([0.7, 0.4], dtype=torch.float64)
    def function(value: torch.Tensor) -> torch.Tensor:
        return torch.stack([torch.sin(value[0]), torch.exp(value[1])])
    large = relative_taylor_error(function, point, direction * 0.1)
    small = relative_taylor_error(function, point, direction * 0.001)
    assert small < large / 20

"""Jacobian-vector products and first-order approximation checks."""

from collections.abc import Callable
from typing import cast

import torch


def jvp(
    function: Callable[[torch.Tensor], torch.Tensor],
    point: torch.Tensor,
    direction: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    value, product = cast(
        tuple[torch.Tensor, torch.Tensor],
        torch.autograd.functional.jvp(function, point, direction),
    )
    return value.detach(), product.detach()


def taylor_approximation(
    function: Callable[[torch.Tensor], torch.Tensor],
    point: torch.Tensor,
    delta: torch.Tensor,
) -> torch.Tensor:
    value, product = jvp(function, point, delta)
    return value + product


def relative_taylor_error(
    function: Callable[[torch.Tensor], torch.Tensor],
    point: torch.Tensor,
    delta: torch.Tensor,
    eps: float = 1e-12,
) -> float:
    truth = function(point + delta).detach()
    approximation = taylor_approximation(function, point, delta)
    return float(torch.linalg.vector_norm(truth - approximation) / (torch.linalg.vector_norm(truth) + eps))

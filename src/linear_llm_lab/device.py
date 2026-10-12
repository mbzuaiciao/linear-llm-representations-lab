"""Device selection shared by tutorials and experiments."""

import torch


def best_device() -> torch.device:
    """Select CUDA, then Apple MPS, then CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


"""Headless plotting and result serialization helpers."""

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def save_json(data: Any, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(data, indent=2) + "\n")


def save_curve(
    x: list[float] | list[int],
    series: dict[str, list[float]],
    path: str | Path,
    xlabel: str,
    ylabel: str,
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(6, 4))
    for label, values in series.items():
        axis.plot(x, values, marker="o", label=label)
    axis.set(xlabel=xlabel, ylabel=ylabel)
    axis.grid(alpha=0.25)
    if len(series) > 1:
        axis.legend()
    figure.tight_layout()
    figure.savefig(destination, dpi=160)
    plt.close(figure)


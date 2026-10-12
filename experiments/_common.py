"""Shared CLI and result helpers for reproducible experiments."""

import argparse
from pathlib import Path

import numpy as np

from linear_llm_lab.activations import (
    extract_hidden_states,
    flatten_valid_tokens,
)
from linear_llm_lab.comparison import evaluate_models, fit_candidate_models
from linear_llm_lab.datasets import prompt_domains, sentiment_prompts, synthetic_dynamics
from linear_llm_lab.jacobians import relative_taylor_error
from linear_llm_lab.linear_maps import fit_affine
from linear_llm_lab.low_rank import fit_low_rank
from linear_llm_lab.metrics import regression_metrics
from linear_llm_lab.models import load_causal_lm
from linear_llm_lab.plotting import save_curve, save_json
from linear_llm_lab.regimes import fit_regimes
from linear_llm_lab.trajectories import trajectory_statistics


def base_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--samples", type=int, default=600)
    parser.add_argument("--output", type=Path)
    return parser


def layer_linearity(seed: int, model_name: str, synthetic: bool) -> dict[str, object]:
    if synthetic:
        data = synthetic_dynamics(seed, n=240, dimension=16)
        pairs = [(data.x_train, data.y_train, data.x_test, data.y_test)]
    else:
        prompts, _ = sentiment_prompts(6)
        split = int(len(prompts) * 0.7)
        train = flatten_valid_tokens(extract_hidden_states(load_causal_lm(model_name), prompts[:split]))
        test = flatten_valid_tokens(extract_hidden_states(load_causal_lm(model_name), prompts[split:]))
        pairs = [(train[index], train[index + 1], test[index], test[index + 1]) for index in range(len(train) - 1)]
    rows = []
    for layer, (x_train, y_train, x_test, y_test) in enumerate(pairs):
        model = fit_affine(x_train, y_train, ridge=1e-4)
        rows.append({"layer_pair": [layer, layer + 1], **regression_metrics(y_test, model.predict(x_test)).as_dict(), "parameters": model.parameter_count})
    return {"seed": seed, "model": "synthetic" if synthetic else model_name, "layers": rows}


def rank_sweep(seed: int, samples: int) -> dict[str, object]:
    data = synthetic_dynamics(seed, n=samples, dimension=32)
    rows = []
    for rank in [1, 2, 4, 8, 16, 32]:
        model = fit_low_rank(data.x_train, data.y_train, rank, ridge=1e-6)
        rows.append({"rank": rank, "parameters": model.parameter_count, **regression_metrics(data.y_test, model.predict(data.x_test)).as_dict()})
    save_curve([row["rank"] for row in rows], {"test R²": [row["r2"] for row in rows]}, "figures/rank_sweep.png", "rank", "held-out R²")
    return {"seed": seed, "rows": rows}


def trajectory(seed: int, samples: int) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    layers = []
    state = np.cumsum(rng.normal(size=(max(10, samples // 20), 24)), axis=0)
    for layer in range(6):
        smoothed = state.copy()
        for _ in range(layer):
            smoothed[1:-1] = 0.25 * smoothed[:-2] + 0.5 * smoothed[1:-1] + 0.25 * smoothed[2:]
        stats = trajectory_statistics(smoothed)
        layers.append({"layer": layer, "mean_step": float(stats.step_lengths.mean()), "mean_turn_angle": float(stats.turning_angles.mean()), "mean_displacement_cosine": float(stats.displacement_cosines.mean())})
    save_curve([row["layer"] for row in layers], {"turn angle": [row["mean_turn_angle"] for row in layers]}, "figures/trajectory_analysis.png", "layer", "mean radians")
    return {"seed": seed, "layers": layers, "warning": "PCA/summary geometry is descriptive, not causal evidence."}


def local_linearity(seed: int) -> dict[str, object]:
    import torch

    torch.manual_seed(seed)
    matrix = torch.randn(16, 16, dtype=torch.float64) / 4
    point = torch.randn(16, dtype=torch.float64)
    direction = torch.randn(16, dtype=torch.float64)
    def function(value: torch.Tensor) -> torch.Tensor:
        return value + torch.tanh(value @ matrix)
    sizes = np.geomspace(1e-3, 1.0, 10)
    errors = [relative_taylor_error(function, point, float(size) * direction) for size in sizes]
    save_curve(sizes.tolist(), {"Taylor error": errors}, "figures/local_linearity.png", "perturbation norm scale", "relative error")
    return {"seed": seed, "sizes": sizes.tolist(), "relative_errors": errors, "natural_transition_scale": float(torch.linalg.vector_norm(function(point) - point))}


def regime_comparison(seed: int, samples: int) -> dict[str, object]:
    data = synthetic_dynamics(seed, n=samples, dimension=16)
    global_model = fit_affine(data.x_train, data.y_train)
    rows: dict[str, object] = {"global": {**regression_metrics(data.y_test, global_model.predict(data.x_test)).as_dict(), "parameters": global_model.parameter_count}}
    for count in [2, 4, 8]:
        model = fit_regimes(data.x_train, data.y_train, count, seed)
        rows[f"local_{count}"] = {**regression_metrics(data.y_test, model.predict(data.x_test)).as_dict(), "parameters": model.parameter_count}
    return {"seed": seed, "models": rows}


def capstone(seed: int, samples: int) -> dict[str, object]:
    data = synthetic_dynamics(seed, n=samples, dimension=16)
    models = fit_candidate_models(data.x_train, data.y_train, seed, rank=4, regimes=2)
    interpolation = evaluate_models(models, data.x_test, data.y_test)
    shift = evaluate_models(models, data.x_shift, data.y_shift)
    save_curve(
        list(range(len(interpolation))),
        {"interpolation": [float(row["r2"]) for row in interpolation.values()], "shift": [float(row["r2"]) for row in shift.values()]},
        "figures/capstone.png",
        "model index: identity, mean, affine, rank-4, local-2, MLP",
        "R²",
    )
    return {
        "seed": seed,
        "samples": samples,
        "data": "synthetic 16-D activation-like states with low-rank and two-regime updates",
        "interpolation": interpolation,
        "distribution_shift": shift,
    }


def transformer_capstone(seed: int, model_name: str, samples: int) -> dict[str, object]:
    """Compare maps on a pretrained model with a question-style distribution shift."""
    count = max(36, min(samples, 120))
    declaratives, questions = prompt_domains(count)
    split = max(24, int(0.7 * count))
    bundle = load_causal_lm(model_name)
    train_layers = flatten_valid_tokens(extract_hidden_states(bundle, declaratives[:split]))
    test_layers = flatten_valid_tokens(extract_hidden_states(bundle, declaratives[split:]))
    shift_layers = flatten_valid_tokens(extract_hidden_states(bundle, questions[split:]))
    x_train, y_train = train_layers[0], train_layers[-1]
    x_test, y_test = test_layers[0], test_layers[-1]
    x_shift, y_shift = shift_layers[0], shift_layers[-1]
    rank = max(1, min(4, x_train.shape[1] // 2))
    models = fit_candidate_models(x_train, y_train, seed, rank=rank, regimes=2)
    interpolation = evaluate_models(models, x_test, y_test)
    shift = evaluate_models(models, x_shift, y_shift)

    output_embeddings = bundle.model.get_output_embeddings()
    if output_embeddings is None:
        raise ValueError(f"Model {model_name!r} does not expose output embeddings")
    weight = output_embeddings.weight.detach().float().cpu().numpy().astype(np.float64)

    def add_logit_agreement(
        table: dict[str, dict[str, float | int]],
        x: np.ndarray,
        y: np.ndarray,
    ) -> None:
        true_tokens = np.argmax(y @ weight.T, axis=1)
        for name, model in models.items():
            predicted_tokens = np.argmax(model.predict(x) @ weight.T, axis=1)
            table[name]["downstream_top1_agreement"] = float(np.mean(predicted_tokens == true_tokens))

    add_logit_agreement(interpolation, x_test, y_test)
    add_logit_agreement(shift, x_shift, y_shift)

    layer_pair_results: dict[str, object] = {}
    layer_pairs = list(dict.fromkeys([(i, i + 1) for i in range(len(train_layers) - 1)] + [(0, len(train_layers) - 1)]))
    for source, target in layer_pairs:
        pair_rank = max(1, min(4, train_layers[source].shape[1] // 2))
        pair_models = fit_candidate_models(
            train_layers[source],
            train_layers[target],
            seed,
            rank=pair_rank,
            regimes=2,
        )
        layer_pair_results[f"{source}->{target}"] = {
            "interpolation": evaluate_models(pair_models, test_layers[source], test_layers[target]),
            "distribution_shift": evaluate_models(
                pair_models,
                shift_layers[source],
                shift_layers[target],
            ),
        }
    save_curve(
        list(range(len(interpolation))),
        {
            "interpolation": [float(row["r2"]) for row in interpolation.values()],
            "questions": [float(row["r2"]) for row in shift.values()],
        },
        "figures/capstone.png",
        "model index (see results table)",
        "R²",
    )
    return {
        "seed": seed,
        "samples": count,
        "model": model_name,
        "layer_pair": [0, len(train_layers) - 1],
        "layer_pair_results": layer_pair_results,
        "data": (
            "pretrained token residual states; trained on short declarative sentences, "
            "tested on held-out declaratives and question-form prompts"
        ),
        "interpolation": interpolation,
        "distribution_shift": shift,
    }


def write_capstone_report(result: dict[str, object]) -> None:
    interpolation = result["interpolation"]
    shift = result["distribution_shift"]
    assert isinstance(interpolation, dict) and isinstance(shift, dict)
    is_transformer = "model" in result
    setup_detail = (
        f"Pretrained model `{result['model']}`, layer pair `{result['layer_pair']}`. "
        "The interpolation split holds out declarative prompts; the shift split uses questions."
        if is_transformer
        else "The interpolation set shares the generating distribution; the shift set moves the latent-coordinate mean by 1.2 standard units."
    )
    interpretation = (
        "These are measurements on a tiny pretrained model, not a broad claim about transformer families. Compare compact maps with identity, the mean, the full affine map, and the MLP. Top-1 agreement tests preservation through the frozen unembedding but is coarse when several tokens have close logits."
        if is_transformer
        else "This controlled system tests whether the estimators recover known low-complexity structure; it is not evidence that a pretrained transformer follows the same dynamics. Compare the low-rank and local models with the full affine map and MLP, not just against the mean baseline. A model is more scientifically interesting when it retains accuracy with fewer parameters and under shift."
    )
    failure_modes = (
        "The tiny model has hidden width two, so an affine map has only six parameters and rank one is nearly full capacity. Repeated prompt templates and shared tokens can inflate generalization; k-means assignments need not be causal regimes; the question shift is narrow; and top-token agreement hides logit calibration errors."
        if is_transformer
        else "A full affine map can be overparameterized; the generator favors low-rank and local structure; k-means assignments need not match causal regimes; and one mean shift is a narrow robustness test. Hyperparameter selection on the test set would also leak information."
    )
    lines = [
        "# Capstone results",
        "",
        "## Experimental setup",
        "",
        f"Seed `{result['seed']}`; `{result['samples']}` prompts/samples. {result['data']}. {setup_detail}",
        "",
        "## Quantitative results",
        "",
        "| model | interpolation MSE | interpolation R² | shift MSE | shift R² | top-1 agreement | parameters |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in interpolation.items():
        shifted = shift[name]
        agreement = shifted.get("downstream_top1_agreement", "n/a")
        if isinstance(agreement, float):
            agreement = f"{agreement:.3f}"
        lines.append(f"| {name} | {row['mse']:.6f} | {row['r2']:.4f} | {shifted['mse']:.6f} | {shifted['r2']:.4f} | {agreement} | {row['parameters']} |")
    lines.extend([
        "",
        "![Capstone comparison](../figures/capstone.png)",
        "",
        "## Interpretation",
        "",
        interpretation,
        "",
        "## Failure modes",
        "",
        failure_modes,
        "",
        "## What would falsify the linear-dynamics hypothesis?",
        "",
        "The useful version would be weakened if compact affine or piecewise-affine maps consistently failed to beat identity/mean baselines on held-out transformer activations, lost all advantage under modest shifts, or required nearly full-rank maps while small nonlinear models generalized reliably.",
        "",
        "## Follow-up experiments",
        "",
        "Repeat on pretrained residual streams across model families; split by prompt templates; measure downstream logits; compare natural with matched off-manifold perturbations; and intervene along predicted update directions.",
    ])
    Path("results/capstone.md").write_text("\n".join(lines) + "\n")


def emit(result: dict[str, object], output: Path | None, default: str) -> None:
    path = output or Path(default)
    save_json(result, path)
    print(f"saved {path}")

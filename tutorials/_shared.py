"""Shared executable demonstrations; each public tutorial supplies its own framing."""

import argparse

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

from linear_llm_lab.activations import extract_hidden_states, last_token_states, mean_pool_states
from linear_llm_lab.comparison import evaluate_models, fit_candidate_models
from linear_llm_lab.datasets import sentiment_prompts, synthetic_dynamics
from linear_llm_lab.jacobians import relative_taylor_error
from linear_llm_lab.linear_maps import fit_affine
from linear_llm_lab.low_rank import fit_low_rank
from linear_llm_lab.metrics import regression_metrics
from linear_llm_lab.models import DEFAULT_MODEL, load_causal_lm
from linear_llm_lab.probes import train_probe
from linear_llm_lab.regimes import fit_regimes
from linear_llm_lab.trajectories import trajectory_statistics


def parser(number: int) -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=f"Executable tutorial {number:02d}")
    result.add_argument("--seed", type=int, default=0)
    result.add_argument("--model", default=DEFAULT_MODEL)
    result.add_argument("--synthetic", action="store_true", help="Avoid model downloads")
    return result


def transformer_layers(model_name: str, synthetic: bool, seed: int) -> list[np.ndarray]:
    if synthetic:
        rng = np.random.default_rng(seed)
        base = rng.normal(size=(48, 32))
        return [base, base + 0.15 * np.tanh(base), base + 0.30 * np.tanh(base)]
    prompts, _ = sentiment_prompts(4)
    return last_token_states(extract_hidden_states(load_causal_lm(model_name), prompts))


def lesson_01(seed: int) -> None:
    rng = np.random.default_rng(seed)
    x = np.linspace(-3, 3, 300)[:, None]
    y = np.sin(x)
    train = np.abs(x[:, 0]) < 0.4
    global_map = fit_affine(x, y)
    local_map = fit_affine(x[train], y[train])
    global_local_mse = np.mean((global_map.predict(x[train]) - y[train]) ** 2)
    local_mse = np.mean((local_map.predict(x[train]) - y[train]) ** 2)
    a = rng.normal(size=(3, 2))
    u, v = rng.normal(size=(2, 2))
    superposition_error = np.linalg.norm((u + v) @ a.T - (u @ a.T + v @ a.T))
    print({"superposition_error": superposition_error, "global_local_mse": global_local_mse, "local_mse": local_mse})


def lesson_02(model_name: str, synthetic: bool, seed: int) -> None:
    layers = transformer_layers(model_name, synthetic, seed)
    rows = []
    for index, layer in enumerate(layers):
        norm = float(np.mean(np.linalg.norm(layer, axis=1)))
        if index:
            update = layer - layers[index - 1]
            cosine = regression_metrics(layer, layers[index - 1]).cosine
            rows.append({"layer": index, "norm": norm, "adjacent_cosine": cosine, "update_norm": float(np.mean(np.linalg.norm(update, axis=1)))})
        else:
            rows.append({"layer": index, "norm": norm})
    print(rows)


def lesson_03(model_name: str, synthetic: bool, seed: int) -> None:
    prompts, labels = sentiment_prompts(4)
    if synthetic:
        rng = np.random.default_rng(seed)
        layers = [rng.normal(size=(len(labels), 20)) for _ in range(3)]
        layers[1][:, 0] += labels * 3
        layers[2][:, 1] += labels * 4
    else:
        layers = mean_pool_states(extract_hidden_states(load_causal_lm(model_name), prompts))
    train_indices, test_indices = train_test_split(np.arange(len(labels)), test_size=0.35, random_state=seed, stratify=labels)
    scores = []
    for index, layer in enumerate(layers):
        result = train_probe(layer[train_indices], labels[train_indices], layer[test_indices], labels[test_indices], seed)
        scores.append({"layer": index, "train": result.train_accuracy, "validation": result.validation_accuracy})
    # The predictor ignores coordinate 0 even though a probe can decode it perfectly.
    unused = np.column_stack([labels * 4 - 2, np.arange(len(labels)) % 2])
    causal_predictor = unused[:, 1]
    intervention_changed_prediction = bool(np.any(causal_predictor != causal_predictor))
    print({"probe_scores": scores, "recoverable_but_unused_intervention_effect": intervention_changed_prediction})


def lesson_04(seed: int) -> None:
    data = synthetic_dynamics(seed, n=360, dimension=12)
    models = fit_candidate_models(data.x_train, data.y_train, seed, rank=4, regimes=2)
    print(evaluate_models(models, data.x_test, data.y_test))


def lesson_05(seed: int) -> None:
    data = synthetic_dynamics(seed, n=360, dimension=16)
    rows = []
    for rank in [1, 2, 4, 8, 16]:
        model = fit_low_rank(data.x_train, data.y_train, rank)
        rows.append({"rank": rank, "parameters": model.parameter_count, **regression_metrics(data.y_test, model.predict(data.x_test)).as_dict()})
    print(rows)


def lesson_06(seed: int) -> None:
    rng = np.random.default_rng(seed)
    trajectory = np.cumsum(rng.normal(size=(30, 12)), axis=0)
    trajectory[1:] = 0.8 * trajectory[1:] + 0.2 * np.arange(1, 30)[:, None]
    stats = trajectory_statistics(trajectory)
    projection = PCA(2).fit_transform(trajectory)
    print({"mean_step": float(stats.step_lengths.mean()), "mean_turning_angle": float(stats.turning_angles.mean()), "pca_variance": PCA(2).fit(trajectory).explained_variance_ratio_.tolist(), "projected_shape": projection.shape})


def lesson_07(seed: int) -> None:
    rng = np.random.default_rng(seed)
    context = rng.normal(size=(80, 20))
    direction = rng.normal(size=20)
    direction /= np.linalg.norm(direction)
    positive = context + direction + 0.1 * rng.normal(size=context.shape)
    negative = context - direction + 0.1 * rng.normal(size=context.shape)
    estimate = np.mean(positive - negative, axis=0)
    consistency = (positive - negative) @ (estimate / np.linalg.norm(estimate))
    readout = rng.normal(size=20)
    steering_effect = float((readout @ (context[0] + estimate)) - (readout @ context[0]))
    print({"mean_direction_consistency": float(consistency.mean()), "steering_readout_change": steering_effect, "caution": "steering one readout is causal evidence for that intervention, not a complete mechanism"})


def lesson_08(seed: int) -> None:
    torch.manual_seed(seed)
    point = torch.randn(12, dtype=torch.float64)
    direction = torch.randn(12, dtype=torch.float64)
    matrix = torch.randn(12, 12, dtype=torch.float64)

    def function(value: torch.Tensor) -> torch.Tensor:
        return torch.tanh(value @ matrix)
    sizes = [1.0, 0.3, 0.1, 0.03, 0.01]
    print([{"size": size, "relative_error": relative_taylor_error(function, point, size * direction)} for size in sizes])


def lesson_09(seed: int) -> None:
    rng = np.random.default_rng(seed)
    theta = rng.uniform(-np.pi, np.pi, 500)
    manifold = np.column_stack([np.cos(theta), np.sin(theta), theta / 3])
    off_manifold = manifold + rng.normal(scale=0.7, size=manifold.shape)
    def function(x: np.ndarray) -> np.ndarray:
        radius_error = x[:, [0]] ** 2 + x[:, [1]] ** 2 - 1
        return x[:, [0]] + x[:, [1]] ** 2 + 2 * radius_error**2
    model = fit_affine(manifold[:350], function(manifold[:350]))
    spectrum = PCA().fit(manifold).explained_variance_
    participation_ratio = float(spectrum.sum() ** 2 / np.sum(spectrum**2))
    print({"natural_mse": regression_metrics(function(manifold[350:]), model.predict(manifold[350:])).mse, "off_manifold_mse": regression_metrics(function(off_manifold), model.predict(off_manifold)).mse, "participation_ratio": participation_ratio})


def lesson_10(seed: int) -> None:
    data = synthetic_dynamics(seed, n=480, dimension=12)
    global_model = fit_affine(data.x_train, data.y_train)
    output = {"global": regression_metrics(data.y_test, global_model.predict(data.x_test)).as_dict()}
    for count in [2, 4, 8]:
        model = fit_regimes(data.x_train, data.y_train, count, seed)
        output[f"local_{count}"] = {**regression_metrics(data.y_test, model.predict(data.x_test)).as_dict(), "parameters": model.parameter_count}
    print(output)


def lesson_11(seed: int) -> None:
    rng = np.random.default_rng(seed)
    dynamics = np.array([[0.92, -0.25, 0.0], [0.25, 0.92, 0.0], [0.0, 0.0, 1.04]])
    state = rng.normal(size=3)
    trajectory = [state]
    for _ in range(25):
        trajectory.append(trajectory[-1] @ dynamics)
    estimate = fit_affine(np.asarray(trajectory[:-1]), np.asarray(trajectory[1:]))
    print({"true_eigenvalues": [str(value) for value in np.linalg.eigvals(dynamics)], "estimated_eigenvalues": [str(value) for value in np.linalg.eigvals(estimate.matrix)]})


def lesson_12(seed: int) -> None:
    rng = np.random.default_rng(seed)
    x = rng.uniform(-0.85, 0.85, 400)[:, None]
    next_x = x**2
    raw = fit_affine(x[:300], next_x[:300])
    phi = np.column_stack([x[:, 0], x[:, 0] ** 2, x[:, 0] ** 4])
    next_phi = np.column_stack([next_x[:, 0], next_x[:, 0] ** 2, next_x[:, 0] ** 4])
    lifted = fit_affine(phi[:300], next_phi[:300])
    print({"raw_state_mse": regression_metrics(next_x[300:], raw.predict(x[300:])).mse, "lifted_coordinate_mse": regression_metrics(next_phi[300:], lifted.predict(phi[300:])).mse})


def lesson_13(seed: int) -> None:
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(500, 10))
    transform = rng.normal(size=(10, 14))
    b = a @ transform + 0.02 * rng.normal(size=(500, 14))
    model = fit_affine(a[:350], b[:350])
    # A source-space probe vector transfers through predictions in the target space.
    print({"alignment": regression_metrics(b[350:], model.predict(a[350:])).as_dict(), "rectangular_shape": model.matrix.shape})


def lesson_14(seed: int) -> None:
    data = synthetic_dynamics(seed, n=600, dimension=16)
    models = fit_candidate_models(data.x_train, data.y_train, seed, rank=4, regimes=2)
    print({"interpolation": evaluate_models(models, data.x_test, data.y_test), "distribution_shift": evaluate_models(models, data.x_shift, data.y_shift)})


def run(number: int) -> None:
    args = parser(number).parse_args()
    if number == 1:
        lesson_01(args.seed)
    elif number == 2:
        lesson_02(args.model, args.synthetic, args.seed)
    elif number == 3:
        lesson_03(args.model, args.synthetic, args.seed)
    else:
        globals()[f"lesson_{number:02d}"](args.seed)

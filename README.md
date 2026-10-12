# Linear LLM Representations Lab

> Transformers are deeply nonlinear systems, yet their hidden representations often display surprisingly simple linear structure.

This research-oriented tutorial repository asks:

> **Why can a globally nonlinear transformer appear approximately linear on the hidden states it actually visits?**

It investigates whether the apparent simplicity reflects residual architecture, optimization, restricted activation manifolds, local Taylor structure, learned coordinate systems, or genuinely low-complexity latent dynamics. It does **not** teach that “LLMs are linear.” Every positive fit is paired with capacity, held-out, distribution-shift, or causal cautions.

```text
tokens
  ↓
nonlinear transformer
  ↓
hidden-state manifold
  ↓
┌───────────────────────────────┐
│ possible simple structures    │
│                               │
│ linear concept directions     │
│ layer-to-layer affine maps    │
│ straight token trajectories   │
│ cross-model alignment         │
│ local linear regimes          │
└───────────────────────────────┘
  ↓
question: how much of this structure is real, general, and causal?
```

## Start here

Install [uv](https://docs.astral.sh/uv/), then run:

```bash
uv sync
uv run pytest
uv run ruff check .
uv run pyright
```

Python 3.12 is pinned in `.python-version`; `uv` manages the environment and lockfile. Experiments prefer CUDA, then Apple MPS, then CPU. The synthetic tutorials and tests are CPU-friendly. Tutorials 02 and 03 download `sshleifer/tiny-gpt2` by default; pass `--synthetic` for an offline smoke run.

## Reading order

Each executable tutorial contains intuition, mathematics, an experiment, and interpretation/limitations.

| # | Tutorial | Central distinction |
|---:|---|---|
| 01 | `01_linear_maps.py` | global fit versus local Jacobian |
| 02 | `02_residual_stream.py` | additive updates versus global linearity |
| 03 | `03_linear_probes.py` | decodability versus causal computation |
| 04 | `04_layer_affine_maps.py` | learned map versus trivial baselines |
| 05 | `05_high_dimensional_trap.py` | predictability versus compressibility |
| 06 | `06_token_trajectories.py` | full geometry versus PCA appearance |
| 07 | `07_concept_directions.py` | decoding versus steering versus mechanism |
| 08 | `08_local_linearization.py` | infinitesimal truth versus useful radius |
| 09 | `09_activation_manifolds.py` | natural manifold versus ambient space |
| 10 | `10_global_vs_local.py` | global affine versus piecewise affine |
| 11 | `11_latent_dynamics.py` | descriptive spectra versus dynamics |
| 12 | `12_koopman_intuition.py` | nonlinear states versus simple coordinates |
| 13 | `13_cross_model_alignment.py` | reconstruction versus transferable behavior |
| 14 | `14_capstone.py` | interpolation versus structural generalization |

Run any lesson from the repository root:

```bash
uv run python tutorials/01_linear_maps.py
uv run python tutorials/02_residual_stream.py --synthetic
uv run python tutorials/14_capstone.py --seed 7
```

The concise notes in [`notes/`](notes/) supply the conceptual spine. Begin with [`linearity_taxonomy.md`](notes/linearity_taxonomy.md), then read the note matching each tutorial. [`failure_modes.md`](notes/failure_modes.md) and [`literature_map.md`](notes/literature_map.md) are useful before interpreting results.

## Experiments

All commands accept `--seed`; dataset-based commands also accept `--samples`. They save machine-readable results under `results/` and figures under `figures/`.

```bash
uv run python experiments/run_layer_linearity.py --synthetic
uv run python experiments/run_layer_linearity.py --model sshleifer/tiny-gpt2
uv run python experiments/run_rank_sweep.py
uv run python experiments/run_trajectory_analysis.py
uv run python experiments/run_local_linearity.py
uv run python experiments/run_regime_comparison.py
uv run python experiments/run_capstone.py
uv run python experiments/run_capstone.py --transformer --samples 60
```

Synthetic smoke runs generally take seconds on a laptop CPU. The first pretrained-model command may take a few minutes because it downloads weights; later runs reuse the Hugging Face cache. The default capstone is a controlled synthetic recovery and falsification test. Add `--transformer` for held-out last-token residual states, a declarative-to-question shift, and downstream top-token preservation. The most recently generated actual report is [`results/capstone.md`](results/capstone.md).

## Package map

Reusable, explicit implementations live in `src/linear_llm_lab/`: model/device loading, activation extraction, affine and low-rank regression, probes, trajectories, JVPs, local regimes, metrics, plotting, datasets, and matched model comparisons. Core functionality does not depend on notebooks.

## How to read a positive result

Do not collapse these claims:

```text
linear decodability
        ↓
linear predictability
        ↓
low-complexity linear predictability
        ↓
out-of-distribution generalization
        ↓
causal linear structure
```

Each step is substantially stronger. A probe can read information the model ignores. A `d×d` affine map has `d²+d` parameters. Residual connections make identity unusually competitive. Every smooth function is locally linear at a sufficiently small scale. PCA can make a curved high-dimensional path look straight. Prompt-template leakage can masquerade as generalization. Always inspect sample counts, parameter counts, baselines, train/test grouping, shift performance, and interventions.

## Research takeaway

> The interesting hypothesis is not that transformers are linear. It is that nonlinear representation learning may construct coordinates in which important parts of prediction become unusually simple.

The experiments here are instruments for trying to disprove that hypothesis as much as illustrate it.

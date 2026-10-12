# Linearity taxonomy

“Linear” names several non-equivalent claims.

| Claim | Operational test | What it does not establish |
|---|---|---|
| Linear representation | Controlled differences align with a subspace/direction | The model reads or uses it |
| Linear decodability | A linear probe predicts a label on held-out data | Linear computation or causality |
| Linear computation | The relevant update obeys superposition | Global model linearity |
| Affine mapping | `y ≈ Ax+b` | A small or unique mechanism |
| Local linearization | A Jacobian predicts sufficiently small perturbations | A large useful neighborhood |
| Low-rank structure | A compact `UVᵀ` map generalizes | Causal semantics |
| Piecewise-linear dynamics | A few local affine maps outperform one global map | That clusters are causal regimes |

Use the hierarchy: **decodability → predictability → low-complexity predictability → out-of-distribution generalization → causal linear structure**. Each arrow demands new evidence.


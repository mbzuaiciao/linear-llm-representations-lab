# Literature map

These four claims are related but not equivalent.

## 1. Token-to-token trajectory linearity

Hernandez et al., *Large language models implicitly learn to straighten neural sentence trajectories to construct a predictive representation of natural language* (arXiv:2311.04930, 2023), measures sentence-trajectory curvature and reports decreasing curvature in deeper layers. This concerns geometry across token positions, not affine maps across layers.

## 2. Layer-to-layer affine mappings

Yom Din, Karidi, Choshen, and Geva, *Jump to Conclusions: Short-Cutting Transformers With Linear Transformations* (arXiv:2303.09435, 2023), learns transformations from intermediate representations toward final-layer space and tests early prediction/shortcutting. This is predictive mapping, not automatically a low-rank or causal account.

## 3. Linear concept representations

Park, Choe, and Veitch, *The Linear Representation Hypothesis and the Geometry of Large Language Models* (arXiv:2311.03658, 2023; later conference versions may use 2024), formalizes representation using counterfactuals and relates it to probing and steering. Jiang, Rajendran, Ravikumar, Aragam, and Veitch, *On the Origins of Linear Representations in Large Language Models* (arXiv:2403.03867, 2024), gives a latent-variable account in which next-token prediction and optimization bias promote linear concept structure.

## 4. Cross-model affine alignment

Chen, Merullo, Stolfo, and Pavlick, *Transferring Linear Features Across Language Models With Model Stitching* (OpenReview preprint, surfaced in 2025–2026), studies affine residual-stream mappings and transfer of probes, steering vectors, and sparse-autoencoder features. Because the accessible record is a recent preprint and venue metadata may change, verify its final bibliographic status before formal citation.

## Relationship

Trajectory straightness is within a model across tokens; shortcut maps are within a model across depth; concept directions describe semantic contrasts; alignment maps compare coordinate systems across models. Any one may hold while the others fail. Shared restricted manifolds, residual updates, optimization, or learned coordinates could connect them, but that connection is a hypothesis to test.

Primary sources: [trajectory straightening](https://arxiv.org/abs/2311.04930), [Jump to Conclusions](https://arxiv.org/abs/2303.09435), [Linear Representation Hypothesis](https://arxiv.org/abs/2311.03658), [origins of linear representations](https://arxiv.org/abs/2403.03867), and [cross-model feature transfer](https://openreview.net/pdf?id=Qvvy0X63Fv).


# Failure modes

- **Overparameterized maps:** a `d×d` matrix has `d²` parameters. Report sample count, rank, and parameter-matched baselines.
- **Train/test leakage:** split prompt templates or domains, not only rows.
- **Decodability without causal use:** add intervention and mediation tests.
- **Local versus global:** every smooth function is locally linear; estimate the useful radius.
- **PCA artifacts:** score full-dimensional geometry.
- **Distribution dependence:** test natural and off-distribution activations.
- **Context dependence:** directions and maps can vary by prompt, layer, task, and conversational state.
- **Identity strength:** residual connections make `h_l` a strong predictor of `h_{l+1}`.
- **Metric ambiguity:** MSE, cosine, `R²`, and downstream logits answer different questions.


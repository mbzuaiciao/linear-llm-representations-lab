# Cross-model alignment

Given identical inputs, fit `h_B ≈ Ah_A+b` on training prompts and evaluate separate prompt families. A rectangular `A` naturally handles different hidden widths. Compare mean, PCA/Procrustes when applicable, random orthogonal, and learned affine baselines.

Behavioral tests—logits, transferred probes, or steering vectors—are more informative than reconstruction alone. Tokenizer alignment, layer choice, anisotropy, and duplicated corpora can dominate the result.


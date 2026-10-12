# Residual stream

A pre-norm transformer block can be sketched as `h' = h + attention(norm(h))`, followed by another additive MLP update. This makes differences, vector addition, and incremental writes natural objects of study. The attention weights, normalization, MLP activation, and their dependence on the entire context remain nonlinear. A small residual update can make adjacent layers highly cosine-similar even when the update carries decisive information.

Measure norms, adjacent-layer cosine, and update norms together. Identity is therefore a mandatory baseline for layer-to-layer prediction.


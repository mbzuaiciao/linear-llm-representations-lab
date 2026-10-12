# Trajectory straightening

For token states `h_t`, define displacements `Δ_t=h_{t+1}-h_t`. Straightness can be summarized by step length, turning angle, or `cos(Δ_t,Δ_{t+1})`. Compare the same statistic across layers and against token/order controls.

PCA projections discard dimensions. A visually straight 2-D path is only qualitative evidence; compute curvature in the original space and report sensitivity to normalization and special tokens.


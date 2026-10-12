# Local linearization

For differentiable `F`, `F(h+δ) = F(h) + J_F(h)δ + O(||δ||²)`. A Jacobian-vector product computes `Jδ` without materializing the full Jacobian. Plot error over perturbation scale and compare artificial directions with naturally occurring activation differences.

The scientific question is not whether Taylor's theorem works as `δ→0`; it is whether a neighborhood large enough to contain meaningful transitions has a stable, compressible Jacobian.


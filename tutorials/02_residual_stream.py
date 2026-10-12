"""Tutorial 02 — Residual stream. Intuition: blocks write additive updates. Mathematics: h[l+1]=h[l]+delta. Experiment: inspect layer norms, cosines, and update sizes. Interpretation: additive architecture is not global linearity."""
from _shared import run

if __name__ == "__main__":
    run(2)

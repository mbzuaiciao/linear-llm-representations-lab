# Koopman intuition

A nonlinear state update `x'=f(x)` can induce approximately linear evolution of observables `z=φ(x)`: `φ(x')≈Aφ(x)`. Polynomial lifting gives a concrete synthetic example. Representation learning might similarly choose coordinates that simplify prediction.

This is motivation, not a claim that transformers learn a Koopman operator. Finite learned coordinates usually give only an approximation, and a good one-step map may compound error over longer rollouts.


# Section 3.3 — EnKF–NR (Windowed Likelihood with Ensemble KF)

This stub outlines the planned EnKF–NR algorithm that combines a windowed EnKF with a Newton–Raphson MLE update of parameters.

## Idea
- Run EnKF over a short window with current θ and cache the innovation sequence and covariances.
- Define a windowed log-likelihood ℓ(θ) using EnKF-derived innovations.
- Compute gradients and Hessian via finite differences (with cached randomness) and perform Newton updates.

## Planned Implementation
- File: Sources/KalmanCore/estimation/EnKFNewtonRaphson.swift (scaffold present)
- Steps per window:
  1. Forecast/analysis over L steps; collect innovations {ν_t} and covariances {S_t}
  2. ℓ(θ) = Σ_t [ -0.5 (m log(2π) + log det S_t + ν_t^T S_t^{-1} ν_t) ]
  3. ∇ℓ and ∇²ℓ via finite differences with consistent random seeds
  4. Newton step with line search and bounds

## Status
- Scaffold class EnKFNewtonRaphson exists; likelihood evaluator to be implemented.

## References
- Pulido, M., et al. (2018). Section 3.
- Nocedal & Wright (2006). Numerical Optimization.
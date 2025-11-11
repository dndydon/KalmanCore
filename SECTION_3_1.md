# Section 3.1 — Augmented-state Ensemble Kalman Filter (EnKF)

This note summarizes the augmented-state EnKF that jointly estimates the system state x and parameters θ by operating on concatenated ensemble members [x; θ].

## Overview

- State augmentation: each ensemble member is z = [x; θ]
- Forecast:
  - State: x_{k+1} = M(x_k, θ_k, dt) + process noise
  - Parameters: evolve by chosen model (constant, random walk, AR(1))
- Analysis:
  - Compute anomalies for both predicted observations and augmented-state
  - Kalman gain uses cross-covariance between z and y
  - Optional inflation (multiplicative/additive) and simple localization

## Implementation

- File: Sources/KalmanCore/filters/EnsembleKalmanFilter.swift
- Config options:
  - ensembleSize, inflation (state-only), additiveInflation (state-only)
  - parameterEvolution: .constant, .randomWalk(Qθ), .ar1(ρ, Qθ)
  - usePerturbedObservations (stochastic vs deterministic)
  - localizationRadius: simple global scalar taper for state rows of P_zy
- Key methods:
  - initializeEnsemble(x0, P0, θ0, Pθ0)
  - forecast(ensemble, dt)
  - analyze(ensemble, observation)

## Notes

- Localization: current implementation supports a global scalar taper on state rows.
- Square-root deterministic variants may be added later.
- Diagnostics and more advanced localization can be layered onto the current API.

## References
- Evensen, G. (1994). The Ensemble Kalman Filter.
- Pulido, M., Tandeo, P., Bocquet, M., Carrassi, A., & Lucini, M. (2018). Section 3.

# Section 3.1 — Augmented-state Ensemble Kalman Filter (EnKF)

This note summarizes the augmented-state EnKF that jointly estimates the system state x and parameters θ by operating on concatenated ensemble members [x; θ].

## Overview

- State augmentation: each ensemble member is z = [x; θ]
- Forecast:
  - State: x_{k+1} = M(x_k, θ_k, dt) + process noise
  - Parameters: evolve by chosen model (constant, random walk, AR(1))
- Analysis:
  - Stochastic EnKF (perturbed observations) or deterministic square-root (ETKF-style) that updates anomalies in ensemble space
  - Kalman gain uses cross-covariance between z and y; mean update uses K d
  - Inflation (multiplicative/additive)
  - Covariance localization via Schur (Gaspari–Cohn) taper on state–obs cross-covariance P_{zy} (Identity/Partial obs supported)

## Implementation

- File: Sources/KalmanCore/filters/EnsembleKalmanFilter.swift
- Config options:
  - ensembleSize, inflation (state-only), additiveInflation (state-only)
  - parameterEvolution: .constant, .randomWalk(Qθ), .ar1(ρ, Qθ)
  - useSquareRootAnalysis: Bool (default false) — ETKF-style anomalies transform
  - usePerturbedObservations: Bool (default true) — stochastic EnKF when square-root disabled
  - localization: LocalizationConfig(method: .schurGaspariCohn1D(lengthScale: ℓ, periodic: Bool), observedIndices: [Int]?)
- Key methods:
  - initializeEnsemble(x0, P0, θ0, Pθ0)
  - forecast(ensemble, dt)
  - analyze(ensemble, observation)

## Minimal Usage Example (Lorenz-96 with square-root EnKF + Schur localization)

```swift
import KalmanCore

// Lorenz-96 model (additive noise) with n=40
let model = Lorenz96Model.standard(stochasticType: .additive)
let n = model.stateDimension

// Partial observation: every other variable, R = I * 0.5
let obsModel = PartialObservationModel(
  stateDimension: n,
  observedIndices: Array(stride(from: 0, to: n, by: 2)),
  noiseVariance: 0.5
)

// EnKF config: square-root analysis + Gaspari–Cohn localization (periodic 1D, ℓ=6)
let enkfConfig = EnKFConfig(
  ensembleSize: 20,
  inflation: 1.05,
  additiveInflation: nil,
  parameterEvolution: .constant,
  useSquareRootAnalysis: true,         // deterministic ETKF-style
  usePerturbedObservations: false,
  localization: LocalizationConfig(
    method: .schurGaspariCohn1D(lengthScale: 6.0, periodic: true)
  ),
  verbose: false
)
let enkf = EnsembleKalmanFilter(model: model, observationModel: obsModel, config: enkfConfig)

// Initialize augmented ensemble [x; θ]
let x0 = model.typicalInitialState()
let P0 = Matrix.identity(size: n) * 0.1
let theta0: [Double] = [0.3]
let Ptheta0 = Matrix.diagonal([0.05])
var Z = enkf.initializeEnsemble(x0: x0, P0: P0, theta0: theta0, Ptheta0: Ptheta0)

// One forecast + analysis cycle
let dt = 0.01
Z = enkf.forecast(ensemble: Z, dt: dt)

// Synthetic observation
let y = obsModel.generateObservation(state: x0)
Z = enkf.analyze(ensemble: Z, observation: y)

print("Analysis mean state:", Z.mean.prefix(5))
```

## Notes

- Localization support (Stage 1): builds an n×m taper for state–obs using 1D (optionally periodic) distances.
  - For Identity/Partial observation models, observed indices are inferred automatically; otherwise pass observedIndices or localization is skipped.
  - Parameters are not localized in this stage.
- Square-root analysis: anomalies are transformed with T = (I + (A_y^T R^{-1} A_y)/(N−1))^{-1/2}.
- Future work: augmented-ensemble (rSVD) and Bocquet perturbation update; hybrid localization.

## References
- Evensen, G. (1994). The Ensemble Kalman Filter.
- Pulido, M., Tandeo, P., Bocquet, M., Carrassi, A., & Lucini, M. (2018). Section 3.
- Farchi, A., & Bocquet, M. (2019). On the Efficiency of Covariance Localisation of the EnKF Using Augmented Ensembles.
- Gaspari, G., & Cohn, S.E. (1999). Construction of correlation functions in two and three dimensions.

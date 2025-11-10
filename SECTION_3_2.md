# Section 3.2 — EnKF–EM (Windowed Sequential Parameter Estimation)

This document outlines the windowed Ensemble Kalman Filter with Expectation–Maximization (EnKF–EM) used for sequential identification of stochastic parameters, following Pulido et al. (2018), Section 3.

## Overview

We consider a stochastic dynamical system

- dx/dt = M(x, θ) + σ(x, θ) ξ(t)
- y_k = H(x_k) + v_k,  v_k ~ N(0, R)

Goal: estimate parameters θ from observations y_{1:T}. In the additive-noise case, σ = θ₀·I (single parameter), we can derive a simple M-step.

EnKF–EM runs over short windows of length L:
1. Filtering: run EnKF (augmented state [x; θ]) over the window using current θ, collecting filtered means and uncertainty proxies.
2. M-step: update θ via a closed-form estimate (additive-noise) based on residuals between consecutive filtered means and deterministic drift.
3. Repeat for a small number of EM iterations per window, resetting θ and re-assimilating the window each time (tight loop).

## Algorithm (Additive-noise case)

Given time step Δt, state dimension n, window length L, filtered means x̄_t and covariance trace tr(P_t) after analysis:

1) Deterministic prediction at mean
- x̂_{t+1|t} = x̄_t + f(x̄_t)·Δt,  where f = M(·, θ) without noise.

2) Residual and uncertainty proxy
- r_t = x̄_{t+1} − x̂_{t+1|t}
- Accumulate S = Σ_t ( ||r_t||² + tr(P_{t+1}) )

3) M-step (variance), then parameter
- Var ≈ S / (n · (L−1) · Δt)
- θ₀ ← sqrt( max(ε, Var) )

Repeat steps 1–3 for a few EM iterations; after each M-step, reset θ and re-run the window.

Notes
- This proxy leverages filtered means and covariance traces as an E-step approximation, avoiding a full RTS smoother in the EnKF loop.
- For partial observations, EnKF handles H(·); diagnostics report innovation norms for monitoring assimilation quality.

## Implementation in KalmanCore

- Class: Sources/KalmanCore/estimation/EnKFEM.swift
  - runWindow(observations:dt:initialParameters:initialEnsemble:)
    - Tight EM loop with re-assimilation per M-step
    - Diagnostics: observedCount, observedFraction, meanInnovationNorm, innovationVariance
- Filter: Sources/KalmanCore/filters/EnsembleKalmanFilter.swift
  - Augmented-state EnKF with optional multiplicative/additive inflation and simple localization

## Minimal Usage

```swift
import KalmanCore

let model = Lorenz96Model.standard(stochasticType: .additive)
let n = model.stateDimension
let obsModel = PartialObservationModel(
  stateDimension: n,
  observedIndices: Array(stride(from: 0, to: n, by: 2)),
  noiseVariance: 1e-2
)

let enkfConfig = EnKFConfig(
  ensembleSize: 10,
  inflation: 1.05,
  additiveInflation: nil,
  parameterEvolution: .constant,
  usePerturbedObservations: false,
  localizationRadius: 0.9,
  verbose: false
)
let enkf = EnsembleKalmanFilter(model: model, observationModel: obsModel, config: enkfConfig)

let x0 = model.typicalInitialState()
let P0 = Matrix.identity(size: n) * 0.2
let theta0: [Double] = [0.15]
let Ptheta0 = Matrix.diagonal([0.05])
let Z0 = enkf.initializeEnsemble(x0: x0, P0: P0, theta0: theta0, Ptheta0: Ptheta0)

// Build a short observation window
aut var xTruth = x0
let dt = 0.01
let steps = 20
var observations: [[Double]] = []
for _ in 0..<steps {
  xTruth = model.transition(state: xTruth, parameters: [0.35], dt: dt)
  observations.append(obsModel.generateObservation(state: xTruth))
}

var emConfig = EnKFEMConfig()
emConfig.window = 15
emConfig.maxEMItersPerWindow = 2
emConfig.verbose = false

let enkfem = EnKFEM(model: model, observationModel: obsModel, enkf: enkf, config: emConfig)
let (newTheta, Zf, diags) = enkfem.runWindow(
  observations: observations,
  dt: dt,
  initialParameters: theta0,
  initialEnsemble: Z0
)

print("Updated θ:", newTheta)
print(diags.summary())
```

## References
- Pulido, M., Tandeo, P., Bocquet, M., Carrassi, A., & Lucini, M. (2018). Stochastic parameterization identification using ensemble Kalman filtering combined with maximum likelihood methods. Tellus A 70:1, 1–17.
- Evensen, G. (1994). The ensemble Kalman filter.

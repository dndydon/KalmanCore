# KalmanCore README

A Swift package implementing various Kalman filtering algorithms and estimation techniques.

##Project overview
- KalmanCore is a Swift Package (swift-tools-version 6.1) providing numerical primitives, stochastic state-space models, observation models, and parameter-estimation algorithms (EM and Newton–Raphson MLE).
- Targets: a single library target KalmanCore and a test target KalmanCoreTests.
- Platforms: macOS 13+, iOS 16+, tvOS 16+, watchOS 9+.
- External dependencies: none declared in Package.swift; numerics rely on Apple’s Accelerate framework.


For Basic Understanding: [Kalman Filter For Dummies](https://bilgin.esme.org/BitsAndBytes/KalmanFilterforDummies)

![KalmanFilter summary]
(https://bilgin.esme.org/Content/Images/BitsAndBytes/KalmanFilterForDummies/iteration_steps.gif)

## Features

- New in this release:
  - Linear Kalman Filter and Extended Kalman Filter with tests
  - Shared Linearization utilities (finite-diff F, H)
  - Likelihood utilities and Matrix utilities (Accelerate-backed with fallbacks)
  - EnKF–EM windowed estimator with diagnostics

### Core Components
- **Matrix Operations**: Optimized matrix algebra using Accelerate framework
- **Random Utilities**: Gaussian noise generation with covariance support
- **Ensemble Structures**: Ensemble representation for EnKF methods

### Stochastic Models [Section 2.1](SECTION_2_1.md)
Implementation of stochastic dynamical systems from Pulido et al. (2018):
- **Stochastic Dynamical System Protocol**: dx/dt = M(x, θ) + σ(x, θ)ξ(t)
- **Lorenz96 Model**: Classic chaotic system with multiple stochastic parameterization types
  - Additive noise: σ = θ₀·I
  - State-dependent noise: σᵢ = θ₀·|xᵢ|
  - Diagonal noise: σ = diag(θ)

### Observation Models
- **Observation Protocol**: y_k = H(x_k) + v_k
- **Linear Observation**: Full observation matrix support
- **Identity Observation**: Direct state observation
- **Partial Observation**: Observe subset of state variables

### Filters
- Kalman Filter (linear) — implemented
- Extended Kalman Filter (EKF) — implemented
- Unscented Kalman Filter (UKF) — planned
- Ensemble Kalman Filter (EnKF, augmented-state) — analysis implemented; ongoing enhancements
- Particle Filter — planned

### Estimation
- Expectation-Maximization (EM)
- Newton-Raphson Maximum Likelihood Estimation (MLE)
- EnKF–EM (windowed, additive-noise case) — see [Section 3.2](SECTION_3_2.md)

### Section 3 Documentation
- Augmented-state EnKF — see [Section 3.1](SECTION_3_1.md)
- EnKF–EM — see [Section 3.2](SECTION_3_2.md)
- EnKF–NR (stub) — see [Section 3.3](SECTION_3_3.md)

### Examples
- Lorenz96 system demonstrations
- Stochastic parameterization identification

## Installation

### Swift Package Manager

Add KalmanCore to your `Package.swift`:

```swift
dependencies: [
    .package(url: "https://github.com/dndydon/KalmanCore.git", from: "1.0.0")
]
```

Or add it in Xcode via File → Add Package Dependencies.

## Requirements

- Swift 5.9+
- macOS 13.0+ / iOS 16.0+ / tvOS 16.0+ / watchOS 9.0+

## Usage

### Using KF/EKF
- Linear KF
```swift
import KalmanCore

let F = Matrix.identity(size: 2)
let Q = Matrix.identity(size: 2) * 0.01
let H = Matrix.identity(size: 2)
let R = Matrix.identity(size: 2) * 0.1

let kf = KalmanFilter(F: F, Q: Q, H: H, R: R, x0: [0.0, 0.0], P0: Matrix.identity(size: 2))
let (state1, res1) = kf.step(y: [1.0, -1.0])
print("KF state:", state1.x, "LL+:", res1.logLikelihoodIncrement)
```
- Extended KF (nonlinear models)
```swift
import KalmanCore

let model = Lorenz96Model.standard(stochasticType: .additive)
let n = model.stateDimension
let obs = IdentityObservationModel(dimension: n, noiseVariance: 1e-2)

let ekf = ExtendedKalmanFilter(
  model: model,
  observationModel: obs,
  initialState: model.typicalInitialState(),
  initialCovariance: Matrix.identity(size: n) * 0.1,
  parameters: [0.3],
  dt: 0.01
)
let y = obs.generateObservation(state: ekf.state.x)
let (state2, res2) = ekf.step(y: y)
print("EKF state:", state2.x.prefix(3), "LL+:", res2.logLikelihoodIncrement)
```

### Section 3: Sequential Parameter Estimation (EnKF–EM)
See [Section 3.2](SECTION_3_2.md) for the algorithm outline and a minimal usage example combining the augmented-state EnKF with an EM M-step in the additive-noise case.

### Section 2.1: Stochastic Parameterization

```swift
import KalmanCore

// Create a Lorenz96 model with stochastic parameterization
let model = Lorenz96Model.standard(stochasticType: .additive)

// Define initial state and parameters
let initialState = model.typicalInitialState()
let parameters = [0.5]  // Stochastic noise level θ₀

// Simulate stochastic trajectory
let trajectory = model.simulateTrajectory(
    initialState: initialState,
    parameters: parameters,
    dt: 0.01,
    steps: 1000
)

// Create ensemble for EnKF
let ensemble = Ensemble(
    mean: initialState,
    covariance: Matrix.identity(size: 40) * 0.1,
    ensembleSize: 50
)

// Forecast ensemble forward
let forecast = model.forecastEnsemble(
    ensemble: ensemble,
    parameters: parameters,
    dt: 0.01
)

// Create observation model (observe every other variable)
let observedIndices = stride(from: 0, to: 40, by: 2).map { $0 }
let obsModel = PartialObservationModel(
    stateDimension: 40,
    observedIndices: observedIndices,
    noiseVariance: 0.5
)

// Generate observation
let observation = obsModel.generateObservation(state: initialState)

// Run demonstrations
Lorenz96Demo.runAll()
```

## License

[LICENSE](./LICENSE.txt)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

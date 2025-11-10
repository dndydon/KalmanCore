# KalmanCore

A Swift package implementing various Kalman filtering algorithms and estimation techniques.

For Basic Understanding: [Kalman Filter For Dummies](https://bilgin.esme.org/BitsAndBytes/KalmanFilterforDummies)

![KalmanFilter summary]
(https://bilgin.esme.org/Content/Images/BitsAndBytes/KalmanFilterForDummies/iteration_steps.gif)

## Features

### Core Components
- **Matrix Operations**: Optimized matrix algebra using Accelerate framework
- **Random Utilities**: Gaussian noise generation with covariance support
- **Ensemble Structures**: Ensemble representation for EnKF methods

### Stochastic Models (Section 2.1)
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

### Filters (In Development)
- Kalman Filter (linear)
- Extended Kalman Filter (EKF)
- Unscented Kalman Filter (UKF)
- Ensemble Kalman Filter (EnKF)
- Particle Filter

### Estimation (In Development)
- Expectation-Maximization (EM)
- Newton-Raphson Maximum Likelihood Estimation (MLE)
- EnKF–EM (windowed, additive-noise case) — see SECTION_3_2.md

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

### Section 3: Sequential Parameter Estimation (EnKF–EM)
See SECTION_3_2.md for the algorithm outline and a minimal usage example combining the augmented-state EnKF with an EM M-step in the additive-noise case.

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

[./LICENSE.txt]

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

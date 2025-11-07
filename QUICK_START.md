# Quick Start Guide - Section 2.1

## Installation

This is a Swift Package. Add it to your project:

### In Xcode
File → Add Package Dependencies → Enter repository URL

### In Package.swift
```swift
dependencies: [
    .package(url: "https://github.com/dndydon/KalmanCore.git", from: "1.0.0")
]
```

## Quick Examples

### 1. Simulate a Stochastic System

```swift
import KalmanCore

// Create Lorenz96 model
let model = Lorenz96Model.standard(stochasticType: .additive)

// Set up initial conditions
let x0 = model.typicalInitialState()
let theta = [0.5]  // Stochastic parameter

// Simulate
let trajectory = model.simulateTrajectory(
    initialState: x0,
    parameters: theta,
    dt: 0.01,
    steps: 1000
)

print("Final state mean: \(trajectory.last!.reduce(0, +) / 40.0)")
```

### 2. Work with Ensembles

```swift
// Create ensemble
let ensemble = Ensemble(
    mean: x0,
    covariance: Matrix.identity(size: 40) * 0.1,
    ensembleSize: 50
)

print("Initial spread: \(ensemble.spread)")

// Forecast ensemble
let forecast = model.forecastEnsemble(
    ensemble: ensemble,
    parameters: theta,
    dt: 0.01
)

print("Forecast spread: \(forecast.spread)")
```

### 3. Generate Observations

```swift
// Observe every other variable
let observedIndices = stride(from: 0, to: 40, by: 2).map { $0 }

let obsModel = PartialObservationModel(
    stateDimension: 40,
    observedIndices: observedIndices,
    noiseVariance: 0.5
)

let observation = obsModel.generateObservation(state: x0)
print("Observed \(observation.count) variables")
```

### 4. Matrix Operations

```swift
// Create matrices
let A = Matrix.identity(size: 3)
let B = Matrix.diagonal([1.0, 2.0, 3.0])

// Operations
let C = A + B
let D = 2.0 * C
let E = D.transposed

// Matrix-vector multiplication
let x = [1.0, 2.0, 3.0]
let y = B.multiply(vector: x)
```

### 5. Run All Demos

```swift
// See everything in action
Lorenz96Demo.runAll()
```

## Key Classes

| Class | Purpose |
|-------|---------|
| `Matrix` | Accelerate-optimized matrix operations |
| `Ensemble` | Ensemble of state vectors for EnKF |
| `Lorenz96Model` | Chaotic system with stochastic forcing |
| `PartialObservationModel` | Observe subset of state variables |
| `RandomUtils` | Gaussian noise generation |

## Stochastic Parameterization Types

```swift
// Additive noise: σ = θ₀·I
Lorenz96Model(dimension: 40, forcing: 8.0, stochasticType: .additive)

// State-dependent: σᵢ = θ₀·|xᵢ|
Lorenz96Model(dimension: 40, forcing: 8.0, stochasticType: .stateDependent)

// Diagonal: σ = diag(θ)
Lorenz96Model(dimension: 40, forcing: 8.0, stochasticType: .diagonal)
```

## File Structure

```
Sources/KalmanCore/
├── core/                    # Core utilities
│   ├── Matrix.swift
│   ├── RandomUtils.swift
│   └── Ensemble.swift
├── models/                  # Dynamical systems
│   ├── StochasticDynamicalSystem.swift
│   └── Lorenz96Model.swift
├── observations/           # Observation operators
│   └── ObservationModel.swift
└── examples/              # Demonstrations
    └── Lorenz96Demo.swift
```

## Build & Test

```bash
# Build
swift build

# Run tests (when available)
swift test

# Generate documentation
swift package generate-documentation
```

## What's Next?

This implementation covers Section 2.1 (framework setup). Future additions:
- Section 2.2: Ensemble Kalman Filter
- Section 2.3: Maximum Likelihood parameter estimation
- Section 3: Combined EnKF-ML algorithm

## Learn More

- Full documentation: `SECTION_2_1.md`
- Paper: [Pulido et al. (2018)](https://doi.org/10.1080/16000870.2018.1442099)
- Examples: Run `Lorenz96Demo.runAll()`

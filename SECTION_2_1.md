# Section 2.1 Implementation: Stochastic Parameterization Framework

## Overview

This document describes the implementation of Section 2.1 from:

> Pulido, M., Tandeo, P., Bocquet, M., Carrassi, A., & Lucini, M. (2018).  
> Stochastic parameterization identification using ensemble Kalman filtering combined with maximum likelihood methods.  
> *Tellus A: Dynamic Meteorology and Oceanography*, 70:1, 1-17.  
> DOI: [10.1080/16000870.2018.1442099](https://doi.org/10.1080/16000870.2018.1442099)

Section 2.1 establishes the mathematical framework for stochastic dynamical systems used in parameter identification with ensemble Kalman filtering.

## Mathematical Framework

### Stochastic Dynamical System

The framework models dynamical systems as:

```
dx/dt = M(x, θ) + σ(x, θ)ξ(t)
```

Where:
- **x**: State vector (dimension n)
- **θ**: Parameter vector to be identified (dimension p)
- **M(x, θ)**: Deterministic dynamics (drift term)
- **σ(x, θ)**: Stochastic parameterization (diffusion matrix)
- **ξ(t)**: White noise process

### Discretization

Using the Euler-Maruyama scheme:

```
x_{k+1} = x_k + M(x_k, θ)Δt + σ(x_k, θ)√(Δt)W_k
```

Where W_k ~ N(0, I) are independent standard normal random variables.

### Observation Model

Observations are related to the state by:

```
y_k = H(x_k) + v_k
```

Where:
- **y_k**: Observation vector at time k
- **H**: Observation operator (can be nonlinear)
- **v_k**: Observation noise ~ N(0, R)

## Implementation Structure

### Core Components

#### 1. Matrix Operations (`core/Matrix.swift`)
- Accelerate-optimized matrix operations
- Matrix multiplication, addition, transpose
- Identity and diagonal matrix constructors
- Frobenius norm and trace computations

#### 2. Random Utilities (`core/RandomUtils.swift`)
- Gaussian noise generation using Box-Muller transform
- Covariance-aware noise generation
- Cholesky decomposition for general covariances
- Resampling methods for particle filters

#### 3. Ensemble Structure (`core/Ensemble.swift`)
- Ensemble representation for EnKF
- Mean and covariance computation
- Anomaly matrix calculation
- Covariance inflation methods

### Stochastic Models

#### Protocol (`models/StochasticDynamicalSystem.swift`)
Defines the interface for stochastic dynamical systems:
- `deterministicDynamics(state:parameters:)` → Drift term M(x, θ)
- `stochasticParameterization(state:parameters:)` → Diffusion matrix σ(x, θ)
- `transition(state:parameters:dt:)` → Euler-Maruyama step
- `simulateTrajectory(...)` → Generate time series
- `forecastEnsemble(...)` → Ensemble forecasting

#### Lorenz96 Model (`models/Lorenz96Model.swift`)
Classic chaotic system implementation:

**Deterministic dynamics:**
```
dx_i/dt = (x_{i+1} - x_{i-2})x_{i-1} - x_i + F
```

**Stochastic parameterization types:**
1. **Additive**: σ = θ₀·I (constant noise)
2. **State-dependent**: σᵢ = θ₀·|xᵢ| (multiplicative noise)
3. **Diagonal**: σ = diag(θ₀, θ₁, ..., θₙ₋₁) (component-specific noise)

**Features:**
- Periodic boundary conditions
- Typical attractor forcing (F=8.0)
- Spin-up methods
- Lyapunov exponent calculation
- Climatological statistics

### Observation Models

#### Protocol (`observations/ObservationModel.swift`)
Defines observation operators:
- `observationOperator(state:)` → H(x)
- `generateObservation(state:)` → H(x) + noise
- `observeEnsemble(ensemble:)` → Apply H to ensemble

#### Implementations
1. **LinearObservationModel**: y = Hx + v
2. **IdentityObservationModel**: y = x + v (observe all states)
3. **PartialObservationModel**: Observe subset of state variables

## Usage Examples

### Basic Stochastic Simulation

```swift
// Create Lorenz96 model
let model = Lorenz96Model.standard(stochasticType: .additive)

// Initial conditions
let initialState = model.typicalInitialState()
let parameters = [0.5]  // θ₀ = 0.5

// Simulate trajectory
let trajectory = model.simulateTrajectory(
    initialState: initialState,
    parameters: parameters,
    dt: 0.01,
    steps: 1000
)
```

### Ensemble Forecasting

```swift
// Create initial ensemble
let ensemble = Ensemble(
    mean: initialState,
    covariance: Matrix.identity(size: 40) * 0.1,
    ensembleSize: 50
)

// Forecast forward
let forecast = model.forecastEnsemble(
    ensemble: ensemble,
    parameters: parameters,
    dt: 0.01
)

print("Ensemble spread: \(forecast.spread)")
```

### Observation Generation

```swift
// Partial observation (every other variable)
let observedIndices = stride(from: 0, to: 40, by: 2).map { $0 }
let obsModel = PartialObservationModel(
    stateDimension: 40,
    observedIndices: observedIndices,
    noiseVariance: 0.5
)

// Generate synthetic observation
let observation = obsModel.generateObservation(state: trueState)
```

### Running Demonstrations

```swift
import KalmanCore

// Run all section 2.1 demonstrations
Lorenz96Demo.runAll()

// Or run individually
Lorenz96Demo.demonstrateBasicSimulation()
Lorenz96Demo.demonstrateEnsembleForecasting()
Lorenz96Demo.demonstrateObservationModel()
Lorenz96Demo.compareStochasticTypes()
```

## File Organization

```
KalmanCore/Sources/KalmanCore/
├── core/
│   ├── Matrix.swift              # Matrix operations
│   ├── RandomUtils.swift         # Random number generation
│   └── Ensemble.swift            # Ensemble structure
├── models/
│   ├── StochasticDynamicalSystem.swift  # Protocol definition
│   └── Lorenz96Model.swift              # Example implementation
├── observations/
│   └── ObservationModel.swift    # Observation operators
└── examples/
    └── Lorenz96Demo.swift        # Demonstrations
```

## Next Steps

Section 2.1 provides the foundation for:

1. **Section 2.2**: Ensemble Kalman Filter implementation
2. **Section 2.3**: Parameter identification using maximum likelihood
3. **Section 3**: Combined EnKF-ML algorithm for stochastic parameter estimation

The current implementation provides all necessary components for data assimilation and parameter estimation workflows.

## References

- Lorenz, E. N. (1996). Predictability: A problem partly solved. *Seminar on Predictability*, 1, 1-18.
- Pulido et al. (2018). Stochastic parameterization identification using ensemble Kalman filtering. *Tellus A*, 70:1, 1-17.

## Testing

Build the package:
```bash
cd /path/to/KalmanCore
swift build
```

Run tests (when implemented):
```bash
swift test
```

## Contributing

This is an open-source implementation suitable for research and educational purposes. Contributions are welcome for:
- Additional stochastic model implementations
- Performance optimizations
- Extended Kalman filter variants
- Parameter estimation algorithms

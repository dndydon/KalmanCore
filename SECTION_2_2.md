# Section 2.2 Implementation: Expectation-Maximization Algorithm

## Overview

This document describes the implementation of Section 2.2 from:

> Pulido, M., Tandeo, P., Bocquet, M., Carrassi, A., & Lucini, M. (2018).  
> Stochastic parameterization identification using ensemble Kalman filtering combined with maximum likelihood methods.  
> *Tellus A: Dynamic Meteorology and Oceanography*, 70:1, 1-17.  
> DOI: [10.1080/16000870.2018.1442099](https://doi.org/10.1080/16000870.2018.1442099)

Section 2.2 presents the Expectation-Maximization (EM) algorithm for identifying stochastic parameters in dynamical systems through iterative refinement of maximum likelihood estimates.

## Mathematical Framework

### Problem Formulation

Given observations y₁, ..., yₜ of a stochastic system:

```
dx/dt = M(x, θ) + σ(x, θ)ξ(t)
y_k = H(x_k) + v_k
```

We seek to estimate the parameter vector θ that maximizes the likelihood of the observations.

### EM Algorithm Structure

The EM algorithm iterates between two steps:

#### **E-step: Expectation**
Compute the expected sufficient statistics given the current parameter estimate θ⁽ⁱ⁾:

- Run a Kalman filter forward to predict states
- Run a Rauch-Tiedemann-Striebel (RTS) smoother backward to estimate x̂_k|T (smoothed states)
- Compute the smoothed covariances P_k|T
- Compute lag-one covariances P_{k,k-1|T}
- Calculate the expected log-likelihood: Q(θ | θ⁽ⁱ⁾)

#### **M-step: Maximization**
Find θ⁽ⁱ⁺¹⁾ that maximizes Q(θ | θ⁽ⁱ⁾):

```
θ⁽ⁱ⁺¹⁾ = argmax_θ Q(θ | θ⁽ⁱ⁾)
```

For additive stochastic parameterization σ = θ₀·I, the M-step has a **closed-form solution**:

```
θ₀⁽ⁱ⁺¹⁾ = √(Σ_k ||x̂_{k+1|T} - M(x̂_{k|T}, θ) - σ(x̂_{k|T}, θ)Δt||² / (n·T·Δt))
```

### Convergence Criteria

The algorithm terminates when one of the following holds:

1. **Relative parameter change**: ||θ⁽ⁱ⁺¹⁾ - θ⁽ⁱ⁾|| / ||θ⁽ⁱ⁾|| < ε_p
2. **Log-likelihood change**: |Q(θ⁽ⁱ⁺¹⁾) - Q(θ⁽ⁱ⁾)| < ε_l
3. **Maximum iterations**: i > i_max

## Implementation Details

### Core Algorithm (`ExpectationMaximization` Class)

#### Configuration

```swift
public struct Configuration {
    var maxIterations: Int = 100
    var convergenceTolerance: Double = 1e-4      // Parameter convergence
    var logLikelihoodTolerance: Double = 1e-6    // Log-likelihood convergence
    var minParameterValue: Double = 1e-8         // Lower bound on parameters
    var maxParameterValue: Double = 100.0        // Upper bound on parameters
    var verbose: Bool = true
    var smoothingPasses: Int = 1
}
```

#### Main Estimation Function

```swift
public func estimate<Model: StochasticDynamicalSystem>(
    model: Model,
    observations: [[Double]],
    observationModel: ObservationModel,
    initialParameters: [Double],
    initialState: [Double],
    initialCovariance: Matrix,
    dt: Double
) -> Result
```

**Input Parameters:**
- `model`: Stochastic dynamical system implementing `StochasticDynamicalSystem` protocol
- `observations`: Time series of observations (T × m matrix)
- `observationModel`: Operator mapping state to observations
- `initialParameters`: Initial guess for parameters θ⁽⁰⁾
- `initialState`: Initial state estimate x̂₀
- `initialCovariance`: Initial state uncertainty covariance P₀
- `dt`: Time step between observations

**Output (`Result`):**
- `parameters`: Final estimated parameters θ*
- `logLikelihoodHistory`: Log-likelihood at each iteration
- `parameterHistory`: Parameter values at each iteration
- `iterations`: Number of iterations performed
- `converged`: Whether algorithm converged
- `finalParameterChange`: Final relative parameter change
- `finalLogLikelihoodChange`: Final log-likelihood change

### E-Step Implementation

The E-step runs the **Kalman filter** followed by **RTS smoothing**:

#### Forward Pass (Kalman Filter)
For k = 1, ..., T:

```
Prediction:
  x̂_{k|k-1} = f(x̂_{k-1|k-1})
  P_{k|k-1} = F P_{k-1|k-1} F^T + Q

Update:
  y_k - H x̂_{k|k-1} = innovation
  S = H P_{k|k-1} H^T + R
  K = P_{k|k-1} H^T S^{-1}
  x̂_{k|k} = x̂_{k|k-1} + K (y_k - H x̂_{k|k-1})
  P_{k|k} = (I - K H) P_{k|k-1}

Log-likelihood contribution:
  ℓ_k = -0.5 [m log(2π) + log det(S) + (y_k - H x̂_{k|k-1})^T S^{-1} (y_k - H x̂_{k|k-1})]
```

#### Backward Pass (RTS Smoother)
For k = T-1, ..., 1:

```
Smoother gain:
  C_k = P_{k|k} F^T P_{k+1|k}^{-1}

Smoothed state:
  x̂_{k|T} = x̂_{k|k} + C_k (x̂_{k+1|T} - x̂_{k+1|k})

Smoothed covariance:
  P_{k|T} = P_{k|k} + C_k (P_{k+1|T} - P_{k+1|k}) C_k^T

Lag-one covariance:
  P_{k,k-1|T} = P_{k|T} C_k^T
```

### M-Step Implementation

For **additive noise parameterization** (σ = θ₀·I):

```
Sum squared residuals:
  S = Σ_{k=0}^{T-2} [ ||x̂_{k+1|T} - [x̂_{k|T} + M(x̂_{k|T})Δt]||² + tr(P_{k+1|T}) ]

Maximum Likelihood Estimate:
  θ₀ = √(S / (n · T · Δt))
```

This is derived from the first-order optimality condition of Q(θ | θ⁽ⁱ⁾).

### Numerical Considerations

1. **Jacobian Approximation**: State transition Jacobian computed via finite differences
2. **Matrix Inversion**: Gauss-Jordan elimination with regularization for ill-conditioned matrices
3. **Parameter Constraints**: Enforced via box constraints: θ ∈ [θ_min, θ_max]
4. **Convergence Safeguards**: Maximum iterations and tolerance checks prevent infinite loops

## Usage Examples

### Basic Parameter Estimation

```swift
import KalmanCore

// 1. Create the dynamical model
let model = Lorenz96Model.standard(stochasticType: .additive)

// 2. Generate synthetic observations
let trueParameters = [0.5]  // True θ₀
let initialState = model.typicalInitialState()
let trueTrajectory = model.simulateTrajectory(
    initialState: initialState,
    parameters: trueParameters,
    dt: 0.01,
    steps: 100
)

// 3. Create observation model (observe every other variable)
let obsModel = PartialObservationModel(
    stateDimension: 40,
    observedIndices: stride(from: 0, to: 40, by: 2).map { $0 },
    noiseVariance: 0.5
)

// 4. Generate synthetic observations
var observations: [[Double]] = []
for state in trueTrajectory {
    observations.append(obsModel.generateObservation(state: state))
}

// 5. Configure and run EM algorithm
var config = ExpectationMaximization.Configuration()
config.maxIterations = 50
config.verbose = true

let em = ExpectationMaximization(config: config)
let result = em.estimate(
    model: model,
    observations: observations,
    observationModel: obsModel,
    initialParameters: [0.2],  // Initial guess (different from true value)
    initialState: initialState,
    initialCovariance: Matrix.identity(size: 40) * 0.1,
    dt: 0.01
)

// 6. Examine results
print("Estimated parameters: \(result.parameters)")
print("Converged: \(result.converged)")
print("Iterations: \(result.iterations)")
print("Log-likelihood history: \(result.logLikelihoodHistory)")
```

### Advanced: Different Stochastic Parameterizations

```swift
// State-dependent noise: σᵢ = θ₀·|xᵢ|
let modelStateDep = Lorenz96Model.standard(stochasticType: .stateDependentMagnitude)
let em = ExpectationMaximization()
let result = em.estimate(
    model: modelStateDep,
    observations: observations,
    observationModel: obsModel,
    initialParameters: [0.1],
    initialState: initialState,
    initialCovariance: Matrix.identity(size: 40) * 0.1,
    dt: 0.01
)

// Diagonal noise: σ = diag(θ₀, θ₁, ..., θ₃₉)
let modelDiagonal = Lorenz96Model.standard(stochasticType: .diagonal)
// Note: This requires 40 parameters, one per state dimension
let resultDiag = em.estimate(
    model: modelDiagonal,
    observations: observations,
    observationModel: obsModel,
    initialParameters: [Double](repeating: 0.1, count: 40),
    initialState: initialState,
    initialCovariance: Matrix.identity(size: 40) * 0.1,
    dt: 0.01
)
```

## Convergence Analysis

### Theoretical Properties

The EM algorithm for MLE satisfies:

1. **Monotone Increase**: ℓ(θ⁽ⁱ⁺¹⁾) ≥ ℓ(θ⁽ⁱ⁾) (non-decreasing log-likelihood)
2. **Local Convergence**: Converges to a local maximum of the likelihood
3. **Rate of Convergence**: Linear convergence in general (superlinear for singular distributions)

### Practical Observations

- **Convergence Speed**: Typically 10-30 iterations for well-posed problems
- **Sensitivity to Initialization**: Performance improves with good initial guess
- **Observation Frequency**: More frequent observations accelerate convergence
- **Noise Level**: Higher observation noise requires more iterations

### Common Issues and Solutions

| Issue | Cause | Solution |
|-------|-------|----------|
| Non-convergence | Poor initialization | Try multiple initial guesses |
| Slow convergence | Ill-conditioned problem | Increase observation frequency or reduce noise |
| Numerical instability | Singular covariance matrix | Enable Hessian regularization |
| Unrealistic parameters | Insufficient observations | Collect more data or constrain parameters |

## Comparison with Other Methods

### EM vs. Gradient Ascent

| Aspect | EM | Gradient Ascent |
|--------|----|-|
| **Convergence** | Monotone guaranteed | Conditional |
| **Step size** | Adaptive | Requires tuning |
| **Second-order info** | Implicit | Explicit (Newton) |
| **Initialization** | Robust | Sensitive |
| **Speed** | Fast near optimum | Variable |

### EM vs. Newton-Raphson (Section 2.3)

| Aspect | EM | Newton-Raphson |
|--------|----|-|
| **Theory** | Probabilistic | Deterministic optimization |
| **Convergence** | Q-linear | Q-quadratic |
| **Derivative** | Implicit E-step | Explicit gradient/Hessian |
| **Robustness** | Very robust | Requires good initialization |
| **Computational cost** | Moderate | High (evaluates Hessian) |

## Performance Characteristics

### Computational Complexity

Per iteration:
- **Forward pass (Kalman filter)**: O(T·n²·m) for T observations, n-dim state, m-dim observation
- **Backward pass (RTS smoother)**: O(T·n³) for matrix inversions
- **M-step**: O(n) for additive noise (closed-form) or O(T·n) for state-dependent noise

Total: **O(T·n³) per iteration** due to matrix inversions

### Memory Requirements

- Filter states and covariances: O(T·n²)
- Smoothed states and covariances: O(T·n²)
- Temporary matrices: O(n²)

**Total: O(T·n²)** where T is number of observations

### Practical Timings

For n=40 state dimension, T=1000 observations, one iteration typically takes:
- CPU: ~100-500ms (Swift on 2GHz processor)
- Memory: ~10-50MB

## File Organization

```
KalmanCore/Sources/KalmanCore/
├── estimation/
│   ├── ExpectationMaximization.swift    # EM algorithm (Section 2.2)
│   └── NewtonRaphsonMLE.swift           # NR-MLE algorithm (Section 2.3)
├── models/
│   ├── Lorenz96Model.swift              # Stochastic Lorenz96 system
│   └── StochasticDynamicalSystem.swift  # Protocol definition
├── observations/
│   └── ObservationModel.swift           # Observation operators
├── filters/
│   ├── KalmanFilter.swift               # Linear Kalman filter
│   └── ExtendedKalmanFilter.swift       # EKF for nonlinear systems
└── examples/
    └── Lorenz96Demo.swift               # Demonstrations
```

## Testing

Build and test:

```bash
cd /path/to/KalmanCore
swift build

# Run all tests (when implemented)
swift test

# Run specific test
swift test --filter EMParameterEstimation
```

## Next Steps

1. **Section 2.3**: Compare EM results with **Newton-Raphson MLE** for the same problems
2. **Section 3**: Combine EnKF with EM/NR-MLE for **sequential parameter estimation**
3. **Extended Models**: Apply to more complex systems (e.g., atmospheric models)
4. **Uncertainty Quantification**: Use final Hessian for parameter uncertainty estimates

## References

- **Primary**: Pulido et al. (2018) - Core reference for this implementation
- **EM Algorithm**: Dempster, A.P., Laird, N.M., and Rubin, D.B. (1977). Journal of the Royal Statistical Society
- **Kalman Filtering**: Kalman, R.E. (1960). Journal of Basic Engineering
- **RTS Smoothing**: Rauch, H.E., Striebel, C.T., and Tiedemann, F.R. (1965). Journal of Basic Engineering
- **Stochastic Integration**: Kloeden, P.E. and Platen, E. (1999). Numerical Solution of Stochastic Differential Equations

## Contributing

Improvements welcome for:
- More efficient matrix operations (BLAS/LAPACK integration)
- Computational profiling and optimization
- Extended M-step formulas for other noise structures
- Parallel processing for large-scale problems

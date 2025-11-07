# Section 2.3 Implementation: Newton-Raphson Maximum Likelihood Estimation

## Overview

This document describes the implementation of Section 2.3 from:

> Pulido, M., Tandeo, P., Bocquet, M., Carrassi, A., & Lucini, M. (2018).  
> Stochastic parameterization identification using ensemble Kalman filtering combined with maximum likelihood methods.  
> *Tellus A: Dynamic Meteorology and Oceanography*, 70:1, 1-17.  
> DOI: [10.1080/16000870.2018.1442099](https://doi.org/10.1080/16000870.2018.1442099)

Section 2.3 presents the Newton-Raphson method for Maximum Likelihood Estimation (MLE), providing an alternative optimization approach for identifying stochastic parameters via direct gradient-based optimization of the likelihood function.

## Mathematical Framework

### Problem Formulation

Given observations y₁, ..., yₜ of a stochastic system:

```
dx/dt = M(x, θ) + σ(x, θ)ξ(t)
y_k = H(x_k) + v_k
```

We seek to find the parameter θ that maximizes the log-likelihood:

```
ℓ(θ) = Σ_{k=1}^T ℓ_k(θ)
```

where each contribution comes from the innovation distribution at time k.

### Newton-Raphson Optimization

The Newton-Raphson method iterates:

```
θ⁽ⁱ⁺¹⁾ = θ⁽ⁱ⁾ + α H⁽ⁱ⁾⁻¹ ∇ℓ(θ⁽ⁱ⁾)
```

Where:
- **∇ℓ(θ)**: Gradient (score) of log-likelihood
- **H(θ)**: Hessian (observed information matrix)
- **α**: Step size from line search
- **H⁻¹∇ℓ**: Newton direction

### Gradient Computation

The gradient with respect to parameters is computed via the Kalman filter's innovation sequence:

```
∂ℓ/∂θ_j = Σ_{k=1}^T ∂ℓ_k/∂θ_j

where

∂ℓ_k/∂θ_j = -1/2 [(y_k - ŷ_k)^T S^{-1} ∂S/∂θ_j S^{-1} (y_k - ŷ_k) 
             - tr(S^{-1} ∂S/∂θ_j)]
```

Computed efficiently via **finite differences** for practical implementation:

```
∂ℓ/∂θ_j ≈ [ℓ(θ + ε e_j) - ℓ(θ)] / ε
```

### Hessian Computation

The Hessian matrix elements are:

```
∂²ℓ/∂θ_j∂θ_k ≈ [∂ℓ/∂θ_j(θ + ε e_k) - ∂ℓ/∂θ_j(θ)] / ε
```

Computed via **finite differences on the gradient**.

### Line Search

To ensure step acceptance, we use the **Armijo condition**:

```
ℓ(θ + α·d) ≥ ℓ(θ) + c·α·∇ℓ^T·d
```

With:
- `d = H⁻¹∇ℓ`: Newton direction
- `c = 0.1`: Armijo parameter (typical)
- `α`: Starting at 1.0, reduced by factor β = 0.5 until condition satisfied

## Implementation Details

### Core Algorithm (`NewtonRaphsonMLE` Class)

#### Configuration

```swift
public struct Configuration {
    var maxIterations: Int = 50
    var gradientTolerance: Double = 1e-4          // Gradient norm convergence
    var parameterTolerance: Double = 1e-4         // Parameter change convergence
    var minParameterValue: Double = 1e-8          // Lower bound
    var maxParameterValue: Double = 100.0         // Upper bound
    
    // Line search parameters
    var lineSearchMaxSteps: Int = 10
    var lineSearchAlpha: Double = 0.1              // Armijo parameter
    var lineSearchBeta: Double = 0.5               // Step reduction factor
    
    // Finite differences
    var finiteDifferenceEps: Double = 1e-6        // FD step size
    
    // Numerical stability
    var hessianRegularization: Double = 1e-8      // Damping for ill-conditioning
    
    var verbose: Bool = true
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
- `observations`: Time series of observations (T × m)
- `observationModel`: Observation operator H
- `initialParameters`: Initial parameter guess θ⁽⁰⁾
- `initialState`: Initial state estimate x̂₀
- `initialCovariance`: Initial uncertainty P₀
- `dt`: Time step

**Output (`Result`):**
- `parameters`: Final estimated parameters θ*
- `logLikelihoodHistory`: Log-likelihood at each iteration
- `parameterHistory`: Parameter trajectory
- `gradientNorms`: Gradient norm at each iteration
- `iterations`: Number of iterations
- `converged`: Convergence flag
- `finalGradientNorm`: Final gradient norm (optimality measure)
- `finalHessian`: Estimated Hessian (for uncertainty quantification)

### Likelihood Evaluation

#### Log-Likelihood Computation

Using the Kalman filter:

```swift
private func computeLogLikelihood(
    model, observations, observationModel, parameters, 
    initialState, initialCovariance, dt
) -> Double
```

For each time step k = 1, ..., T:

```
Prediction:
  x̂_{k|k-1} = f(x̂_{k-1|k-1})
  P_{k|k-1} = F P_{k-1|k-1} F^T + Q

Update:
  innovation: y_k - H x̂_{k|k-1}
  S = H P_{k|k-1} H^T + R
  K = P_{k|k-1} H^T S^{-1}
  x̂_{k|k} = x̂_{k|k-1} + K innovation
  P_{k|k} = (I - K H) P_{k|k-1}

Log-likelihood contribution:
  ℓ_k = -1/2 [m log(2π) + log det(S) + innovation^T S^{-1} innovation]

Total:
  ℓ = Σ_k ℓ_k
```

#### Gradient Computation

Via **forward finite differences** of log-likelihood:

```
∇ℓ ≈ [ℓ(θ + ε e₁) - ℓ(θ)]/ε, ..., [ℓ(θ + ε eₚ) - ℓ(θ)]/ε
```

Requires **p+1 likelihood evaluations** (p = number of parameters).

#### Hessian Computation

Via **forward finite differences** of gradient:

```
H_{jk} ≈ [∂ℓ/∂θ_j(θ + ε e_k) - ∂ℓ/∂θ_j(θ)]/ε
```

Requires **p(p+1) likelihood evaluations** total.

**Regularization** for numerical stability:

```
H_reg = H + λI,  where λ = 1e-8
```

### Newton Step Computation

Solve the linear system:

```
H · Δθ = ∇ℓ(θ)
```

Using **Gaussian elimination with partial pivoting**:

```swift
private func solveLinearSystem(A: Matrix, b: [Double]) -> [Double]
```

1. Forward elimination with row pivoting
2. Back substitution
3. Regularization for near-singular matrices

### Line Search

```swift
private func lineSearch(
    model, observations, observationModel, currentParameters, 
    direction, currentLogLikelihood, gradient, 
    initialState, initialCovariance, dt
) -> (stepSize: Double, newParameters: [Double])
```

Algorithm:
1. Start with α = 1.0 (full Newton step)
2. Check Armijo condition: ℓ(θ + α·d) ≥ ℓ(θ) + c·α·∇ℓ^T·d
3. If satisfied, return (α, θ + α·d)
4. Otherwise reduce α ← β·α and repeat (max 10 times)
5. Return best step found

## Convergence Properties

### Theoretical Analysis

For optimization problems with twice-differentiable objectives:

1. **Local Convergence**: If α = 1 is ultimately accepted, convergence is **Q-quadratic**
2. **Global Convergence**: Line search ensures global convergence under mild conditions
3. **Robustness**: Works for ill-conditioned problems with regularization

### Practical Convergence Criteria

Algorithm terminates when:

1. **Gradient criterion**: ||∇ℓ(θ)|| < ε_g (typical: 1e-4)
2. **Parameter criterion**: ||θ⁽ⁱ⁺¹⁾ - θ⁽ⁱ⁾|| / ||θ⁽ⁱ⁾|| < ε_p (typical: 1e-4)
3. **Iteration limit**: i > i_max (typical: 50)

### Comparison with EM Algorithm

| Metric | Newton-Raphson | EM |
|--------|----------------|----|
| **Convergence rate** | Q-quadratic | Q-linear |
| **Per-iteration cost** | Very high (Hessian) | Moderate (E+M steps) |
| **Iterations to converge** | 5-20 (few) | 10-50 (more) |
| **Robustness** | Sensitive to init | Very robust |
| **Second-order info** | Explicit (Hessian) | Implicit (E-step) |
| **Uncertainty estimates** | Direct (from H⁻¹) | Requires additional step |

**Recommendation**: 
- **EM**: For robust convergence with poor initialization
- **NR-MLE**: For fast convergence with good initialization or final refinement after EM

## Usage Examples

### Basic Parameter Estimation

```swift
import KalmanCore

// 1. Create model
let model = Lorenz96Model.standard(stochasticType: .additive)

// 2. Generate synthetic data
let trueParameters = [0.5]
let initialState = model.typicalInitialState()
let trueTrajectory = model.simulateTrajectory(
    initialState: initialState,
    parameters: trueParameters,
    dt: 0.01,
    steps: 100
)

// 3. Create observations
let obsModel = PartialObservationModel(
    stateDimension: 40,
    observedIndices: stride(from: 0, to: 40, by: 2).map { $0 },
    noiseVariance: 0.5
)

var observations: [[Double]] = []
for state in trueTrajectory {
    observations.append(obsModel.generateObservation(state: state))
}

// 4. Configure Newton-Raphson
var config = NewtonRaphsonMLE.Configuration()
config.maxIterations = 20
config.verbose = true

// 5. Run optimization
let nr = NewtonRaphsonMLE(config: config)
let result = nr.estimate(
    model: model,
    observations: observations,
    observationModel: obsModel,
    initialParameters: [0.2],
    initialState: initialState,
    initialCovariance: Matrix.identity(size: 40) * 0.1,
    dt: 0.01
)

// 6. Examine results
print("Estimated parameters: \(result.parameters)")
print("Converged: \(result.converged)")
print("Final gradient norm: \(result.finalGradientNorm)")
```

### Two-Stage Estimation (EM then NR-MLE)

```swift
// First: Run EM for robust convergence to neighborhood of optimum
let em = ExpectationMaximization()
let emResult = em.estimate(
    model: model,
    observations: observations,
    observationModel: obsModel,
    initialParameters: [0.2],
    initialState: initialState,
    initialCovariance: Matrix.identity(size: 40) * 0.1,
    dt: 0.01
)

// Second: Use EM result as initialization for Newton-Raphson refinement
let nr = NewtonRaphsonMLE()
let nrResult = nr.estimate(
    model: model,
    observations: observations,
    observationModel: obsModel,
    initialParameters: emResult.parameters,  // Start from EM solution
    initialState: initialState,
    initialCovariance: Matrix.identity(size: 40) * 0.1,
    dt: 0.01
)

print("EM iterations: \(emResult.iterations)")
print("NR iterations: \(nrResult.iterations)")
print("Final parameters: \(nrResult.parameters)")
```

### Uncertainty Quantification

```swift
let result = nr.estimate(...)

// Use final Hessian for uncertainty estimates
if let finalHessian = result.finalHessian {
    // Fisher information matrix (negative of Hessian)
    let fisherInfo = finalHessian * (-1.0)
    
    // Standard errors: sqrt(diagonal of inverse)
    let fisherInv = matrixInverse(fisherInfo)
    let stdErrors = (0..<result.parameters.count).map { i in
        sqrt(abs(fisherInv[i, i]))
    }
    
    // Approximate 95% confidence intervals
    for (i, param) in result.parameters.enumerated() {
        let lower = param - 1.96 * stdErrors[i]
        let upper = param + 1.96 * stdErrors[i]
        print("θ[\(i)]: \(param) ± \(stdErrors[i]) [\(lower), \(upper)]")
    }
}
```

## Numerical Stability

### Finite Difference Step Size

The step size ε for gradient and Hessian affects accuracy:

- **Too small** (< 1e-8): Numerical roundoff errors dominate
- **Too large** (> 1e-4): Truncation errors dominate  
- **Optimal** (~1e-6): Balance of both error sources

Recommendation: Use `eps ~ √(machine_epsilon) ≈ 1e-8` for 64-bit floats.

### Hessian Regularization

For ill-conditioned problems:

```
H_regularized = H + λI,  where λ > 0
```

This ensures positive-definiteness and improves solver stability. Typical λ = 1e-8.

### Parameter Constraints

Box constraints prevent unrealistic parameters:

```
θ ∈ [θ_min, θ_max]
```

After each Newton step, apply:

```
θⱼ ← max(θ_min, min(θ_max, θⱼ))
```

## Computational Complexity

### Per-Iteration Cost

1. **Log-likelihood evaluation**: O(T·n²·m)
2. **Gradient computation** (p+1 LL evals): O((p+1)·T·n²·m)
3. **Hessian computation** (p² gradient evals): O(p²·(p+1)·T·n²·m)
4. **Linear system solve**: O(p³)
5. **Line search** (typically 1-3 steps): O(1-3 iterations)

**Dominant term**: O(p²·T·n²·m) due to Hessian

### Comparison with EM

| Method | Per-iteration | Iterations | Total | Practical |
|--------|---------------|-----------|-------|-----------|
| EM | O(T·n³) | 20-50 | O(T·n³·20-50) | ~1-5 sec |
| NR-MLE | O(p²·T·n²·m) | 5-20 | O(p²·T·n²·m·5-20) | ~5-30 sec |

For typical case: n=40, m=20, p=1, T=1000:
- EM: ~1 sec per iteration → 20-50 sec total
- NR-MLE: ~1 sec per iteration → 5-20 sec total

## File Organization

```
KalmanCore/Sources/KalmanCore/
├── estimation/
│   ├── ExpectationMaximization.swift    # EM algorithm (Section 2.2)
│   └── NewtonRaphsonMLE.swift           # NR-MLE algorithm (Section 2.3)
├── core/
│   └── Matrix.swift                     # Matrix operations
└── filters/
    └── KalmanFilter.swift               # Basic Kalman filter
```

## Testing and Validation

### Build

```bash
cd /path/to/KalmanCore
swift build
```

### Unit Tests

```bash
# Run all tests
swift test

# Run specific test
swift test --filter NewtonRaphsonTest
```

### Example Demonstrations

```bash
# Run demonstrations
swift run
```

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| Non-convergence | Poor initialization | Use EM first, then refine with NR |
| Slow convergence | Line search failing | Reduce FD step size or adjust Armijo α |
| Negative definiteness | Ill-conditioned Hessian | Increase regularization λ |
| NaN/Inf values | Singular covariance | Check observation model, increase noise |

## References

- **Primary**: Pulido et al. (2018) - Main reference
- **Optimization**: Nocedal, J. & Wright, S.J. (2006). Numerical Optimization. Springer
- **Kalman Filter**: Gelb, A. (1974). Applied Optimal Estimation
- **Fisher Information**: Lehmann, E.L. & Casella, G. (1998). Theory of Point Estimation

## Next Steps

1. **Combine with EnKF**: Implement sequential parameter estimation (Section 3)
2. **Extended Scenarios**: Apply to higher-dimensional models
3. **Computational Optimization**: Parallelize Hessian computation
4. **Ensemble Methods**: Combine multiple NR-MLE estimates for robustness

## Contributing

Areas for improvement:
- Quasi-Newton methods (BFGS) to reduce Hessian computation cost
- Adaptive line search strategies
- Parallel Hessian computation
- GPU acceleration for large-scale problems
- Second-derivative-free optimization methods

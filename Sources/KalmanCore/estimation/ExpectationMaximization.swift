import Foundation

/// Expectation-Maximization (EM) Algorithm for parameter identification (Section 2.2)
///
/// Identifies stochastic parameters θ in dynamical systems:
/// dx/dt = M(x, θ) + σ(x, θ)ξ(t)
///
/// The EM algorithm iterates between:
/// - E-step: Compute expected sufficient statistics given current parameters
/// - M-step: Update parameters to maximize expected log-likelihood
///
/// Reference: Pulido et al. (2018), Section 2.2
public class ExpectationMaximization {

  /// Configuration for EM algorithm
  public struct Configuration {
    /// Maximum number of EM iterations
    public var maxIterations: Int = 100

    /// Convergence tolerance (relative change in parameters)
    public var convergenceTolerance: Double = 1e-4

    /// Convergence tolerance for log-likelihood
    public var logLikelihoodTolerance: Double = 1e-6

    /// Minimum parameter value (for numerical stability)
    public var minParameterValue: Double = 1e-8

    /// Maximum parameter value (for numerical stability)
    public var maxParameterValue: Double = 100.0

    /// Verbose output
    public var verbose: Bool = true

    /// Number of smoothing passes (RTS smoother iterations)
    public var smoothingPasses: Int = 1

    public init() {}
  }

  /// Result of EM parameter estimation
  public struct Result {
    /// Estimated parameters
    public let parameters: [Double]

    /// Log-likelihood values at each iteration
    public let logLikelihoodHistory: [Double]

    /// Parameter values at each iteration
    public let parameterHistory: [[Double]]

    /// Number of iterations performed
    public let iterations: Int

    /// Whether the algorithm converged
    public let converged: Bool

    /// Final relative parameter change
    public let finalParameterChange: Double

    /// Final log-likelihood change
    public let finalLogLikelihoodChange: Double
  }

  public let config: Configuration

  public init(config: Configuration = Configuration()) {
    self.config = config
  }

  /// Estimate parameters using EM algorithm
  /// - Parameters:
  ///   - model: Stochastic dynamical system
  ///   - observations: Sequence of observations y_1, ..., y_T
  ///   - observationModel: Observation operator
  ///   - initialParameters: Initial parameter guess
  ///   - initialState: Initial state estimate
  ///   - initialCovariance: Initial state covariance
  ///   - dt: Time step between observations
  /// - Returns: EM estimation result
  public func estimate<Model: StochasticDynamicalSystem>(
    model: Model,
    observations: [[Double]],
    observationModel: ObservationModel,
    initialParameters: [Double],
    initialState: [Double],
    initialCovariance: Matrix,
    dt: Double
  ) -> Result {

    precondition(observations.count > 0, "Need at least one observation")
    precondition(initialParameters.count == model.parameterDimension, "Parameter dimension mismatch")
    precondition(initialState.count == model.stateDimension, "State dimension mismatch")

    var currentParameters = initialParameters
    var logLikelihoodHistory: [Double] = []
    var parameterHistory: [[Double]] = [currentParameters]

    if config.verbose {
      print("\n=== EM Algorithm for Parameter Estimation ===")
      print("Observations: \(observations.count)")
      print("State dimension: \(model.stateDimension)")
      print("Parameter dimension: \(model.parameterDimension)")
      print("Initial parameters: \(currentParameters.map { String(format: "%.4f", $0) })")
      print("\nStarting iterations...\n")
    }

    for iteration in 0..<config.maxIterations {
      // E-step: Run Kalman smoother and compute sufficient statistics
      let smoothedStats = eStep(
        model: model,
        observations: observations,
        observationModel: observationModel,
        parameters: currentParameters,
        initialState: initialState,
        initialCovariance: initialCovariance,
        dt: dt
      )

      // Compute log-likelihood
      let logLikelihood = smoothedStats.logLikelihood
      logLikelihoodHistory.append(logLikelihood)

      // M-step: Update parameters
      let newParameters = mStep(
        model: model,
        smoothedStats: smoothedStats,
        observations: observations,
        dt: dt
      )

      // Enforce parameter constraints
      let constrainedParameters = newParameters.map {
        max(config.minParameterValue, min(config.maxParameterValue, $0))
      }

      // Check convergence
      let parameterChange = relativeChange(from: currentParameters, to: constrainedParameters)
      let logLikelihoodChange = iteration > 0 ?
      abs(logLikelihood - logLikelihoodHistory[iteration - 1]) : Double.infinity

      if config.verbose {
        print("Iteration \(iteration + 1):")
        print("  Log-likelihood: \(String(format: "%.4f", logLikelihood))")
        print("  Parameters: \(constrainedParameters.map { String(format: "%.4f", $0) })")
        print("  Parameter change: \(String(format: "%.6f", parameterChange))")
        print("  LL change: \(String(format: "%.6f", logLikelihoodChange))")
      }

      currentParameters = constrainedParameters
      parameterHistory.append(currentParameters)

      // Check convergence criteria
      if iteration > 0 {
        let converged = parameterChange < config.convergenceTolerance ||
        logLikelihoodChange < config.logLikelihoodTolerance

        if converged {
          if config.verbose {
            print("\n✓ Converged after \(iteration + 1) iterations")
          }
          return Result(
            parameters: currentParameters,
            logLikelihoodHistory: logLikelihoodHistory,
            parameterHistory: parameterHistory,
            iterations: iteration + 1,
            converged: true,
            finalParameterChange: parameterChange,
            finalLogLikelihoodChange: logLikelihoodChange
          )
        }
      }
    }

    if config.verbose {
      print("\n⚠ Maximum iterations reached without convergence")
    }

    let finalChange = parameterHistory.count > 1 ?
    relativeChange(from: parameterHistory[parameterHistory.count - 2], to: currentParameters) : 0.0
    let finalLLChange = logLikelihoodHistory.count > 1 ?
    abs(logLikelihoodHistory.last! - logLikelihoodHistory[logLikelihoodHistory.count - 2]) : 0.0

    return Result(
      parameters: currentParameters,
      logLikelihoodHistory: logLikelihoodHistory,
      parameterHistory: parameterHistory,
      iterations: config.maxIterations,
      converged: false,
      finalParameterChange: finalChange,
      finalLogLikelihoodChange: finalLLChange
    )
  }

  /// E-step: Compute expected sufficient statistics using Kalman smoother
  private func eStep<Model: StochasticDynamicalSystem>(
    model: Model,
    observations: [[Double]],
    observationModel: ObservationModel,
    parameters: [Double],
    initialState: [Double],
    initialCovariance: Matrix,
    dt: Double
  ) -> SmoothedStatistics {

    let T = observations.count
    let n = model.stateDimension

    // Forward pass: Kalman filter
    var filteredStates = [[Double]]()
    var filteredCovariances = [Matrix]()
    var predictedStates = [[Double]]()
    var predictedCovariances = [Matrix]()

    var currentState = initialState
    var currentCov = initialCovariance
    var logLikelihood = 0.0

    for t in 0..<T {
      // Prediction step
      let predicted = model.transition(state: currentState, parameters: parameters, dt: dt)
      let Q = computeProcessNoiseCovariance(model: model, state: currentState, parameters: parameters, dt: dt)
      let F = approximateJacobian(model: model, state: currentState, parameters: parameters, dt: dt)
      let predictedCov = F * currentCov * F.transposed + Q

      predictedStates.append(predicted)
      predictedCovariances.append(predictedCov)

      // Update step
      let observation = observations[t]
      let H = getObservationMatrix(observationModel: observationModel, state: predicted)
      let R = observationModel.observationNoiseCovariance

      let innovation = vectorSubtract(
        observation,
        observationModel.observationOperator(state: predicted)
      )
      let S = H * predictedCov * H.transposed + R
      let K = predictedCov * H.transposed * matrixInverse(S)

      let updated = vectorAdd(predicted, K.multiply(vector: innovation))
      let updatedCov = (Matrix.identity(size: n) - K * H) * predictedCov

      // Update log-likelihood
      logLikelihood += computeLogLikelihoodContribution(innovation: innovation, S: S)

      filteredStates.append(updated)
      filteredCovariances.append(updatedCov)

      currentState = updated
      currentCov = updatedCov
    }

    // Backward pass: RTS smoother
    var smoothedStates = filteredStates
    var smoothedCovariances = filteredCovariances
    var lagOneCovariances = [Matrix]()

    for t in stride(from: T - 2, through: 0, by: -1) {
      let F = approximateJacobian(
        model: model,
        state: filteredStates[t],
        parameters: parameters,
        dt: dt
      )

      let C = filteredCovariances[t] * F.transposed * matrixInverse(predictedCovariances[t + 1])

      let stateDiff = vectorSubtract(smoothedStates[t + 1], predictedStates[t + 1])
      smoothedStates[t] = vectorAdd(filteredStates[t], C.multiply(vector: stateDiff))

      let covDiff = smoothedCovariances[t + 1] - predictedCovariances[t + 1]
      smoothedCovariances[t] = filteredCovariances[t] + C * covDiff * C.transposed

      // Compute lag-one covariance
      lagOneCovariances.insert(smoothedCovariances[t + 1] * C.transposed, at: 0)
    }

    return SmoothedStatistics(
      states: smoothedStates,
      covariances: smoothedCovariances,
      lagOneCovariances: lagOneCovariances,
      logLikelihood: logLikelihood
    )
  }

  /// M-step: Update parameters to maximize expected log-likelihood
  private func mStep<Model: StochasticDynamicalSystem>(
    model: Model,
    smoothedStats: SmoothedStatistics,
    observations: [[Double]],
    dt: Double
  ) -> [Double] {

    // For additive noise parameterization: σ = θ₀·I
    // The M-step has a closed-form solution

    let T = smoothedStats.states.count
    let n = model.stateDimension
    var sumSquaredResiduals = 0.0

    for t in 0..<(T - 1) {
      let currentState = smoothedStats.states[t]
      let nextState = smoothedStats.states[t + 1]

      // Compute deterministic prediction
      let drift = model.deterministicDynamics(state: currentState, parameters: [])
      let predicted = vectorAdd(currentState, drift.map { $0 * dt })

      // Residual: difference between smoothed next state and deterministic prediction
      let residual = vectorSubtract(nextState, predicted)

      // Sum squared residuals
      for i in 0..<n {
        sumSquaredResiduals += residual[i] * residual[i]
      }

      // Add trace of smoothed covariance
      sumSquaredResiduals += smoothedStats.covariances[t + 1].trace
    }

    // MLE estimate for additive noise: θ₀ = sqrt(sum / (n * T * dt))
    let variance = sumSquaredResiduals / (Double(n) * Double(T - 1) * dt)
    let sigma = sqrt(max(config.minParameterValue, variance))

    return [sigma]
  }

  // MARK: - Helper Methods

  private func relativeChange(from old: [Double], to new: [Double]) -> Double {
    var maxChange = 0.0
    for i in 0..<old.count {
      let change = abs(new[i] - old[i]) / max(abs(old[i]), 1e-10)
      maxChange = max(maxChange, change)
    }
    return maxChange
  }

  private func computeProcessNoiseCovariance<Model: StochasticDynamicalSystem>(
    model: Model,
    state: [Double],
    parameters: [Double],
    dt: Double
  ) -> Matrix {
    let sigma = model.stochasticParameterization(state: state, parameters: parameters)
    return dt * (sigma * sigma.transposed)
  }

  private func approximateJacobian<Model: StochasticDynamicalSystem>(
    model: Model,
    state: [Double],
    parameters: [Double],
    dt: Double
  ) -> Matrix {
    // Approximate Jacobian using finite differences
    let n = model.stateDimension
    let eps = 1e-7
    var jacobian = Matrix.identity(size: n)

    let f0 = model.deterministicDynamics(state: state, parameters: parameters)

    for j in 0..<n {
      var perturbedState = state
      perturbedState[j] += eps
      let f1 = model.deterministicDynamics(state: perturbedState, parameters: parameters)

      for i in 0..<n {
        jacobian[i, j] += (f1[i] - f0[i]) / eps * dt
      }
    }

    return jacobian
  }

  private func getObservationMatrix(
    observationModel: ObservationModel,
    state: [Double]
  ) -> Matrix {
    // For linear observation models, extract H matrix
    if let linearModel = observationModel as? LinearObservationModel {
      return linearModel.H
    }

    // For nonlinear models, approximate with finite differences
    let n = state.count
    let m = observationModel.observationDimension
    let eps = 1e-7
    var H = Matrix(rows: m, cols: n)

    let y0 = observationModel.observationOperator(state: state)

    for j in 0..<n {
      var perturbedState = state
      perturbedState[j] += eps
      let y1 = observationModel.observationOperator(state: perturbedState)

      for i in 0..<m {
        H[i, j] = (y1[i] - y0[i]) / eps
      }
    }

    return H
  }

  private func computeLogLikelihoodContribution(innovation: [Double], S: Matrix) -> Double {
    let n = innovation.count
    let detS = matrixDeterminant(S)

    guard detS > 0 else { return -Double.infinity }

    let Sinv = matrixInverse(S)
    let innovVec = innovation
    let mahalanobis = dotProduct(innovVec, Sinv.multiply(vector: innovVec))

    return -0.5 * (Double(n) * log(2 * .pi) + log(detS) + mahalanobis)
  }
}

// MARK: - Supporting Structures

/// Statistics from Kalman smoother
struct SmoothedStatistics {
  let states: [[Double]]
  let covariances: [Matrix]
  let lagOneCovariances: [Matrix]
  let logLikelihood: Double
}

// MARK: - Vector Operations

func vectorAdd(_ a: [Double], _ b: [Double]) -> [Double] {
  precondition(a.count == b.count)
  return zip(a, b).map { $0 + $1 }
}

func vectorSubtract(_ a: [Double], _ b: [Double]) -> [Double] {
  precondition(a.count == b.count)
  return zip(a, b).map { $0 - $1 }
}

func dotProduct(_ a: [Double], _ b: [Double]) -> Double {
  precondition(a.count == b.count)
  return zip(a, b).map { $0 * $1 }.reduce(0, +)
}

// MARK: - Matrix Utilities

func matrixInverse(_ matrix: Matrix) -> Matrix {
  // Simplified inverse for small matrices
  // In production, use LAPACK routines from Accelerate
  precondition(matrix.rows == matrix.cols, "Matrix must be square")

  let n = matrix.rows

  if n == 1 {
    return Matrix(rows: 1, cols: 1, data: [1.0 / matrix[0, 0]])
  }

  // For larger matrices, use Gauss-Jordan elimination (simplified)
  var augmented = Matrix(rows: n, cols: 2 * n)

  // Copy matrix and identity
  for i in 0..<n {
    for j in 0..<n {
      augmented[i, j] = matrix[i, j]
      augmented[i, j + n] = (i == j) ? 1.0 : 0.0
    }
  }

  // Forward elimination
  for i in 0..<n {
    var pivot = augmented[i, i]
    if abs(pivot) < 1e-10 {
      // Add small regularization
      augmented[i, i] += 1e-6
      pivot = augmented[i, i]
    }

    for j in 0..<(2 * n) {
      augmented[i, j] /= pivot
    }

    for k in 0..<n {
      if k != i {
        let factor = augmented[k, i]
        for j in 0..<(2 * n) {
          augmented[k, j] -= factor * augmented[i, j]
        }
      }
    }
  }

  // Extract inverse
  var inverse = Matrix(rows: n, cols: n)
  for i in 0..<n {
    for j in 0..<n {
      inverse[i, j] = augmented[i, j + n]
    }
  }

  return inverse
}

func matrixDeterminant(_ matrix: Matrix) -> Double {
  precondition(matrix.rows == matrix.cols, "Matrix must be square")

  let n = matrix.rows

  if n == 1 {
    return matrix[0, 0]
  }

  if n == 2 {
    return matrix[0, 0] * matrix[1, 1] - matrix[0, 1] * matrix[1, 0]
  }

  // For larger matrices, use LU decomposition (simplified)
  // In production, use LAPACK routines
  var det = 1.0
  var A = matrix

  for i in 0..<n {
    if abs(A[i, i]) < 1e-10 {
      return 1e-10 // Avoid division by zero
    }

    det *= A[i, i]

    for k in (i + 1)..<n {
      let factor = A[k, i] / A[i, i]
      for j in i..<n {
        A[k, j] -= factor * A[i, j]
      }
    }
  }

  return det
}

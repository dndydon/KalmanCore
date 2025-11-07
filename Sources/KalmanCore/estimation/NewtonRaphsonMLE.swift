import Foundation

/// Newton-Raphson Maximum Likelihood Estimation (Section 2.3)
///
/// Identifies stochastic parameters θ in dynamical systems using direct optimization
/// of the likelihood function via Newton's method:
/// dx/dt = M(x, θ) + σ(x, θ)ξ(t)
///
/// The algorithm computes:
/// - Gradient of log-likelihood: ∇_θ ℓ(θ)
/// - Hessian of log-likelihood: ∇²_θ ℓ(θ)
/// - Newton step: Δθ = H⁻¹ ∇_θ ℓ(θ)
///
/// Reference: Pulido et al. (2018), Section 2.3
public class NewtonRaphsonMLE {

  /// Configuration for Newton-Raphson algorithm
  public struct Configuration {
    /// Maximum number of Newton iterations
    public var maxIterations: Int = 50

    /// Convergence tolerance (relative gradient norm)
    public var gradientTolerance: Double = 1e-4

    /// Convergence tolerance (relative parameter change)
    public var parameterTolerance: Double = 1e-4

    /// Minimum parameter value (for numerical stability)
    public var minParameterValue: Double = 1e-8

    /// Maximum parameter value (for numerical stability)
    public var maxParameterValue: Double = 100.0

    /// Line search parameters
    public var lineSearchMaxSteps: Int = 10
    public var lineSearchAlpha: Double = 0.1  // Armijo condition parameter
    public var lineSearchBeta: Double = 0.5   // Step size reduction factor

    /// Finite difference step for gradient/Hessian computation
    public var finiteDifferenceEps: Double = 1e-6

    /// Regularization for Hessian (for ill-conditioning)
    public var hessianRegularization: Double = 1e-8

    /// Verbose output
    public var verbose: Bool = true

    /// Number of smoothing passes (RTS smoother iterations)
    public var smoothingPasses: Int = 1

    public init() {}
  }

  /// Result of Newton-Raphson parameter estimation
  public struct Result {
    /// Estimated parameters
    public let parameters: [Double]

    /// Log-likelihood values at each iteration
    public let logLikelihoodHistory: [Double]

    /// Parameter values at each iteration
    public let parameterHistory: [[Double]]

    /// Gradient norms at each iteration
    public let gradientNorms: [Double]

    /// Number of iterations performed
    public let iterations: Int

    /// Whether the algorithm converged
    public let converged: Bool

    /// Final gradient norm
    public let finalGradientNorm: Double

    /// Final relative parameter change
    public let finalParameterChange: Double

    /// Estimated Hessian at solution (for uncertainty quantification)
    public let finalHessian: Matrix?
  }

  public let config: Configuration

  public init(config: Configuration = Configuration()) {
    self.config = config
  }

  /// Estimate parameters using Newton-Raphson method
  /// - Parameters:
  ///   - model: Stochastic dynamical system
  ///   - observations: Sequence of observations y_1, ..., y_T
  ///   - observationModel: Observation operator
  ///   - initialParameters: Initial parameter guess
  ///   - initialState: Initial state estimate
  ///   - initialCovariance: Initial state covariance
  ///   - dt: Time step between observations
  /// - Returns: Newton-Raphson estimation result
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

    let _ = model.parameterDimension  // For future use with parameter count tracking
    var currentParameters = initialParameters
    var logLikelihoodHistory: [Double] = []
    var parameterHistory: [[Double]] = [currentParameters]
    var gradientNorms: [Double] = []

    if config.verbose {
      print("\n=== Newton-Raphson MLE Parameter Estimation ===")
      print("Observations: \(observations.count)")
      print("State dimension: \(model.stateDimension)")
      print("Parameter dimension: \(model.parameterDimension)")
      let paramStr = currentParameters.map { String(format: "%.4f", $0) }.joined(separator: ", ")
      print("Initial parameters: [\(paramStr)]")
      print("\nStarting iterations...\n")
    }

    var finalHessian: Matrix? = nil

    for iteration in 0..<config.maxIterations {
      // Compute log-likelihood and its derivatives
      let evaluation = evaluateLikelihood(
        model: model,
        observations: observations,
        observationModel: observationModel,
        parameters: currentParameters,
        initialState: initialState,
        initialCovariance: initialCovariance,
        dt: dt
      )

      let logLikelihood = evaluation.logLikelihood
      let gradient = evaluation.gradient
      let hessian = evaluation.hessian

      logLikelihoodHistory.append(logLikelihood)

      // Compute gradient norm
      let gradientNorm = sqrt(dotProduct(gradient, gradient))
      gradientNorms.append(gradientNorm)

      if config.verbose {
        print("Iteration \(iteration + 1):")
        print("  Log-likelihood: \(String(format: "%.4f", logLikelihood))")
        print("  Gradient norm: \(String(format: "%.6e", gradientNorm))")
        let paramStr = currentParameters.map { String(format: "%.4f", $0) }.joined(separator: ", ")
        print("  Parameters: [\(paramStr)]")
      }

      // Check gradient convergence
      if gradientNorm < config.gradientTolerance {
        if config.verbose {
          print("\n✓ Converged: gradient norm below tolerance")
        }
        finalHessian = hessian
        return Result(
          parameters: currentParameters,
          logLikelihoodHistory: logLikelihoodHistory,
          parameterHistory: parameterHistory,
          gradientNorms: gradientNorms,
          iterations: iteration + 1,
          converged: true,
          finalGradientNorm: gradientNorm,
          finalParameterChange: 0.0,
          finalHessian: finalHessian
        )
      }

      // Compute Newton step
      let step = computeNewtonStep(gradient: gradient, hessian: hessian)

      // Line search for step size
      let (stepSize, newParameters) = lineSearch(
        model: model,
        observations: observations,
        observationModel: observationModel,
        currentParameters: currentParameters,
        direction: step,
        currentLogLikelihood: logLikelihood,
        gradient: gradient,
        initialState: initialState,
        initialCovariance: initialCovariance,
        dt: dt
      )

      // Apply constraints
      let constrainedParameters = newParameters.map {
        max(config.minParameterValue, min(config.maxParameterValue, $0))
      }

      // Check parameter convergence
      let parameterChange = relativeChange(from: currentParameters, to: constrainedParameters)

      if config.verbose {
        print("  Step size: \(String(format: "%.6f", stepSize))")
        print("  Parameter change: \(String(format: "%.6e", parameterChange))")
      }

      currentParameters = constrainedParameters
      parameterHistory.append(currentParameters)

      // Check parameter convergence criterion
      if iteration > 0 && parameterChange < config.parameterTolerance {
        if config.verbose {
          print("\n✓ Converged: parameter change below tolerance")
        }
        finalHessian = hessian
        return Result(
          parameters: currentParameters,
          logLikelihoodHistory: logLikelihoodHistory,
          parameterHistory: parameterHistory,
          gradientNorms: gradientNorms,
          iterations: iteration + 1,
          converged: true,
          finalGradientNorm: gradientNorm,
          finalParameterChange: parameterChange,
          finalHessian: finalHessian
        )
      }
    }

    if config.verbose {
      print("\n⚠ Maximum iterations reached without convergence")
    }

    let finalChange = parameterHistory.count > 1 ?
    relativeChange(from: parameterHistory[parameterHistory.count - 2], to: currentParameters) : 0.0

    return Result(
      parameters: currentParameters,
      logLikelihoodHistory: logLikelihoodHistory,
      parameterHistory: parameterHistory,
      gradientNorms: gradientNorms,
      iterations: config.maxIterations,
      converged: false,
      finalGradientNorm: gradientNorms.last ?? Double.infinity,
      finalParameterChange: finalChange,
      finalHessian: finalHessian
    )
  }

  // MARK: - Likelihood Evaluation

  /// Likelihood evaluation result
  struct LikelihoodEvaluation {
    let logLikelihood: Double
    let gradient: [Double]
    let hessian: Matrix
  }

  /// Evaluate log-likelihood and compute gradient and Hessian
  private func evaluateLikelihood<Model: StochasticDynamicalSystem>(
    model: Model,
    observations: [[Double]],
    observationModel: ObservationModel,
    parameters: [Double],
    initialState: [Double],
    initialCovariance: Matrix,
    dt: Double
  ) -> LikelihoodEvaluation {

    let p = model.parameterDimension

    // Compute log-likelihood at current parameters
    let llAtParams = computeLogLikelihood(
      model: model,
      observations: observations,
      observationModel: observationModel,
      parameters: parameters,
      initialState: initialState,
      initialCovariance: initialCovariance,
      dt: dt
    )

    // Compute gradient via finite differences
    var gradient = [Double](repeating: 0.0, count: p)
    for j in 0..<p {
      var paramsPlus = parameters
      paramsPlus[j] += config.finiteDifferenceEps

      let llPlus = computeLogLikelihood(
        model: model,
        observations: observations,
        observationModel: observationModel,
        parameters: paramsPlus,
        initialState: initialState,
        initialCovariance: initialCovariance,
        dt: dt
      )

      gradient[j] = (llPlus - llAtParams) / config.finiteDifferenceEps
    }

    // Compute Hessian via finite differences
    var hessian = Matrix(rows: p, cols: p)
    for j in 0..<p {
      var paramsPlus = parameters
      paramsPlus[j] += config.finiteDifferenceEps

      let llPlus = computeLogLikelihood(
        model: model,
        observations: observations,
        observationModel: observationModel,
        parameters: paramsPlus,
        initialState: initialState,
        initialCovariance: initialCovariance,
        dt: dt
      )

      var gradPlus = [Double](repeating: 0.0, count: p)
      for k in 0..<p {
        var paramsDoublePlus = paramsPlus
        paramsDoublePlus[k] += config.finiteDifferenceEps

        let llDoublePlus = computeLogLikelihood(
          model: model,
          observations: observations,
          observationModel: observationModel,
          parameters: paramsDoublePlus,
          initialState: initialState,
          initialCovariance: initialCovariance,
          dt: dt
        )

        gradPlus[k] = (llDoublePlus - llPlus) / config.finiteDifferenceEps
      }

      for k in 0..<p {
        hessian[j, k] = (gradPlus[k] - gradient[k]) / config.finiteDifferenceEps
      }
    }

    // Regularize Hessian for numerical stability
    for i in 0..<p {
      hessian[i, i] += config.hessianRegularization
    }

    return LikelihoodEvaluation(
      logLikelihood: llAtParams,
      gradient: gradient,
      hessian: hessian
    )
  }

  /// Compute log-likelihood via Kalman filter
  private func computeLogLikelihood<Model: StochasticDynamicalSystem>(
    model: Model,
    observations: [[Double]],
    observationModel: ObservationModel,
    parameters: [Double],
    initialState: [Double],
    initialCovariance: Matrix,
    dt: Double
  ) -> Double {

    let T = observations.count
    let n = model.stateDimension
    var currentState = initialState
    var currentCov = initialCovariance
    var logLikelihood = 0.0

    for t in 0..<T {
      // Prediction step
      let predicted = model.transition(state: currentState, parameters: parameters, dt: dt)
      let Q = computeProcessNoiseCovariance(model: model, state: currentState, parameters: parameters, dt: dt)
      let F = approximateJacobian(model: model, state: currentState, parameters: parameters, dt: dt)
      let predictedCov = F * currentCov * F.transposed + Q

      // Update step
      let observation = observations[t]
      let H = getObservationMatrix(observationModel: observationModel, state: predicted)
      let R = observationModel.observationNoiseCovariance

      let innovation = vectorSubtract(
        observation,
        observationModel.observationOperator(state: predicted)
      )
      let S = H * predictedCov * H.transposed + R

      // Update log-likelihood
      logLikelihood += computeLogLikelihoodContribution(innovation: innovation, S: S)

      // Kalman gain
      let K = predictedCov * H.transposed * matrixInverse(S)

      // Update state and covariance
      currentState = vectorAdd(predicted, K.multiply(vector: innovation))
      currentCov = (Matrix.identity(size: n) - K * H) * predictedCov
    }

    return logLikelihood
  }

  // MARK: - Newton Step Computation

  /// Compute Newton step: Δθ = -H⁻¹ ∇_θ ℓ(θ)
  private func computeNewtonStep(gradient: [Double], hessian: Matrix) -> [Double] {
    // Negative Hessian for maximization (we're maximizing, not minimizing)
    let H = hessian * (-1.0)

    // Solve H * step = gradient
    let step = solveLinearSystem(A: H, b: gradient)

    return step
  }

  // MARK: - Line Search

  /// Line search to find appropriate step size
  private func lineSearch<Model: StochasticDynamicalSystem>(
    model: Model,
    observations: [[Double]],
    observationModel: ObservationModel,
    currentParameters: [Double],
    direction: [Double],
    currentLogLikelihood: Double,
    gradient: [Double],
    initialState: [Double],
    initialCovariance: Matrix,
    dt: Double
  ) -> (stepSize: Double, newParameters: [Double]) {

    var stepSize = 1.0

    for _ in 0..<config.lineSearchMaxSteps {
      // Propose new parameters
      var proposedParameters = currentParameters
      for i in 0..<currentParameters.count {
        proposedParameters[i] += stepSize * direction[i]
      }

      // Evaluate likelihood at proposed parameters
      let newLogLikelihood = computeLogLikelihood(
        model: model,
        observations: observations,
        observationModel: observationModel,
        parameters: proposedParameters,
        initialState: initialState,
        initialCovariance: initialCovariance,
        dt: dt
      )

      // Armijo condition: f(x + α*d) >= f(x) + c*α*∇f^T*d
      let directionalDerivative = dotProduct(gradient, direction)
      let expectedIncrease = config.lineSearchAlpha * stepSize * directionalDerivative

      if newLogLikelihood >= currentLogLikelihood + expectedIncrease {
        return (stepSize: stepSize, newParameters: proposedParameters)
      }

      stepSize *= config.lineSearchBeta
    }

    // Return last attempted step even if line search didn't converge
    var proposedParameters = currentParameters
    for i in 0..<currentParameters.count {
      proposedParameters[i] += stepSize * direction[i]
    }

    return (stepSize: stepSize, newParameters: proposedParameters)
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
    if let linearModel = observationModel as? LinearObservationModel {
      return linearModel.H
    }

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
    let mahalanobis = dotProduct(innovation, Sinv.multiply(vector: innovation))

    return -0.5 * (Double(n) * log(2 * .pi) + log(detS) + mahalanobis)
  }
}

// MARK: - Linear System Solver

/// Solve Ax = b using Gaussian elimination (local implementation)
private func solveLinearSystem(A: Matrix, b: [Double]) -> [Double] {
  let n = A.rows
  precondition(A.cols == n && b.count == n, "Dimension mismatch")

  var augmented = Matrix(rows: n, cols: n + 1)
  for i in 0..<n {
    for j in 0..<n {
      augmented[i, j] = A[i, j]
    }
    augmented[i, n] = b[i]
  }

  for i in 0..<n {
    var maxRow = i
    for k in i+1..<n {
      if abs(augmented[k, i]) > abs(augmented[maxRow, i]) {
        maxRow = k
      }
    }

    if maxRow != i {
      for j in 0...(n) {
        let temp = augmented[i, j]
        augmented[i, j] = augmented[maxRow, j]
        augmented[maxRow, j] = temp
      }
    }

    if abs(augmented[i, i]) < 1e-10 {
      augmented[i, i] = 1e-10
    }

    for k in i+1..<n {
      let factor = augmented[k, i] / augmented[i, i]
      for j in i...n {
        augmented[k, j] -= factor * augmented[i, j]
      }
    }
  }

  var x = [Double](repeating: 0.0, count: n)
  for i in stride(from: n-1, through: 0, by: -1) {
    x[i] = augmented[i, n]
    for j in i+1..<n {
      x[i] -= augmented[i, j] * x[j]
    }
    x[i] /= augmented[i, i]
  }

  return x
}


import Foundation

/// Configuration for the augmented-state Ensemble Kalman Filter (EnKF)
public struct EnKFConfig {
  /// Number of ensemble members
  public var ensembleSize: Int
  /// Multiplicative inflation factor (>= 1.0). Applied in analysis step (TODO)
  public var inflation: Double
  /// Additive inflation (noise covariance) for state (optional, TODO in analysis)
  public var additiveInflation: Matrix?
  /// Parameter evolution model
  public var parameterEvolution: ParameterEvolution
  /// Use perturbed observations (stochastic EnKF) if true; otherwise deterministic variant (future)
  public var usePerturbedObservations: Bool
  /// Optional localization radius (placeholder – not used in scaffold)
  public var localizationRadius: Double?
  /// Verbose logging
  public var verbose: Bool

  public init(
    ensembleSize: Int,
    inflation: Double = 1.0,
    additiveInflation: Matrix? = nil,
    parameterEvolution: ParameterEvolution = .constant,
    usePerturbedObservations: Bool = true,
    localizationRadius: Double? = nil,
    verbose: Bool = false
  ) {
    self.ensembleSize = ensembleSize
    self.inflation = inflation
    self.additiveInflation = additiveInflation
    self.parameterEvolution = parameterEvolution
    self.usePerturbedObservations = usePerturbedObservations
    self.localizationRadius = localizationRadius
    self.verbose = verbose
  }
}

/// Evolution model for (possibly time-varying) parameters θ
public enum ParameterEvolution {
  /// θ_{k+1} = θ_k
  case constant
  /// θ_{k+1} = θ_k + η_k,  η_k ~ N(0, Qθ)
  case randomWalk(Qtheta: Matrix)
  /// θ_{k+1} = ρ θ_k + η_k,  η_k ~ N(0, Qθ)
  case ar1(rho: Double, Qtheta: Matrix)
}

/// Augmented-state EnKF that jointly estimates state x and parameters θ
/// The ensemble members are length (n + p) vectors [x; θ]
public final class EnsembleKalmanFilter<Model: StochasticDynamicalSystem> {
  public let model: Model
  public let observationModel: ObservationModel
  public let config: EnKFConfig

  public let stateDimension: Int
  public let parameterDimension: Int
  public let augmentedDimension: Int

  public init(model: Model, observationModel: ObservationModel, config: EnKFConfig) {
    precondition(observationModel.stateDimension == model.stateDimension,
                "Observation/state dimension mismatch")
    self.model = model
    self.observationModel = observationModel
    self.config = config
    self.stateDimension = model.stateDimension
    self.parameterDimension = model.parameterDimension
    self.augmentedDimension = stateDimension + parameterDimension
  }

  /// Build an initial augmented ensemble Z0 from Gaussian priors
  /// - x0, P0: initial state mean/covariance
  /// - theta0, Ptheta0: initial parameter mean/covariance
  public func initializeEnsemble(
    x0: [Double],
    P0: Matrix,
    theta0: [Double],
    Ptheta0: Matrix,
    seed: UInt64? = nil
  ) -> Ensemble {
    precondition(x0.count == stateDimension, "x0 dimension mismatch")
    precondition(theta0.count == parameterDimension, "theta0 dimension mismatch")
    precondition(P0.rows == stateDimension && P0.cols == stateDimension, "P0 size mismatch")
    precondition(Ptheta0.rows == parameterDimension && Ptheta0.cols == parameterDimension, "Ptheta0 size mismatch")

    var members: [[Double]] = []
    members.reserveCapacity(config.ensembleSize)

    // Note: This ignores state–parameter cross-covariances at init time (block-diagonal prior)
    for _ in 0..<config.ensembleSize {
      let dx = RandomUtils.generateGaussianNoiseWithCovariance(dimension: stateDimension, covariance: P0)
      let dtheta = RandomUtils.generateGaussianNoiseWithCovariance(dimension: parameterDimension, covariance: Ptheta0)
      var z = [Double]()
      z.reserveCapacity(augmentedDimension)
      z.append(contentsOf: zip(x0, dx).map(+))
      z.append(contentsOf: zip(theta0, dtheta).map(+))
      members.append(z)
    }

    if config.verbose {
      print("Initialized augmented ensemble: size=\(config.ensembleSize), dim=\(augmentedDimension)")
    }

    return Ensemble(members: members)
  }

  /// Forecast step: evolve x via model transition, θ via parameter evolution
  public func forecast(ensemble: Ensemble, dt: Double) -> Ensemble {
    precondition(ensemble.stateDimension == augmentedDimension, "Ensemble dimension must be n + p")

    var nextMembers: [[Double]] = []
    nextMembers.reserveCapacity(ensemble.ensembleSize)

    for member in ensemble.members {
      let (x, theta) = split(member)

      // Evolve state with current parameters
      let xNext = model.transition(state: x, parameters: theta, dt: dt)

      // Evolve parameters per chosen model
      let thetaNext: [Double]
      switch config.parameterEvolution {
      case .constant:
        thetaNext = theta
      case .randomWalk(let Qtheta):
        let eta = RandomUtils.generateGaussianNoiseWithCovariance(dimension: parameterDimension, covariance: Qtheta)
        thetaNext = zip(theta, eta).map(+)
      case .ar1(let rho, let Qtheta):
        let eta = RandomUtils.generateGaussianNoiseWithCovariance(dimension: parameterDimension, covariance: Qtheta)
        thetaNext = theta.map { rho * $0 } .enumerated().map { $1 + eta[$0] }
      }

      nextMembers.append(merge(xNext, thetaNext))
    }

    // Note: Inflation and localization will be applied in analyze() in a future implementation
    return Ensemble(members: nextMembers)
  }

  /// Analysis step: update [x; θ] with observation using EnKF (stochastic perturbed-observation or deterministic)
  public func analyze(ensemble: Ensemble, observation: [Double]) -> Ensemble {
    precondition(observation.count == observationModel.observationDimension, "Observation dimension mismatch")
    precondition(ensemble.stateDimension == augmentedDimension, "Ensemble dimension must be n + p")

    let N = ensemble.ensembleSize
    precondition(N > 1, "Ensemble size must be > 1")
    let n = stateDimension
    let p = parameterDimension
    let m = observationModel.observationDimension

    // Optionally apply multiplicative inflation to STATE part only (parameters unchanged)
    var inflatedMembers = ensemble.members
    if config.inflation != 1.0 {
      let alpha = max(1.0, config.inflation)
      // Compute state mean
      var stateMean = [Double](repeating: 0.0, count: n)
      for member in inflatedMembers {
        for i in 0..<n { stateMean[i] += member[i] }
      }
      let invN = 1.0 / Double(N)
      for i in 0..<n { stateMean[i] *= invN }
      // Inflate state deviations
      let sqrtAlpha = sqrt(alpha)
      for k in 0..<N {
        for i in 0..<n {
          let dev = inflatedMembers[k][i] - stateMean[i]
          inflatedMembers[k][i] = stateMean[i] + sqrtAlpha * dev
        }
      }
    }

    // Predicted observations per member
    var yPred = Array(repeating: [Double](repeating: 0.0, count: m), count: N)
    for (idx, member) in inflatedMembers.enumerated() {
      let x = Array(member[0..<n])
      yPred[idx] = observationModel.observationOperator(state: x)
    }

    // Observation perturbations (stochastic EnKF) or none (deterministic)
    let R = observationModel.observationNoiseCovariance
    var yTilde = Array(repeating: [Double](repeating: 0.0, count: m), count: N)
    if config.usePerturbedObservations {
      for i in 0..<N {
        let v = RandomUtils.generateGaussianNoiseWithCovariance(dimension: m, covariance: R)
        for j in 0..<m { yTilde[i][j] = observation[j] + v[j] }
      }
    } else {
      for i in 0..<N { yTilde[i] = observation }
    }

    // Means in observation and augmented spaces
    var yMean = [Double](repeating: 0.0, count: m)
    for i in 0..<N { for j in 0..<m { yMean[j] += yPred[i][j] } }
    let invN = 1.0 / Double(N)
    for j in 0..<m { yMean[j] *= invN }

    // Anomaly matrices: A_y (m×N), A_z ( (n+p)×N )
    var A_y = Matrix(rows: m, cols: N)
    for col in 0..<N {
      for row in 0..<m {
        A_y[row, col] = yPred[col][row] - yMean[row]
      }
    }

    // Build Ensemble from inflatedMembers to reuse existing anomalyMatrix
    let inflatedEnsemble = Ensemble(members: inflatedMembers)
    let A_z = inflatedEnsemble.anomalyMatrix // (n+p)×N

    // Covariances
    let scale = 1.0 / Double(N - 1)
    let S = scale * (A_y * A_y.transposed) + R              // m×m
    let P_zy = scale * (A_z * A_y.transposed)               // (n+p)×m

    // Kalman gain
    let S_inv = matrixInverse(S)
    let K = P_zy * S_inv                                    // (n+p)×m

    // Member-wise update
    var updatedMembers = inflatedMembers
    for i in 0..<N {
      // Innovation per member
      var d = [Double](repeating: 0.0, count: m)
      for j in 0..<m { d[j] = yTilde[i][j] - yPred[i][j] }
      // Delta in augmented space
      let delta = K.multiply(vector: d)
      for k in 0..<(n + p) {
        updatedMembers[i][k] += delta[k]
      }
    }

    return Ensemble(members: updatedMembers)
  }

  // MARK: - Helpers
  @inline(__always)
  private func split(_ z: [Double]) -> (x: [Double], theta: [Double]) {
    let x = Array(z[0..<stateDimension])
    let theta = Array(z[stateDimension..<(stateDimension + parameterDimension)])
    return (x, theta)
  }

  @inline(__always)
  private func merge(_ x: [Double], _ theta: [Double]) -> [Double] {
    precondition(x.count == stateDimension && theta.count == parameterDimension)
    var z = [Double]()
    z.reserveCapacity(augmentedDimension)
    z.append(contentsOf: x)
    z.append(contentsOf: theta)
    return z
  }
}
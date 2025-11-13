import Foundation

/*
 Augmented-state Ensemble Kalman Filter (EnKF)
 --------------------------------------------
 Implements an augmented-state EnKF that jointly updates the system state x and
 parameters θ by operating on concatenated ensemble members [x; θ].

 Features implemented
 - Forecast: state via model.transition, parameters via constant/random-walk/AR(1)
 - Analysis:
   - Stochastic EnKF (perturbed observations) or deterministic square-root variant (ETKF-style)
   - Ensemble-space transform for anomalies in square-root mode
 - Inflation: multiplicative (state-only), additive (state-only)
 - Localization:
   - Optional Schur (Gaspari–Cohn) taper applied to state rows of P_zy (state–obs cross-covariance)
   - Default parameters are not localized (policy configurable in future revisions)

 References
 - Evensen, G. (1994). Ensemble Kalman filter.
 - Pulido, M., Tandeo, P., Bocquet, M., Carrassi, A., & Lucini, M. (2018). Stochastic parameterization identification using EnKF + ML.
 - Farchi, A., & Bocquet, M. (2019). On the Efficiency of Covariance Localisation of the EnKF Using Augmented Ensembles. Front. Appl. Math. Stat., 5. https://doi.org/10.3389/fams.2019.00003
 - Gaspari, G., & Cohn, S.E. (1999). Construction of correlation functions in two and three dimensions.
 */

/// Localization configuration (Stage 1)
public struct LocalizationConfig {
  public enum Method {
    case none
    /// 1D Gaspari–Cohn taper with periodic distance (e.g., Lorenz-96)
    case schurGaspariCohn1D(lengthScale: Double, periodic: Bool)
  }
  /// Localization method
  public var method: Method
  /// Optional observed state indices for building a state–obs taper (if nil, will try to infer
  /// for IdentityObservationModel / PartialObservationModel; otherwise localization on P_zy is skipped).
  public var observedIndices: [Int]?
  public init(method: Method = .none, observedIndices: [Int]? = nil) {
    self.method = method
    self.observedIndices = observedIndices
  }
}

/// Configuration for the augmented-state Ensemble Kalman Filter (EnKF)
public struct EnKFConfig {
  /// Number of ensemble members
  public var ensembleSize: Int
  /// Multiplicative inflation factor (>= 1.0). Applied to STATE anomalies during analysis.
  public var inflation: Double
  /// Additive inflation (noise covariance) added to STATE during analysis (optional).
  public var additiveInflation: Matrix?
  /// Parameter evolution model
  public var parameterEvolution: ParameterEvolution
  /// If true, use a deterministic square-root analysis (ETKF-style) that transforms anomalies
  /// in ensemble space; otherwise use the stochastic EnKF (perturbed observations) or deterministic
  /// update without a square-root transform depending on `usePerturbedObservations`.
  public var useSquareRootAnalysis: Bool
  /// Use perturbed observations (stochastic EnKF) if true; if false and useSquareRootAnalysis is false,
  /// members are updated deterministically without perturbing observations.
  public var usePerturbedObservations: Bool
  /// Localization settings (Schur taper). Default: none.
  public var localization: LocalizationConfig
  /// Verbose logging
  public var verbose: Bool

  public init(
    ensembleSize: Int,
    inflation: Double = 1.0,
    additiveInflation: Matrix? = nil,
    parameterEvolution: ParameterEvolution = .constant,
    useSquareRootAnalysis: Bool = false,
    usePerturbedObservations: Bool = true,
    localization: LocalizationConfig = LocalizationConfig(),
    verbose: Bool = false
  ) {
    self.ensembleSize = ensembleSize
    self.inflation = inflation
    self.additiveInflation = additiveInflation
    self.parameterEvolution = parameterEvolution
    self.useSquareRootAnalysis = useSquareRootAnalysis
    self.usePerturbedObservations = usePerturbedObservations
    self.localization = localization
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

  /// Analysis step: update [x; θ] with observation using EnKF
  /// - If useSquareRootAnalysis is true: deterministic ETKF-style transform in ensemble space.
  /// - Else: stochastic EnKF (usePerturbedObservations) or deterministic without square-root if false.
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

    // Optionally apply additive inflation (STATE only): x_i <- x_i + ξ_i,  ξ_i ~ N(0, Q_add)
    if let Qadd = config.additiveInflation {
      precondition(Qadd.rows == n && Qadd.cols == n, "additiveInflation must be n×n for state")
      for k in 0..<N {
        let noise = RandomUtils.generateGaussianNoiseWithCovariance(dimension: n, covariance: Qadd)
        for i in 0..<n {
          inflatedMembers[k][i] += noise[i]
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
    for i in 0..<N {
      for j in 0..<m {
        yMean[j] += yPred[i][j]
      }
    }
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
    var P_zy = scale * (A_z * A_y.transposed)               // (n+p)×m

    // Apply Schur (Gaspari–Cohn) localization to STATE rows of P_zy if configured
    if case .schurGaspariCohn1D(let L, let periodic) = config.localization.method {
      if let taper = buildStateObsTaperMatrix(n: n, m: m, periodic: periodic, lengthScale: L,
                                              observedIndicesHint: config.localization.observedIndices) {
        for i in 0..<n { // state rows
          for j in 0..<m { P_zy[i, j] *= taper[i, j] }
        }
      } else if config.verbose {
        print("[EnKF] Schur localization skipped (no observedIndices mapping available for this observation model)")
      }
    }

    if config.useSquareRootAnalysis {
      // Deterministic square-root analysis (ETKF-style in ensemble space)
      // Mean update uses K d_mean; anomalies updated by ensemble-space transform T = (I + E)^{-1/2}
      let S_inv = matrixInverse(S)
      let K = P_zy * S_inv                                  // (n+p)×m
      var yMean = [Double](repeating: 0.0, count: m)
      for col in 0..<N { for j in 0..<m { yMean[j] += yPred[col][j] } }
      for j in 0..<m { yMean[j] *= (1.0 / Double(N)) }
      var dMean = [Double](repeating: 0.0, count: m)
      for j in 0..<m { dMean[j] = observation[j] - yMean[j] }
      let deltaMean = K.multiply(vector: dMean)              // (n+p)

      // Compute analysis mean: z̄_a = z̄_f + deltaMean
      // Compute z̄_f (augmented mean)
      var zBarF = [Double](repeating: 0.0, count: n + p)
      for member in inflatedMembers { for k in 0..<(n+p) { zBarF[k] += member[k] } }
      for k in 0..<(n+p) { zBarF[k] /= Double(N) }
      var zBarA = [Double](repeating: 0.0, count: n + p)
      for k in 0..<(n+p) { zBarA[k] = zBarF[k] + deltaMean[k] }

      // Ensemble-space transform for anomalies
      let R_inv = matrixInverse(R)
      // E = (1/(N-1)) A_y^T R^{-1} A_y  (N×N)
      let E = scale * (A_y.transposed * (R_inv * A_y))
      let I_N = Matrix.identity(size: N)
      let S_ens = I_N + E                                   // SPD
      // Use Cholesky-based transform: if S_ens = L L^T (lower), set T = (L^{-1})^T so that T T^T = S_ens^{-1}
      let T: Matrix
      if let L = choleskyLowerSPD(S_ens) {
        let Linv = matrixInverse(L)
        T = Linv.transposed
      } else {
        // Fallback: regular inverse then take (inverse)^{1/2} approximately via Newton–Schulz single step
        let Minv = matrixInverse(S_ens)
        T = Minv // acceptable fallback as a transform factor
      }
      let A_z_a = A_z * T                                   // (n+p)×N

      // Recompose members: z_i^a = z̄_a + A_z_a[:, i]
      var updatedMembers = Array(repeating: [Double](repeating: 0.0, count: n+p), count: N)
      for i in 0..<N {
        for k in 0..<(n+p) { updatedMembers[i][k] = zBarA[k] + A_z_a[k, i] }
      }
      return Ensemble(members: updatedMembers)
    } else {
      // Stochastic (or deterministic without square-root) member-wise update
      let S_inv = matrixInverse(S)
      let K = P_zy * S_inv                                  // (n+p)×m

      var updatedMembers = inflatedMembers
      for i in 0..<N {
        // Innovation per member
        var d = [Double](repeating: 0.0, count: m)
        for j in 0..<m { d[j] = yTilde[i][j] - yPred[i][j] }
        // Delta in augmented space
        let delta = K.multiply(vector: d)
        for k in 0..<(n + p) { updatedMembers[i][k] += delta[k] }
      }
      return Ensemble(members: updatedMembers)
    }
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

  // Build an n×m taper matrix for state–observation pairs on a 1D grid
  // Uses Gaspari–Cohn compactly supported correlation with support 2*lengthScale.
  private func buildStateObsTaperMatrix(n: Int, m: Int, periodic: Bool, lengthScale: Double,
                                        observedIndicesHint: [Int]?) -> Matrix? {
    // Infer observed indices for Identity / PartialObservationModel if not provided
    var observed: [Int]? = observedIndicesHint
    if observed == nil {
      if let id = observationModel as? IdentityObservationModel {
        observed = Array(0..<id.observationDimension)
      } else if let partial = observationModel as? PartialObservationModel {
        // Reconstruct observed indices by inspecting H rows
        var indices: [Int] = []
        indices.reserveCapacity(partial.observationDimension)
        for j in 0..<partial.observationDimension {
          var idx: Int = -1
          for i in 0..<partial.stateDimension { if partial.H[j, i] == 1.0 { idx = i; break } }
          if idx >= 0 { indices.append(idx) } else { return nil }
        }
        observed = indices
      }
    }
    guard let obsIdx = observed, obsIdx.count == m else { return nil }

    var taper = Matrix(rows: n, cols: m)
    let L = max(1e-12, lengthScale)
    for i in 0..<n {
      for j in 0..<m {
        let sIdx = i
        let oIdx = obsIdx[j]
        let d = oneDDistance(a: sIdx, b: oIdx, size: n, periodic: periodic)
        let r = d / L
        taper[i, j] = gaspariCohn(r: r)
      }
    }
    return taper
  }

  @inline(__always)
  private func oneDDistance(a: Int, b: Int, size: Int, periodic: Bool) -> Double {
    let diff = abs(a - b)
    if periodic { return Double(min(diff, size - diff)) }
    return Double(diff)
  }

  // Standard Gaspari–Cohn correlation (support r in [0, 2])
  @inline(__always)
  private func gaspariCohn(r: Double) -> Double {
    let x = abs(r)
    if x >= 2.0 { return 0.0 }
    if x <= 1.0 {
      // 1 - 5/3 r^2 + 5/8 r^3 + 1/2 r^4 - 1/4 r^5
      let r2 = x * x
      let r3 = r2 * x
      let r4 = r2 * r2
      let r5 = r3 * r2
      return 1.0 - (5.0/3.0)*r2 + (5.0/8.0)*r3 + 0.5*r4 - 0.25*r5
    } else {
      // 4 - 5 r + 5/3 r^2 + 5/8 r^3 - 1/2 r^4 + 1/12 r^5
      let r2 = x * x
      let r3 = r2 * x
      let r4 = r2 * r2
      let r5 = r3 * r2
      return 4.0 - 5.0*x + (5.0/3.0)*r2 + (5.0/8.0)*r3 - 0.5*r4 + (1.0/12.0)*r5
    }
  }

  // Cholesky factorization (lower) for small SPD matrices
  private func choleskyLowerSPD(_ A: Matrix) -> Matrix? {
    let n = A.rows
    precondition(n == A.cols)
    var L = Matrix(rows: n, cols: n)
    for i in 0..<n {
      for j in 0...i {
        var sum = A[i, j]
        for k in 0..<j { sum -= L[i, k] * L[j, k] }
        if i == j {
          if sum <= 0 { return nil }
          L[i, j] = sqrt(max(sum, 1e-12))
        } else {
          L[i, j] = sum / L[j, j]
        }
      }
    }
    return L
  }
}

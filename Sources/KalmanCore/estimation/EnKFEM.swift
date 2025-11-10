import Foundation

/*
 EnKF–EM (Section 3, Pulido et al. 2018)
 --------------------------------------
 This file implements a windowed Ensemble Kalman Filter (EnKF) combined with
 Expectation–Maximization (EM) for sequential parameter estimation. The tight
 EM loop re-assimilates the window after each M-step, updating the augmented
 parameter state θ and reflecting its impact on the forecast/analysis.

 References
 - Pulido, M., Tandeo, P., Bocquet, M., Carrassi, A., & Lucini, M. (2018).
   Stochastic parameterization identification using ensemble Kalman filtering
   combined with maximum likelihood methods. Tellus A 70:1, 1–17. (Sec. 3)
 - Evensen, G. (1994). Sequential data assimilation with a nonlinear
   quasi-geostrophic model using Monte Carlo methods to forecast error statistics.
   J. Geophys. Res.
*/

/// Windowed EnKF-driven EM (expectation–maximization) parameter updates (scaffold)
/// Uses EnKF outputs over short windows to approximate E-step statistics,
/// then performs one or more M-steps to update θ.
public struct EnKFEMConfig {
  public var window: Int = 50
  public var maxEMItersPerWindow: Int = 1
  public var parameterBounds: (min: Double, max: Double) = (1e-8, 100.0)
  public var verbose: Bool = true
  public init() {}
}

public struct EnKFEMDiagnostics {
  public let iterations: Int
  public let logLikelihoodBefore: Double?
  public let logLikelihoodAfter: Double?
  // Observation diagnostics (supports partial observation models)
  /// Number of observed dimensions (m)
  public let observedCount: Int
  /// Fraction of observed state variables (m/n)
  public let observedFraction: Double
  /// Average L2-norm of innovations over the window
  public let meanInnovationNorm: Double
  /// Mean squared innovation per observed dimension over the window
  public let innovationVariance: Double

  /// Human-readable summary for quick inspection
  public func summary() -> String {
    return """
    EnKFEM Diagnostics:
      Iterations: \(iterations)
      Observed dims: \(observedCount) (fraction: \(String(format: "%.3f", observedFraction)))
      Mean innovation norm: \(String(format: "%.4f", meanInnovationNorm))
      Innovation variance (per-dim): \(String(format: "%.6f", innovationVariance))
    """.trimmingCharacters(in: .whitespacesAndNewlines)
  }
}

extension EnKFEMDiagnostics: CustomStringConvertible {
  public var description: String { summary() }
}

///  EnKF-EM expectation–maximization algorithm
public final class EnKFEM<Model: StochasticDynamicalSystem> {
  public let model: Model
  public let observationModel: ObservationModel
  public let enkf: EnsembleKalmanFilter<Model>
  public let config: EnKFEMConfig

  public init(model: Model, observationModel: ObservationModel, enkf: EnsembleKalmanFilter<Model>, config: EnKFEMConfig = EnKFEMConfig()) {
    self.model = model
    self.observationModel = observationModel
    self.enkf = enkf
    self.config = config
  }

  /// Run EnKF over a window and execute EM (expectation–maximization) M-steps.
  /// Implements a tight loop for additive noise (Pulido et al., 2018, Sec. 3):
  ///   - For each EM iteration: reset θ in the initial ensemble, assimilate the window,
  ///     compute window statistics, perform the additive-noise M-step, and repeat.
  /// Returns updated parameters, the final analyzed ensemble, and diagnostics.
  public func runWindow(
    observations: [[Double]],
    dt: Double,
    initialParameters: [Double],
    initialEnsemble: Ensemble
  ) -> (newParameters: [Double], newEnsemble: Ensemble, diagnostics: EnKFEMDiagnostics) {
    precondition(!observations.isEmpty, "Need at least one observation")

    var currentParams = initialParameters
    var finalEnsemble = initialEnsemble

    let n = model.stateDimension
    let m = observationModel.observationDimension
    let L = min(config.window, observations.count)

    // Local helper: overwrite parameter components θ in augmented ensemble
    func overwriteParameters(in ensemble: Ensemble, with theta: [Double]) -> Ensemble {
      let p = model.parameterDimension
      precondition(theta.count == p, "Parameter dimension mismatch")
      var members = ensemble.members
      for k in 0..<members.count {
        for j in 0..<p {
          members[k][n + j] = theta[j]
        }
      }
      return Ensemble(members: members)
    }

    // One pass of window assimilation + stats
    func assimilateWindow(with params: [Double]) -> (
      ensemble: Ensemble,
      means: [[Double]],
      covTraces: [Double],
      innovationNormSum: Double,
      innovationSqSum: Double
    ) {
      var Z = overwriteParameters(in: initialEnsemble, with: params)

      var filteredMeans: [[Double]] = []
      var filteredCovTraces: [Double] = []
      filteredMeans.reserveCapacity(L)
      filteredCovTraces.reserveCapacity(L)

      var innovationNormSum = 0.0
      var innovationSqSum = 0.0

      for t in 0..<L {
        Z = enkf.forecast(ensemble: Z, dt: dt)

        // Predicted mean observation using forecasted state mean
        let stateMembersForecast = Z.members.map { Array($0[0..<n]) }
        let stateEnsForecast = Ensemble(members: stateMembersForecast)
        let xBarForecast = stateEnsForecast.mean
        let yHat = observationModel.observationOperator(state: xBarForecast)

        // Innovation diagnostics
        let innov = zip(observations[t], yHat).map { $0 - $1 }
        let innovNorm = sqrt(innov.reduce(0.0) { $0 + $1*$1 })
        innovationNormSum += innovNorm
        innovationSqSum += innov.reduce(0.0) { $0 + $1*$1 } / Double(m)

        // Analysis update
        Z = enkf.analyze(ensemble: Z, observation: observations[t])

        // Extract state-only ensemble for mean/cov after analysis
        let stateMembers = Z.members.map { Array($0[0..<n]) }
        let stateEnsemble = Ensemble(members: stateMembers)
        filteredMeans.append(stateEnsemble.mean)
        filteredCovTraces.append(stateEnsemble.covariance.trace)
      }

      return (Z, filteredMeans, filteredCovTraces, innovationNormSum, innovationSqSum)
    }

    // Closed-form additive M-step (single-parameter σ)
    func additiveMStep(using means: [[Double]], covTraces: [Double]) -> [Double] {
      guard !means.isEmpty, model.parameterDimension == 1 else { return currentParams }
      var sumSquaredResiduals = 0.0
      for t in 0..<(means.count - 1) {
        let x_t = means[t]
        let x_tp1 = means[t + 1]
        let drift = model.deterministicDynamics(state: x_t, parameters: [])
        let predicted = zip(x_t, drift).map { $0 + dt * $1 }
        for i in 0..<n { sumSquaredResiduals += (x_tp1[i] - predicted[i]) * (x_tp1[i] - predicted[i]) }
        sumSquaredResiduals += covTraces[t + 1]
      }
      let denom = Double(n) * Double(max(1, means.count - 1)) * dt
      let variance = max(config.parameterBounds.min, sumSquaredResiduals / denom)
      let sigma = sqrt(variance)
      let bounded = min(max(sigma, config.parameterBounds.min), config.parameterBounds.max)
      return [bounded]
    }

    var totalInnovationNorm = 0.0
    var totalInnovationVar = 0.0

    let emIters = max(1, config.maxEMItersPerWindow)
    for _ in 0..<emIters {
      // Assimilate window with current params
      let (Zafter, means, covTraces, innovNormSum, innovSqSum) = assimilateWindow(with: currentParams)
      finalEnsemble = Zafter
      totalInnovationNorm += innovNormSum / Double(L)
      totalInnovationVar += innovSqSum / Double(L)

      // M-step
      currentParams = additiveMStep(using: means, covTraces: covTraces)
    }

    // Diagnostics assembly (averaged over EM iterations)
    let observedCount = m
    let observedFraction = Double(m) / Double(n)
    let meanInnovationNorm = totalInnovationNorm / Double(emIters)
    let innovationVariance = totalInnovationVar / Double(emIters)

    let diags = EnKFEMDiagnostics(
      iterations: emIters,
      logLikelihoodBefore: nil,
      logLikelihoodAfter: nil,
      observedCount: observedCount,
      observedFraction: observedFraction,
      meanInnovationNorm: meanInnovationNorm,
      innovationVariance: innovationVariance
    )

    return (newParameters: currentParams, newEnsemble: finalEnsemble, diagnostics: diags)
  }
}

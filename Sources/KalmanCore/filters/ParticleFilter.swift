import Foundation

/*
 Particle Filter (Bootstrap/SIR)
 ------------------------------
 Basic Sequential Importance Resampling (SIR) particle filter for nonlinear/non-Gaussian models.
 Uses the model.transition to propagate particles and a Gaussian likelihood with the provided
 ObservationModel (y | x ~ N(h(x), R)).

 Notes
 - This is a minimal, general-purpose PF suitable for examples and baselines.
 - Resampling: systematic by default; multinomial available.
 */

public final class ParticleFilter<Model: StochasticDynamicalSystem> {
  public enum ResamplingMethod { case systematic, multinomial }

  public struct Config {
    public var numParticles: Int
    public var resamplingThreshold: Double = 0.5   // as fraction of N (ESS/N)
    public var resamplingMethod: ResamplingMethod = .systematic
    public var verbose: Bool = false
    public init(numParticles: Int,
                resamplingThreshold: Double = 0.5,
                resamplingMethod: ResamplingMethod = .systematic,
                verbose: Bool = false) {
      self.numParticles = numParticles
      self.resamplingThreshold = resamplingThreshold
      self.resamplingMethod = resamplingMethod
      self.verbose = verbose
    }
  }

  public struct State {
    public var particles: [[Double]]
    public var weights: [Double] // normalized
  }

  public struct StepResult {
    public let ess: Double
    public let resampled: Bool
    public let logLikelihoodIncrement: Double
  }

  public let model: Model
  public let observationModel: ObservationModel
  public let config: Config

  public var parameters: [Double]
  public var dt: Double
  public var state: State

  // Initialize by sampling from Gaussian prior N(x0, P0)
  public init(model: Model,
              observationModel: ObservationModel,
              x0: [Double],
              P0: Matrix,
              parameters: [Double],
              dt: Double,
              config: Config) {
    precondition(x0.count == model.stateDimension, "State dimension mismatch")
    precondition(P0.rows == model.stateDimension && P0.cols == model.stateDimension, "Covariance size mismatch")
    precondition(parameters.count == model.parameterDimension, "Parameter dimension mismatch")
    precondition(observationModel.stateDimension == model.stateDimension, "Obs/state dimension mismatch")

    self.model = model
    self.observationModel = observationModel
    self.parameters = parameters
    self.dt = dt
    self.config = config

    var parts: [[Double]] = []
    parts.reserveCapacity(config.numParticles)
    for _ in 0..<config.numParticles {
      let noise = RandomUtils.generateGaussianNoiseWithCovariance(dimension: x0.count, covariance: P0)
      parts.append(vectorAdd(x0, noise))
    }
    let w0 = Array(repeating: 1.0 / Double(config.numParticles), count: config.numParticles)
    self.state = State(particles: parts, weights: w0)
  }

  // Predict: x_i <- f(x_i)
  @discardableResult
  public func predict() -> State {
    for i in 0..<state.particles.count {
      state.particles[i] = model.transition(state: state.particles[i], parameters: parameters, dt: dt)
    }
    return state
  }

  // Update weights by Gaussian likelihood and resample if needed
  @discardableResult
  public func update(y: [Double]) -> (state: State, result: StepResult) {
    let m = observationModel.observationDimension
    precondition(y.count == m)

    // Compute log-weights: log w_i ∝ log N(y; h(x_i), R)
    let R = observationModel.observationNoiseCovariance
    var logw = [Double](repeating: 0.0, count: state.particles.count)
    var maxlogw = -Double.infinity
    for i in 0..<state.particles.count {
      let yi = observationModel.observationOperator(state: state.particles[i])
      let innov = vectorSubtract(y, yi)
      let ll = Likelihood.gaussianInnovationLogLikelihood(innovation: innov, covariance: R)
      logw[i] = ll
      if ll > maxlogw { maxlogw = ll }
    }

    // Normalize in log-space to avoid underflow
    var w = [Double](repeating: 0.0, count: state.particles.count)
    var sumw = 0.0
    for i in 0..<w.count { w[i] = exp(logw[i] - maxlogw); sumw += w[i] }
    if sumw == 0 { // pathological, assign uniform
      let u = 1.0 / Double(w.count)
      for i in 0..<w.count { w[i] = u }
      sumw = 1.0
    }
    for i in 0..<w.count { w[i] /= sumw }

    // Effective sample size
    let ess = 1.0 / w.reduce(0.0) { $0 + $1 * $1 }

    // Resample if ESS/N low
    var resampled = false
    if ess / Double(w.count) < config.resamplingThreshold {
      resampled = true
      let idxs = (config.resamplingMethod == .systematic) ? systematicResample(weights: w)
      : multinomialResample(weights: w)
      var newParts = [[Double]](repeating: state.particles[0], count: w.count)
      for (j, i) in idxs.enumerated() { newParts[j] = state.particles[i] }
      state.particles = newParts
      w = Array(repeating: 1.0 / Double(w.count), count: w.count)
    }

    state.weights = w

    // PF marginal likelihood estimate increment (log of average unnormalized weight)
    let avgUnnorm = sumw / Double(w.count)
    let logLikInc = log(avgUnnorm) + maxlogw - log(Double(w.count))

    return (state, StepResult(ess: ess, resampled: resampled, logLikelihoodIncrement: logLikInc))
  }

  @discardableResult
  public func step(y: [Double]) -> (state: State, result: StepResult) {
    _ = predict()
    return update(y: y)
  }

  // MARK: - Resampling helpers
  private func systematicResample(weights w: [Double]) -> [Int] {
    let N = w.count
    let u0 = Double.random(in: 0..<1.0) / Double(N)
    var cdf = [Double](repeating: 0.0, count: N)
    cdf[0] = w[0]
    for i in 1..<N { cdf[i] = cdf[i-1] + w[i] }
    var idxs = [Int]()
    idxs.reserveCapacity(N)
    var i = 0
    for m in 0..<N {
      let u = u0 + Double(m) / Double(N)
      while u > cdf[i] && i < N-1 { i += 1 }
      idxs.append(i)
    }
    return idxs
  }

  private func multinomialResample(weights w: [Double]) -> [Int] {
    let N = w.count
    var idxs = [Int]()
    idxs.reserveCapacity(N)
    for _ in 0..<N {
      let r = Double.random(in: 0..<1.0)
      var cum = 0.0
      var k = 0
      while k < N-1 && cum + w[k] < r { cum += w[k]; k += 1 }
      idxs.append(k)
    }
    return idxs
  }
}

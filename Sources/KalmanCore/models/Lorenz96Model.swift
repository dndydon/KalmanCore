import Foundation

/// Lorenz96 model with stochastic parameterization (Section 2.1 example)
///
/// Deterministic dynamics:
/// dx_i/dt = (x_{i+1} - x_{i-2})x_{i-1} - x_i + F
///
/// Stochastic parameterization:
/// σ(x, θ) = θ_0 * I  (additive noise)
/// or
/// σ(x, θ) = diag(θ_i)  (state-dependent noise)
///
/// This is a classic chaotic system used for testing data assimilation methods.
/// Reference: Lorenz, E. N. (1996). Predictability: A problem partly solved.
public class Lorenz96Model: BaseStochasticSystem {
  /// Forcing parameter F (typical value: 8.0)
  public let forcing: Double

  /// Type of stochastic parameterization
  public enum StochasticType {
    case additive        // σ = θ_0 * I
    case diagonal        // σ = diag(θ_0, θ_1, ..., θ_{n-1})
    case stateDependent  // σ_i = θ_0 * |x_i|
  }

  public let stochasticType: StochasticType

  /// Initialize Lorenz96 model
  /// - Parameters:
  ///   - dimension: Number of state variables (typically 40)
  ///   - forcing: Forcing parameter F (default: 8.0)
  ///   - stochasticType: Type of stochastic parameterization
  public init(dimension: Int, forcing: Double = 8.0, stochasticType: StochasticType = .additive) {
    self.forcing = forcing
    self.stochasticType = stochasticType

    let paramDim: Int
    switch stochasticType {
      case .additive, .stateDependent:
        paramDim = 1  // Single parameter θ_0
      case .diagonal:
        paramDim = dimension  // One parameter per state variable
    }

    super.init(stateDimension: dimension, parameterDimension: paramDim)
  }

  /// Deterministic Lorenz96 dynamics
  override public func deterministicDynamics(state: [Double], parameters: [Double]) -> [Double] {
    let n = stateDimension
    var dxdt = [Double](repeating: 0.0, count: n)

    for i in 0..<n {
      let im2 = (i - 2 + n) % n  // Periodic boundary conditions
      let im1 = (i - 1 + n) % n
      let ip1 = (i + 1) % n

      // Classic Lorenz96 equation
      dxdt[i] = (state[ip1] - state[im2]) * state[im1] - state[i] + forcing
    }

    return dxdt
  }

  /// Stochastic parameterization
  override public func stochasticParameterization(state: [Double], parameters: [Double]) -> Matrix {
    let n = stateDimension

    switch stochasticType {
      case .additive:
        // Simple additive noise: σ = θ_0 * I
        let sigma = parameters[0]
        return sigma * Matrix.identity(size: n)

      case .diagonal:
        // Diagonal noise: σ = diag(θ_0, θ_1, ..., θ_{n-1})
        return Matrix.diagonal(parameters)

      case .stateDependent:
        // State-dependent noise: σ_i = θ_0 * |x_i|
        let theta = parameters[0]
        let diagonalElements = state.map { abs($0) * theta }
        return Matrix.diagonal(diagonalElements)
    }
  }
}

// MARK: - Convenience Methods

extension Lorenz96Model {
  /// Create standard Lorenz96 model (dimension 40, F=8)
  public static func standard(stochasticType: StochasticType = .additive) -> Lorenz96Model {
    return Lorenz96Model(dimension: 40, forcing: 8.0, stochasticType: stochasticType)
  }

  /// Generate typical initial condition (near attractor)
  public func typicalInitialState() -> [Double] {
    var state = Array(repeating: forcing, count: stateDimension)
    // Add small perturbation to break symmetry
    state[0] += 0.01
    return state
  }

  /// Spin up the model to attractor
  /// - Parameters:
  ///   - initialState: Initial state
  ///   - parameters: Model parameters
  ///   - spinUpTime: Time to spin up (default: 5.0)
  ///   - dt: Time step (default: 0.01)
  /// - Returns: State on attractor
  public func spinUp(
    initialState: [Double],
    parameters: [Double],
    spinUpTime: Double = 5.0,
    dt: Double = 0.01
  ) -> [Double] {
    let steps = Int(spinUpTime / dt)
    var state = initialState

    // Use deterministic dynamics for spin-up
    for _ in 0..<steps {
      let drift = deterministicDynamics(state: state, parameters: parameters)
      for i in 0..<stateDimension {
        state[i] += drift[i] * dt
      }
    }

    return state
  }

  /// Compute Lyapunov exponent (for chaos analysis)
  /// Approximate largest Lyapunov exponent using finite difference method
  public func approximateLyapunovExponent(
    initialState: [Double],
    parameters: [Double],
    integrationTime: Double = 100.0,
    dt: Double = 0.01
  ) -> Double {
    let steps = Int(integrationTime / dt)
    var state = initialState
    var perturbation = Array(repeating: 1e-8, count: stateDimension)

    var sumLog = 0.0
    let normalizationInterval = 10

    for step in 0..<steps {
      // Evolve state
      let drift = deterministicDynamics(state: state, parameters: parameters)
      for i in 0..<stateDimension {
        state[i] += drift[i] * dt
      }

      // Evolve perturbation (linearized dynamics)
      let tangent = computeTangentDynamics(state: state, perturbation: perturbation)
      for i in 0..<stateDimension {
        perturbation[i] += tangent[i] * dt
      }

      // Periodically renormalize perturbation
      if step % normalizationInterval == 0 {
        let norm = sqrt(perturbation.reduce(0) { $0 + $1 * $1 })
        sumLog += log(norm)
        perturbation = perturbation.map { $0 / norm }
      }
    }

    return sumLog / integrationTime
  }

  /// Compute tangent dynamics for Lyapunov calculation
  private func computeTangentDynamics(state: [Double], perturbation: [Double]) -> [Double] {
    let n = stateDimension
    var tangent = [Double](repeating: 0.0, count: n)

    for i in 0..<n {
      let im2 = (i - 2 + n) % n
      let im1 = (i - 1 + n) % n
      let ip1 = (i + 1) % n

      // Jacobian-vector product
      tangent[i] = (state[ip1] - state[im2]) * perturbation[im1]
      + state[im1] * perturbation[ip1]
      - state[im1] * perturbation[im2]
      - perturbation[i]

      // Handle wraparound contributions
      if i == 0 {
        tangent[i] += state[n-1] * perturbation[1]
      }
    }

    return tangent
  }
}

// MARK: - Statistics and Analysis

extension Lorenz96Model {
  /// Compute climatological mean from trajectory
  public static func climatologicalMean(trajectory: [[Double]]) -> [Double] {
    guard !trajectory.isEmpty else { return [] }

    let n = trajectory[0].count
    let m = trajectory.count
    var mean = [Double](repeating: 0.0, count: n)

    for state in trajectory {
      for i in 0..<n {
        mean[i] += state[i]
      }
    }

    return mean.map { $0 / Double(m) }
  }

  /// Compute climatological covariance from trajectory
  public static func climatologicalCovariance(trajectory: [[Double]]) -> Matrix {
    guard !trajectory.isEmpty else { return Matrix(rows: 0, cols: 0) }

    let n = trajectory[0].count
    let mean = climatologicalMean(trajectory: trajectory)
    var cov = Matrix(rows: n, cols: n)

    for state in trajectory {
      for i in 0..<n {
        for j in 0..<n {
          let devI = state[i] - mean[i]
          let devJ = state[j] - mean[j]
          cov[i, j] += devI * devJ
        }
      }
    }

    let scale = 1.0 / Double(trajectory.count - 1)
    for i in 0..<n {
      for j in 0..<n {
        cov[i, j] *= scale
      }
    }

    return cov
  }
}

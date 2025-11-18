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

  /// Deterministic Lorenz96 dynamics (continuous-time right-hand side)
  /// Implements the classic n-dimensional Lorenz-96 system with cyclic (periodic)
  /// indexing:
  ///   dx_i/dt = (x_{i+1} - x_{i-2}) x_{i-1} - x_i + F
  /// where indices are taken modulo n. The three terms are:
  ///   - (x_{i+1} - x_{i-2}) x_{i-1}: nonlinear advection-like coupling
  ///   - −x_i: linear damping
  ///   - +F: constant external forcing (set by `forcing` property)
  /// Notes
  /// - `parameters` are not used here — they control the stochastic parameterization σ(x,θ)
  ///   in `stochasticParameterization`; the deterministic drift uses `forcing` only.
  /// - This function returns the instantaneous time derivative dx/dt; time discretization
  ///   (e.g., Euler–Maruyama) is performed by callers like `transition(state:parameters:dt:)`.
  override public func deterministicDynamics(state: [Double], parameters: [Double]) -> [Double] {
    let n = stateDimension

    /* comment out if you hate inline tests
    precondition(state.count == n, "Lorenz96Model.deterministicDynamics: state.count (\(state.count)) must equal stateDimension (\(n))")
    // Enforce parameter arity based on stochasticType. Deterministic drift doesn't
    // use parameters numerically, but we validate shape to catch test misconfigurations.
    switch stochasticType {
      case .diagonal:
        // Must provide one parameter per state variable
        precondition(parameters.count == parameterDimension, "Lorenz96Model.deterministicDynamics: parameters.count (\(parameters.count)) must equal parameterDimension (\(parameterDimension)) for .diagonal stochasticType")
      case .additive, .stateDependent:
        // Must provide empty (zero count) parameter for one state variable
        // Expect a single parameter (theta_0). Allow empty for callers that omit it
        // during purely deterministic steps.
        if parameters.count == 0 {
          //print("\(parameters.count) parameters implies parameterDimension == 1 (\(parameterDimension)) for \(stochasticType)")
          precondition(parameterDimension == 1, "parameters.count (\(parameters.count)) may be equal to zero (0) for \(stochasticType)")
        } else {
          if parameters.count > 1 {
            fatalError("WARNING: \(parameters.count) parameters for \(stochasticType); only using first (\(parameterDimension))")
          }
          //print("\(parameters.count) parameters: expecting exactly one (\(parameterDimension)) parameterDimension for \(stochasticType)")
          precondition(parameters.count == parameterDimension, "Lorenz96Model.deterministicDynamics: parameters.count (\(parameters.count)) must equal parameterDimension (\(parameterDimension)) for \(stochasticType)")
        }
    }
    */

    // Allocate derivative vector
    var dxdt = [Double](repeating: 0.0, count: n)

    for i in 0..<n {
      // Cyclic neighbor indices with periodic boundary conditions:
      // im2 = i-2, im1 = i-1, ip1 = i+1 (all wrapped into [0, n))
      let im2 = (i - 2 + n) % n
      let im1 = (i - 1 + n) % n
      let ip1 = (i + 1) % n

      // Apply Lorenz-96 formula term-by-term for clarity
      let coupling = (state[ip1] - state[im2]) * state[im1]  // nonlinear neighbor interaction
      let damping  = -state[i]                               // linear damping
      let drive    = forcing                                 // constant forcing F

      dxdt[i] = coupling + damping + drive
    }

    // Return dx/dt; integrators will scale by dt and add to the current state
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
  /// Compute climatological mean from trajectories
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


import Foundation

/// Protocol for stochastic dynamical systems (Section 2.1)
/// Represents systems of the form:
/// dx/dt = M(x, θ) + σ(x, θ)ξ(t)
/// where:
/// - x: state vector
/// - θ: parameters to be identified
/// - M: deterministic dynamics
/// - σ: stochastic parameterization (diffusion term)
/// - ξ: white noise process
public protocol StochasticDynamicalSystem {
  /// Dimension of the state vector
  var stateDimension: Int { get }

  /// Dimension of the parameter vector
  var parameterDimension: Int { get }

  /// Deterministic dynamics: M(x, θ)
  /// - Parameters:
  ///   - state: Current state vector x
  ///   - parameters: Parameter vector θ
  /// - Returns: Deterministic drift term
  func deterministicDynamics(state: [Double], parameters: [Double]) -> [Double]

  /// Stochastic parameterization: σ(x, θ)
  /// Returns the diffusion matrix
  /// - Parameters:
  ///   - state: Current state vector x
  ///   - parameters: Parameter vector θ
  /// - Returns: Diffusion matrix (can be state and parameter dependent)
  func stochasticParameterization(state: [Double], parameters: [Double]) -> Matrix

  /// Discretized state transition using Euler-Maruyama scheme
  /// x_{k+1} = x_k + M(x_k, θ)Δt + σ(x_k, θ)√(Δt)W_k
  /// where W_k ~ N(0, I)
  /// - Parameters:
  ///   - state: Current state x_k
  ///   - parameters: Parameters θ
  ///   - dt: Time step Δt
  /// - Returns: Next state x_{k+1}
  func transition(state: [Double], parameters: [Double], dt: Double) -> [Double]
}

// MARK: - Default Implementation

public extension StochasticDynamicalSystem {
  /// Default implementation of Euler-Maruyama discretization
  func transition(state: [Double], parameters: [Double], dt: Double) -> [Double] {
    precondition(state.count == stateDimension, "State dimension mismatch")
    precondition(parameters.count == parameterDimension, "Parameter dimension mismatch")

    // Drift term: M(x_k, θ)Δt
    let drift = deterministicDynamics(state: state, parameters: parameters)

    // Diffusion term: σ(x_k, θ)√(Δt)W_k
    let diffusion = stochasticParameterization(state: state, parameters: parameters)
    let noise = RandomUtils.generateGaussianNoise(dimension: stateDimension)
    let scaledNoise = noise.map { $0 * sqrt(dt) }
    let stochasticTerm = diffusion.multiply(vector: scaledNoise)

    // x_{k+1} = x_k + drift * dt + stochasticTerm
    var nextState = [Double](repeating: 0.0, count: stateDimension)
    for i in 0..<stateDimension {
      nextState[i] = state[i] + drift[i] * dt + stochasticTerm[i]
    }

    return nextState
  }

  /// Simulate trajectory over multiple time steps
  /// - Parameters:
  ///   - initialState: Initial state x_0
  ///   - parameters: Parameters θ
  ///   - dt: Time step
  ///   - steps: Number of steps
  /// - Returns: Array of states [x_0, x_1, ..., x_n]
  func simulateTrajectory(
    initialState: [Double],
    parameters: [Double],
    dt: Double,
    steps: Int
  ) -> [[Double]] {
    var trajectory = [[Double]]()
    trajectory.reserveCapacity(steps + 1)

    var currentState = initialState
    trajectory.append(currentState)

    for _ in 0..<steps {
      currentState = transition(state: currentState, parameters: parameters, dt: dt)
      trajectory.append(currentState)
    }

    return trajectory
  }

  /// Simulate ensemble of trajectories
  /// - Parameters:
  ///   - ensemble: Initial ensemble
  ///   - parameters: Parameters θ
  ///   - dt: Time step
  /// - Returns: Ensemble of next states
  func forecastEnsemble(
    ensemble: Ensemble,
    parameters: [Double],
    dt: Double
  ) -> Ensemble {
    var forecastMembers = [[Double]]()
    forecastMembers.reserveCapacity(ensemble.ensembleSize)

    for member in ensemble.members {
      let nextState = transition(state: member, parameters: parameters, dt: dt)
      forecastMembers.append(nextState)
    }

    return Ensemble(members: forecastMembers)
  }
}

/// Base class for implementing stochastic dynamical systems
open class BaseStochasticSystem: StochasticDynamicalSystem {
  public let stateDimension: Int
  public let parameterDimension: Int

  public init(stateDimension: Int, parameterDimension: Int) {
    self.stateDimension = stateDimension
    self.parameterDimension = parameterDimension
  }

  /// Override in subclass
  open func deterministicDynamics(state: [Double], parameters: [Double]) -> [Double] {
    fatalError("Must override deterministicDynamics in subclass")
  }

  /// Override in subclass
  open func stochasticParameterization(state: [Double], parameters: [Double]) -> Matrix {
    fatalError("Must override stochasticParameterization in subclass")
  }
}

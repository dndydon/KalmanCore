import Foundation

/*
 Extended Kalman Filter (EKF)
 ---------------------------
 Nonlinear state-space model with local linearization:
   x_{k+1} = f(x_k, θ, dt) + w_k,   w_k ~ N(0, Q_k)
   y_k     = h(x_k) + v_k,          v_k ~ N(0, R)
 Uses finite-difference linearization to approximate F = ∂f/∂x and H = ∂h/∂x,
 and performs linear KF update on the linearized system.

 Notes
 - For SPD covariances, Cholesky-based solvers are preferable; current code uses
   general routines for clarity (see MatrixUtils).
 - When analytical Jacobians are available, they can replace the finite-diff ones
   via a future LinearizableModel interface.

 References
 - Jazwinski, A.H. (1970). Stochastic Processes and Filtering Theory.
 - Gelb, A. (1974). Applied Optimal Estimation.
*/

/// Extended Kalman Filter for nonlinear systems
/// x_{k+1} = f(x_k, θ, dt) + w_k,   w_k ~ N(0, Q_k)
/// y_k     = h(x_k) + v_k,          v_k ~ N(0, R)
public final class ExtendedKalmanFilter<Model: StochasticDynamicalSystem> {
  public struct Config {
    public var verbose: Bool = false
    public var epsilon: Double = 1e-7 // finite-diff step
    public init() {}
  }

  public struct State {
    public var x: [Double]
    public var P: Matrix
  }

  public struct StepResult {
    public let innovation: [Double]
    public let innovationCov: Matrix // S
    public let gain: Matrix          // K
    public let logLikelihoodIncrement: Double
    public let F: Matrix
    public let H: Matrix
  }

  public let model: Model
  public let observationModel: ObservationModel
  public let config: Config

  public var parameters: [Double]
  public var dt: Double
  public var state: State

  public init(model: Model,
              observationModel: ObservationModel,
              initialState: [Double],
              initialCovariance: Matrix,
              parameters: [Double],
              dt: Double,
              config: Config = Config()) {
    precondition(initialState.count == model.stateDimension, "State dimension mismatch")
    precondition(initialCovariance.rows == model.stateDimension && initialCovariance.cols == model.stateDimension, "Covariance size mismatch")
    precondition(parameters.count == model.parameterDimension, "Parameter dimension mismatch")
    precondition(observationModel.stateDimension == model.stateDimension, "Obs/state dimension mismatch")

    self.model = model
    self.observationModel = observationModel
    self.parameters = parameters
    self.dt = dt
    self.state = State(x: initialState, P: initialCovariance)
    self.config = config
  }

  // Predict step using model.transition and linearized F; Q from σ(x,θ)
  @discardableResult
  public func predict() -> (state: State, F: Matrix) {
    let xPred = model.transition(state: state.x, parameters: parameters, dt: dt)
    let F = Linearization.approximateStateJacobian(model: model, state: state.x, parameters: parameters, dt: dt, epsilon: config.epsilon)
    let sigma = model.stochasticParameterization(state: state.x, parameters: parameters)
    let Q = dt * (sigma * sigma.transposed)
    let PPred = F * state.P * F.transposed + Q
    state = State(x: xPred, P: PPred)
    return (state, F)
  }

  // Update step using linearized H
  @discardableResult
  public func update(y: [Double]) -> (state: State, result: StepResult) {
    precondition(y.count == observationModel.observationDimension, "Observation dimension mismatch")

    // Linearize around current predicted state
    let H = Linearization.approximateObservationJacobian(observationModel: observationModel, state: state.x, epsilon: config.epsilon)

    // Innovation
    let yPred = observationModel.observationOperator(state: state.x)
    let innovation = vectorSubtract(y, yPred)

    // Innovation covariance S and Kalman gain K
    let R = observationModel.observationNoiseCovariance
    let S = H * state.P * H.transposed + R
    let Sinv = matrixInverse(S)
    let K = state.P * H.transposed * Sinv

    // State update
    let xUpd = vectorAdd(state.x, K.multiply(vector: innovation))
    let I = Matrix.identity(size: state.P.rows)
    let PUpd = (I - K * H) * state.P

    state = State(x: xUpd, P: PUpd)

    let ll = Likelihood.gaussianInnovationLogLikelihood(innovation: innovation, covariance: S)

    // Also return F used in last predict for debugging; recompute here for clarity
    let F = Linearization.approximateStateJacobian(model: model, state: xUpd, parameters: parameters, dt: dt, epsilon: config.epsilon)

    return (state, StepResult(innovation: innovation, innovationCov: S, gain: K, logLikelihoodIncrement: ll, F: F, H: H))
  }

  @discardableResult
  public func step(y: [Double]) -> (state: State, result: StepResult) {
    _ = predict()
    return update(y: y)
  }
}

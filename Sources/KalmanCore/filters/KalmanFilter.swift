import Foundation

/*
 Linear Kalman Filter (KF)
 ------------------------
 Time-invariant linear-Gaussian state-space model:
   x_{k+1} = F x_k + w_k,   w_k ~ N(0, Q)
   y_k     = H x_k + v_k,   v_k ~ N(0, R)
 Implements the standard predict/update recursions and returns innovation stats
 and the per-step log-likelihood increment under Gaussian assumptions.

 Reference
 - Kalman, R.E. (1960). A new approach to linear filtering and prediction problems.
*/

/// Linear Kalman Filter for time-invariant systems
/// x_{k+1} = F x_k + w_k,   w_k ~ N(0, Q)
/// y_k     = H x_k + v_k,   v_k ~ N(0, R)
public final class KalmanFilter {
  // Configuration (extend as needed)
  public struct Config {
    public var verbose: Bool = false
    public init() {}
  }

  // State container
  public struct State {
    public var x: [Double]
    public var P: Matrix
  }

  // Step diagnostics/result
  public struct StepResult {
    public let innovation: [Double]
    public let innovationCov: Matrix // S
    public let gain: Matrix          // K
    public let logLikelihoodIncrement: Double
  }

  public let F: Matrix
  public let Q: Matrix
  public let H: Matrix
  public let R: Matrix
  public var state: State
  public let config: Config

  public init(F: Matrix, Q: Matrix, H: Matrix, R: Matrix, x0: [Double], P0: Matrix, config: Config = Config()) {
    precondition(F.rows == F.cols, "F must be square")
    precondition(Q.rows == F.rows && Q.cols == F.cols, "Q size mismatch")
    precondition(H.cols == F.rows, "H/F dimension mismatch")
    precondition(R.rows == H.rows && R.cols == H.rows, "R size mismatch")
    precondition(x0.count == F.rows, "x0 dimension mismatch")
    precondition(P0.rows == F.rows && P0.cols == F.rows, "P0 size mismatch")

    self.F = F
    self.Q = Q
    self.H = H
    self.R = R
    self.state = State(x: x0, P: P0)
    self.config = config
  }

  // Predict: x <- F x,  P <- F P F^T + Q
  @discardableResult
  public func predict() -> State {
    let xPred = F.multiply(vector: state.x)
    let PPred = F * state.P * F.transposed + Q
    state = State(x: xPred, P: PPred)
    return state
  }

  // Update with observation y
  @discardableResult
  public func update(y: [Double]) -> (state: State, result: StepResult) {
    precondition(y.count == H.rows, "Observation dimension mismatch")

    // Innovation
    let yPred = H.multiply(vector: state.x)
    let innovation = vectorSubtract(y, yPred)

    // Innovation covariance and gain
    let S = H * state.P * H.transposed + R
    let Sinv = matrixInverse(S)
    let K = state.P * H.transposed * Sinv

    // State update
    let xUpd = vectorAdd(state.x, K.multiply(vector: innovation))
    let I = Matrix.identity(size: state.P.rows)
    let PUpd = (I - K * H) * state.P

    state = State(x: xUpd, P: PUpd)

    // Log-likelihood increment
    let ll = Likelihood.gaussianInnovationLogLikelihood(innovation: innovation, covariance: S)

    return (state, StepResult(innovation: innovation, innovationCov: S, gain: K, logLikelihoodIncrement: ll))
  }

  // Convenience: one full step
  @discardableResult
  public func step(y: [Double]) -> (state: State, result: StepResult) {
    _ = predict()
    return update(y: y)
  }
}

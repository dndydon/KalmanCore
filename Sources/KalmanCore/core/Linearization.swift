import Foundation

/// Finite-difference linearization utilities shared by EKF and estimation code.
public enum Linearization {
  /// Approximate the state Jacobian F = ∂x_{k+1}/∂x_k around the provided state
  /// using a forward-Euler discretization of the deterministic dynamics.
  /// For small dt, F ≈ I + dt * ∂f/∂x where f is deterministicDynamics.
  public static func approximateStateJacobian<Model: StochasticDynamicalSystem>(
    model: Model,
    state x: [Double],
    parameters theta: [Double],
    dt: Double,
    epsilon: Double = 1e-7
  ) -> Matrix {
    let n = model.stateDimension
    var F = Matrix.identity(size: n)

    let f0 = model.deterministicDynamics(state: x, parameters: theta)
    for j in 0..<n {
      var xh = x
      xh[j] += epsilon
      let f1 = model.deterministicDynamics(state: xh, parameters: theta)
      for i in 0..<n {
        F[i, j] += (f1[i] - f0[i]) / epsilon * dt
      }
    }
    return F
  }

  /// Approximate the observation Jacobian H = ∂h/∂x around the provided state
  /// using finite differences of observationOperator.
  public static func approximateObservationJacobian(
    observationModel: ObservationModel,
    state x: [Double],
    epsilon: Double = 1e-7
  ) -> Matrix {
    let n = observationModel.stateDimension
    let m = observationModel.observationDimension
    var H = Matrix(rows: m, cols: n)

    let y0 = observationModel.observationOperator(state: x)
    for j in 0..<n {
      var xh = x
      xh[j] += epsilon
      let y1 = observationModel.observationOperator(state: xh)
      for i in 0..<m {
        H[i, j] = (y1[i] - y0[i]) / epsilon
      }
    }
    return H
  }
}

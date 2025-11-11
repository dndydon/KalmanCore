import Testing
@testable import KalmanCore

@Suite("Extended Kalman Filter (EKF)")
struct ExtendedKalmanFilterTests {

  @Test("Predict step updates state and covariance")
  func testPredict() {
    let model = Lorenz96Model.standard(stochasticType: .additive)
    let n = model.stateDimension
    let obs = IdentityObservationModel(dimension: n, noiseVariance: 1e-2)

    let x0 = model.typicalInitialState()
    let P0 = Matrix.identity(size: n) * 0.1
    let ekf = ExtendedKalmanFilter(model: model,
                                   observationModel: obs,
                                   initialState: x0,
                                   initialCovariance: P0,
                                   parameters: [0.3],
                                   dt: 0.01)
    let (st, _) = ekf.predict()
    #expect(st.x.count == n)
    #expect(st.P.rows == n && st.P.cols == n)
  }

  @Test("Update reduces covariance trace for informative observation")
  func testUpdateReducesCovariance() {
    let model = Lorenz96Model.standard(stochasticType: .additive)
    let n = model.stateDimension
    let obs = IdentityObservationModel(dimension: n, noiseVariance: 1e-2)

    let x0 = model.typicalInitialState()
    let P0 = Matrix.identity(size: n) * 0.2
    let ekf = ExtendedKalmanFilter(model: model,
                                   observationModel: obs,
                                   initialState: x0,
                                   initialCovariance: P0,
                                   parameters: [0.25],
                                   dt: 0.01)
    _ = ekf.predict()
    let prior = ekf.state.P.trace
    let y = obs.generateObservation(state: ekf.state.x)
    let (_, res) = ekf.update(y: y)
    #expect(ekf.state.P.trace < prior)
    #expect(matrixDeterminant(res.innovationCov) > 0.0)
  }

  @Test("Partial observation handled; innovation finite")
  func testPartialObservation() {
    let model = Lorenz96Model.standard(stochasticType: .additive)
    let n = model.stateDimension
    let indices = Array(stride(from: 0, to: n, by: 2))
    let obs = PartialObservationModel(stateDimension: n, observedIndices: indices, noiseVariance: 1e-2)

    let x0 = model.typicalInitialState()
    let P0 = Matrix.identity(size: n) * 0.2
    let ekf = ExtendedKalmanFilter(model: model,
                                   observationModel: obs,
                                   initialState: x0,
                                   initialCovariance: P0,
                                   parameters: [0.3],
                                   dt: 0.01)
    _ = ekf.predict()
    let y = obs.generateObservation(state: ekf.state.x)
    let (_, res) = ekf.update(y: y)
    #expect(res.innovation.count == obs.observationDimension)
    #expect(res.logLikelihoodIncrement.isFinite)
  }

  @Test("One-step step() convenience works")
  func testStepConvenience() {
    let model = Lorenz96Model.standard(stochasticType: .additive)
    let n = model.stateDimension
    let obs = IdentityObservationModel(dimension: n, noiseVariance: 1e-2)
    let x0 = model.typicalInitialState()
    let P0 = Matrix.identity(size: n) * 0.2

    let ekf = ExtendedKalmanFilter(model: model,
                                   observationModel: obs,
                                   initialState: x0,
                                   initialCovariance: P0,
                                   parameters: [0.3],
                                   dt: 0.01)
    let y = obs.generateObservation(state: x0)
    let (_, res) = ekf.step(y: y)
    #expect(res.innovationCov.rows == obs.observationDimension)
  }

  @Test("Finite-difference Jacobians have expected shape")
  func testJacobianShapes() {
    let model = Lorenz96Model.standard(stochasticType: .additive)
    let n = model.stateDimension
    let obs = IdentityObservationModel(dimension: n, noiseVariance: 1e-2)
    let x = model.typicalInitialState()

    let F = Linearization.approximateStateJacobian(model: model, state: x, parameters: [0.3], dt: 0.01)
    let H = Linearization.approximateObservationJacobian(observationModel: obs, state: x)

    #expect(F.rows == n && F.cols == n)
    #expect(H.rows == obs.observationDimension && H.cols == n)
  }
}

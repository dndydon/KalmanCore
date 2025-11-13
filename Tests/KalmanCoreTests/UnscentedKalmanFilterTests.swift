import Testing
@testable import KalmanCore

@Suite("Unscented Kalman Filter (UKF)")
struct UnscentedKalmanFilterTests {

  @Test("Predict step updates state and covariance (Identity obs later)")
  func testPredict() {
    let model = Lorenz96Model.standard(stochasticType: .additive)
    let n = model.stateDimension
    let obs = IdentityObservationModel(dimension: n, noiseVariance: 1e-2)

    let x0 = model.typicalInitialState()
    let P0 = Matrix.identity(size: n) * 0.2

    var ukf = UnscentedKalmanFilter(model: model,
                                    observationModel: obs,
                                    initialState: x0,
                                    initialCovariance: P0,
                                    parameters: [0.3],
                                    dt: 0.01)
    let st = ukf.predict()
    #expect(st.x.count == n)
    #expect(st.P.rows == n && st.P.cols == n)
  }

  @Test("Update reduces covariance trace with informative observation")
  func testUpdateReducesCovariance() {
    let model = Lorenz96Model.standard(stochasticType: .additive)
    let n = model.stateDimension
    let obs = IdentityObservationModel(dimension: n, noiseVariance: 1e-2)

    let x0 = model.typicalInitialState()
    let P0 = Matrix.identity(size: n) * 0.2

    var ukf = UnscentedKalmanFilter(model: model,
                                    observationModel: obs,
                                    initialState: x0,
                                    initialCovariance: P0,
                                    parameters: [0.25],
                                    dt: 0.01)
    _ = ukf.predict()
    let priorTrace = ukf.state.P.trace
    let y = obs.generateObservation(state: ukf.state.x)
    let (_, res) = ukf.update(y: y)
    #expect(ukf.state.P.trace < priorTrace)
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

    var ukf = UnscentedKalmanFilter(model: model,
                                    observationModel: obs,
                                    initialState: x0,
                                    initialCovariance: P0,
                                    parameters: [0.3],
                                    dt: 0.01)
    _ = ukf.predict()
    let y = obs.generateObservation(state: ukf.state.x)
    let (_, res) = ukf.update(y: y)
    #expect(res.innovation.count == obs.observationDimension)
    #expect(res.logLikelihoodIncrement.isFinite)
  }
}

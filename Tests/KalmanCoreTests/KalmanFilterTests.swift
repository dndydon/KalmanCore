import Testing
@testable import KalmanCore

@Suite("Kalman Filter (Linear)")
struct KalmanFilterTests {

  @Test("Initialization and predict step shape")
  func testInitializationAndPredict() {
    // Simple 2D system: identity dynamics
    let F = Matrix.identity(size: 2)
    let Q = Matrix.identity(size: 2) * 0.01
    let H = Matrix.identity(size: 2)
    let R = Matrix.identity(size: 2) * 0.1

    let x0 = [0.0, 0.0]
    let P0 = Matrix.identity(size: 2) * 1.0

    let kf = KalmanFilter(F: F, Q: Q, H: H, R: R, x0: x0, P0: P0)
    let st = kf.predict()

    #expect(st.x.count == 2)
    #expect(st.P.rows == 2 && st.P.cols == 2)
  }

  @Test("Update reduces covariance")
  func testUpdateReducesCovariance() {
    let F = Matrix.identity(size: 2)
    let Q = Matrix.identity(size: 2) * 0.01
    let H = Matrix.identity(size: 2)
    let R = Matrix.identity(size: 2) * 0.1

    let x0 = [0.0, 0.0]
    let P0 = Matrix.identity(size: 2) * 1.0

    let kf = KalmanFilter(F: F, Q: Q, H: H, R: R, x0: x0, P0: P0)
    _ = kf.predict()
    let priorTrace = kf.state.P.trace
    let (_, res) = kf.update(y: [1.0, -1.0])

    #expect(kf.state.P.trace < priorTrace)
    #expect(matrixDeterminant(res.innovationCov) > 0.0)
  }

  @Test("One full step returns finite log-likelihood")
  func testStepLogLikelihoodFinite() {
    let F = Matrix.identity(size: 1)
    let Q = Matrix.identity(size: 1) * 0.001
    let H = Matrix.identity(size: 1)
    let R = Matrix.identity(size: 1) * 0.05

    let kf = KalmanFilter(F: F, Q: Q, H: H, R: R, x0: [0.0], P0: Matrix.identity(size: 1))
    let (_, res) = kf.step(y: [0.2])
    #expect(res.logLikelihoodIncrement.isFinite)
  }

  @Test("Innovation matches y - Hx")
  func testInnovationComputation() {
    let F = Matrix.identity(size: 2)
    let Q = Matrix.identity(size: 2) * 0.01
    let H = Matrix.identity(size: 2)
    let R = Matrix.identity(size: 2) * 0.1
    let kf = KalmanFilter(F: F, Q: Q, H: H, R: R, x0: [0.1, -0.2], P0: Matrix.identity(size: 2))

    _ = kf.predict()
    let y = [0.0, 0.0]
    let yPred = H.multiply(vector: kf.state.x)
    let (_, res) = kf.update(y: y)
    #expect(res.innovation.count == 2)
    #expect(abs(res.innovation[0] - (y[0] - yPred[0])) < 1e-12)
    #expect(abs(res.innovation[1] - (y[1] - yPred[1])) < 1e-12)
  }

  @Test("Stable covariance over multiple steps")
  func testMultipleStepsStability() {
    let F = Matrix.identity(size: 2)
    let Q = Matrix.identity(size: 2) * 0.001
    let H = Matrix.identity(size: 2)
    let R = Matrix.identity(size: 2) * 0.1
    let kf = KalmanFilter(F: F, Q: Q, H: H, R: R, x0: [0.0, 0.0], P0: Matrix.identity(size: 2))

    var lastTrace = kf.state.P.trace
    for _ in 0..<5 {
      let y = [Double.random(in: -1...1), Double.random(in: -1...1)]
      _ = kf.step(y: y)
      #expect(kf.state.P.trace.isFinite)
      lastTrace = kf.state.P.trace
    }
    #expect(lastTrace > 0.0)
  }
}

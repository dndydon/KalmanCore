import Foundation

/*
 Unscented Kalman Filter (UKF)
 ----------------------------
 Nonlinear state-space model using the scaled unscented transform.
 x_{k+1} = f(x_k, θ, dt) + w_k
 y_k     = h(x_k) + v_k
 with w_k ~ N(0, Q_k) and v_k ~ N(0, R).
 
 Notes
 - This is the non–square-root variant. Covariances are formed explicitly with weights.
 - Process noise Q_k is built from σ(x, θ): Q_k = dt · σ σ^T (evaluated at current state).
 - Cholesky with jitter is used to ensure SPD when forming sigma points.
 
 References
 - Julier & Uhlmann (1997, 2004) – Unscented Transform
 - Wan & Van der Merwe (2000) – The Unscented Kalman Filter
 */

public final class UnscentedKalmanFilter<Model: StochasticDynamicalSystem> {
  public struct Config {
    public var alpha: Double = 1e-3
    public var beta: Double = 2.0         // optimal for Gaussian priors
    public var kappa: Double = 0.0
    public var jitter: Double = 1e-9      // added to diagonals before Cholesky
    public var verbose: Bool = false
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
  
  // Predict step via sigma points
  @discardableResult
  public func predict() -> State {
    let n = model.stateDimension
    
    // Compute sigma points from current (x, P)
    let (sigmaPoints, w_m, w_c) = computeSigmaPoints(x: state.x, P: stabilizedCovariance(state.P))
    
    // Propagate through dynamics
    var propagated: [[Double]] = Array(repeating: [Double](repeating: 0.0, count: n), count: sigmaPoints.count)
    for i in 0..<sigmaPoints.count {
      propagated[i] = model.transition(state: sigmaPoints[i], parameters: parameters, dt: dt)
    }
    
    // Predicted mean
    var xPred = [Double](repeating: 0.0, count: n)
    for i in 0..<propagated.count {
      for j in 0..<n { xPred[j] += w_m[i] * propagated[i][j] }
    }
    
    // Predicted covariance
    var PPred = Matrix(rows: n, cols: n)
    for i in 0..<propagated.count {
      let dx = vectorSubtract(propagated[i], xPred)
      let outer = outerProduct(dx, dx)
      PPred = PPred + (w_c[i] * outer)
    }
    
    // Add process noise Q (evaluated at current mean state)
    let sigma = model.stochasticParameterization(state: state.x, parameters: parameters)
    let Q = dt * (sigma * sigma.transposed)
    PPred = PPred + Q
    
    state = State(x: xPred, P: PPred)
    return state
  }
  
  // Update step via measurement sigma propagation
  @discardableResult
  public func update(y: [Double]) -> (state: State, result: StepResult) {
    let n = model.stateDimension
    let m = observationModel.observationDimension
    precondition(y.count == m, "Observation dimension mismatch")
    
    // Sigma points from predicted state
    let (sigmaPoints, w_m, w_c) = computeSigmaPoints(x: state.x, P: stabilizedCovariance(state.P))
    
    // Predicted observations
    var ySigma = Array(repeating: [Double](repeating: 0.0, count: m), count: sigmaPoints.count)
    for i in 0..<sigmaPoints.count {
      ySigma[i] = observationModel.observationOperator(state: sigmaPoints[i])
    }
    
    // Predicted observation mean
    var yPred = [Double](repeating: 0.0, count: m)
    for i in 0..<ySigma.count { for j in 0..<m { yPred[j] += w_m[i] * ySigma[i][j] } }
    
    // Innovation covariance S and cross-covariance P_xy
    var S = Matrix(rows: m, cols: m)
    var P_xy = Matrix(rows: n, cols: m)
    for i in 0..<ySigma.count {
      let dy = vectorSubtract(ySigma[i], yPred)
      let dx = vectorSubtract(sigmaPoints[i], state.x)
      S = S + (w_c[i] * outerProduct(dy, dy))
      P_xy = P_xy + (w_c[i] * outerProduct(dx, dy))
    }
    
    // Add measurement noise R
    let R = observationModel.observationNoiseCovariance
    S = S + R
    
    // Kalman gain
    let S_inv = matrixInverse(S)
    let K = P_xy * S_inv
    
    // Update state and covariance
    let innovation = vectorSubtract(y, yPred)
    let xUpd = vectorAdd(state.x, K.multiply(vector: innovation))
    let PUpd = state.P - K * S * K.transposed
    
    state = State(x: xUpd, P: PUpd)
    
    let ll = Likelihood.gaussianInnovationLogLikelihood(innovation: innovation, covariance: S)
    
    return (state, StepResult(innovation: innovation, innovationCov: S, gain: K, logLikelihoodIncrement: ll))
  }
  
  @discardableResult
  public func step(y: [Double]) -> (state: State, result: StepResult) {
    _ = predict()
    return update(y: y)
  }
  
  // MARK: - Unscented transform helpers
  private func computeSigmaPoints(x: [Double], P: Matrix) -> (points: [[Double]], w_m: [Double], w_c: [Double]) {
    let n = x.count
    let alpha = config.alpha
    let beta = config.beta
    let kappa = config.kappa
    
    let lambda = alpha * alpha * (Double(n) + kappa) - Double(n)
    let c = Double(n) + lambda
    
    // We need sqrt(c * P)
    var Ps = P
    for i in 0..<n { Ps[i, i] += config.jitter }
    guard let L = choleskyLowerSPD(Ps * c) else {
      // Fallback: add larger jitter and retry once
      var Pj = P
      for i in 0..<n { Pj[i, i] += max(1e-6, config.jitter * 100) }
      let L2 = choleskyLowerSPD(Pj * c)
      precondition(L2 != nil, "UKF: covariance not SPD even after jitter")
      return computeSigmaPointsFromCholesky(x: x, L: L2!, n: n, lambda: lambda, alpha: alpha, beta: beta, c: c)
    }
    return computeSigmaPointsFromCholesky(x: x, L: L, n: n, lambda: lambda, alpha: alpha, beta: beta, c: c)
  }
  
  private func computeSigmaPointsFromCholesky(x: [Double], L: Matrix, n: Int, lambda: Double, alpha: Double, beta: Double, c: Double) -> (points: [[Double]], w_m: [Double], w_c: [Double]) {
    var points: [[Double]] = Array(repeating: x, count: 2 * n + 1)
    
    // x0 = x
    // x_i = x + column_i(L), x_{i+n} = x - column_i(L)
    for i in 0..<n {
      var col = [Double](repeating: 0.0, count: n)
      for r in 0..<n { col[r] = L[r, i] }
      points[i + 1] = vectorAdd(x, col)
      points[i + 1 + n] = vectorSubtract(x, col)
    }
    
    // Weights
    var w_m = [Double](repeating: 1.0 / (2.0 * c), count: 2 * n + 1)
    var w_c = w_m
    w_m[0] = lambda / c
    w_c[0] = w_m[0] + (1.0 - alpha * alpha + beta)
    
    return (points, w_m, w_c)
  }
  
  private func stabilizedCovariance(_ P: Matrix) -> Matrix {
    var Pst = P
    let n = P.rows
    for i in 0..<n { Pst[i, i] += config.jitter }
    return Pst
  }
  
  private func outerProduct(_ a: [Double], _ b: [Double]) -> Matrix {
    let m = a.count
    let n = b.count
    var M = Matrix(rows: m, cols: n)
    for i in 0..<m { for j in 0..<n { M[i, j] = a[i] * b[j] } }
    return M
  }
  
  // Simple Cholesky (lower) for small SPD matrices
  private func choleskyLowerSPD(_ A: Matrix) -> Matrix? {
    let n = A.rows
    precondition(n == A.cols)
    var L = Matrix(rows: n, cols: n)
    for i in 0..<n {
      for j in 0...i {
        var sum = A[i, j]
        for k in 0..<j { sum -= L[i, k] * L[j, k] }
        if i == j {
          if sum <= 0 { return nil }
          L[i, j] = sqrt(max(sum, 1e-18))
        } else {
          L[i, j] = sum / L[j, j]
        }
      }
    }
    return L
  }
}

import Testing
@testable import KalmanCore

@Suite("EnKFEM Parameter Recovery (Additive Noise)")
struct EnKFEMParameterRecoveryTests {

  @Test("Parameter moves toward true additive noise using EnKFEM window")
  func testParameterMovesTowardTruth() {
    // Model and observation (identity for simplicity)
    let model = Lorenz96Model.standard(stochasticType: .additive)
    let n = model.stateDimension
    let obsModel = IdentityObservationModel(dimension: n, noiseVariance: 1e-2)

    // True parameter and synthetic data
    let trueTheta: [Double] = [0.35]
    let dt = 0.01
    let steps = 20

    var xTruth = model.typicalInitialState()
    var observations: [[Double]] = []
    observations.reserveCapacity(steps)
    for _ in 0..<steps {
      xTruth = model.transition(state: xTruth, parameters: trueTheta, dt: dt)
      observations.append(obsModel.generateObservation(state: xTruth))
    }

    // EnKF config (small ensemble, deterministic obs)
    let enkfConfig = EnKFConfig(
      ensembleSize: 10,
      inflation: 1.05,
      additiveInflation: nil,
      parameterEvolution: .constant,
      usePerturbedObservations: false,
      localizationRadius: 0.9, // light taper
      verbose: false
    )
    let enkf = EnsembleKalmanFilter(model: model, observationModel: obsModel, config: enkfConfig)

    // Priors
    let x0 = model.typicalInitialState()
    let P0 = Matrix.identity(size: n) * 0.2
    let theta0: [Double] = [0.15]
    let Ptheta0 = Matrix.diagonal([0.05])
    var Z0 = enkf.initializeEnsemble(x0: x0, P0: P0, theta0: theta0, Ptheta0: Ptheta0)

    // EnKF-EM run over a window shorter than the data
    var emConfig = EnKFEMConfig()
    emConfig.window = 15
    emConfig.maxEMItersPerWindow = 1
    emConfig.verbose = false

    let enkfem = EnKFEM(model: model, observationModel: obsModel, enkf: enkf, config: emConfig)
    let (newParams, _, _) = enkfem.runWindow(
      observations: observations,
      dt: dt,
      initialParameters: theta0,
      initialEnsemble: Z0
    )

    // Expect movement toward true parameter (not necessarily full recovery in one window)
    let d0 = abs(theta0[0] - trueTheta[0])
    let d1 = abs(newParams[0] - trueTheta[0])

    #expect(newParams.count == 1)
    #expect(newParams[0].isFinite)
    #expect(d1 <= d0) // non-increasing error
  }
}

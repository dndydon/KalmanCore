import Testing
@testable import KalmanCore

@Suite("EnKFEM Basics")
struct EnKFEMBasicTests {

  @Test("runWindow executes over a short window and returns sane outputs")
  func testRunWindowBasics() {
    // Model and observation
    let model = Lorenz96Model.standard(stochasticType: .additive)
    let n = model.stateDimension
    let obsModel = IdentityObservationModel(dimension: n, noiseVariance: 1e-2)

    // EnKF configuration: small ensemble, deterministic observations (no perturbations)
    let enkfConfig = EnKFConfig(
      ensembleSize: 8,
      inflation: 1.0,
      additiveInflation: nil,
      parameterEvolution: .constant,
      usePerturbedObservations: false,
      localizationRadius: nil,
      verbose: false
    )
    let enkf = EnsembleKalmanFilter(model: model, observationModel: obsModel, config: enkfConfig)

    // Initial priors
    let x0 = model.typicalInitialState()
    let P0 = Matrix.identity(size: n) * 0.1
    let theta0: [Double] = [0.2]
    let Ptheta0 = Matrix.diagonal([0.05])

    // Build initial augmented ensemble
    let Z = enkf.initializeEnsemble(x0: x0, P0: P0, theta0: theta0, Ptheta0: Ptheta0)

    // Generate a short sequence of synthetic observations from truth
    let trueTheta: [Double] = [0.3]
    let dt = 0.01
    let steps = 12

    var xTruth = x0
    var observations: [[Double]] = []
    observations.reserveCapacity(steps)
    for _ in 0..<steps {
      xTruth = model.transition(state: xTruth, parameters: trueTheta, dt: dt)
      let y = obsModel.generateObservation(state: xTruth)
      observations.append(y)
    }

    // EnKF-EM wrapper with a small window
    var emConfig = EnKFEMConfig()
    emConfig.window = 10
    emConfig.maxEMItersPerWindow = 1
    emConfig.verbose = false

    let enkfem = EnKFEM(model: model, observationModel: obsModel, enkf: enkf, config: emConfig)

    let result = enkfem.runWindow(
      observations: observations,
      dt: dt,
      initialParameters: theta0,
      initialEnsemble: Z
    )
    
    // Basic sanity checks on outputs
    #expect(result.newParameters.count == model.parameterDimension)
    #expect(result.newEnsemble.ensembleSize == enkfConfig.ensembleSize)
    #expect(result.newEnsemble.stateDimension == (n + model.parameterDimension))

    // Parameters are bounded and finite
    for p in result.newParameters {
      #expect(p.isFinite)
      #expect(p >= emConfig.parameterBounds.min)
      #expect(p <= emConfig.parameterBounds.max)
    }
  }
}

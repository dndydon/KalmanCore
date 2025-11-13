import Testing
@testable import KalmanCore

@Suite("EnKF Parameter Augmentation")
struct EnKFParameterAugmentationTests {

  @Test("analysis updates parameters in augmented EnKF (deterministic obs)")
  func testAnalysisUpdatesParameters() {
    // Model and observation
    let model = Lorenz96Model.standard(stochasticType: .additive)
    let mDim = 40
    let obsModel = IdentityObservationModel(dimension: mDim, noiseVariance: 1e-2)

    // EnKF configuration (deterministic observation, no perturbed obs)
    let config = EnKFConfig(
      ensembleSize: 5,
      inflation: 1.0,
      additiveInflation: nil,
      parameterEvolution: .constant,
      useSquareRootAnalysis: false,
      usePerturbedObservations: false,
      localization: LocalizationConfig(method: .none),
      verbose: false
    )
    let enkf = EnsembleKalmanFilter(model: model, observationModel: obsModel, config: config)

    // True state and observation
    let xTrue = model.typicalInitialState()
    let y = obsModel.generateObservation(state: xTrue)

    // Deterministic augmented ensemble with correlated x-θ deviations
    let N = config.ensembleSize
    var members: [[Double]] = []
    members.reserveCapacity(N)
    for i in 0..<N {
      let shift = (Double(i) - Double(N-1)/2.0) * 0.005
      var x = xTrue
      for j in 0..<mDim { x[j] += shift }
      let theta = [0.10 + 0.02 * shift * Double(N)] // small correlation with x
      var z = x
      z.append(contentsOf: theta)
      members.append(z)
    }
    let Z0 = Ensemble(members: members)

    // Run analysis (without forecast, to isolate analysis math)
    let Z1 = enkf.analyze(ensemble: Z0, observation: y)

    // Compare mean parameter before/after
    func meanTheta(_ Z: Ensemble) -> Double {
      var s = 0.0
      for member in Z.members { s += member[mDim] }
      return s / Double(Z.ensembleSize)
    }

    let thetaBefore = meanTheta(Z0)
    let thetaAfter = meanTheta(Z1)

    // Expect some change (non-zero Kalman gain effect)
    #expect(abs(thetaAfter - thetaBefore) > 1e-12)
  }
}

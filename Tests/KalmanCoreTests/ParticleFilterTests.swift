import Testing
@testable import KalmanCore

@Suite("Particle Filter (SIR)")
struct ParticleFilterTests {

  @Test("Initialization draws particles and weights normalize")
  func testInit() {
    let model = Lorenz96Model.standard(stochasticType: .additive)
    let n = model.stateDimension
    let obs = IdentityObservationModel(dimension: n, noiseVariance: 1e-2)

    let x0 = model.typicalInitialState()
    let P0 = Matrix.identity(size: n) * 0.2

    let pf = ParticleFilter(model: model,
                            observationModel: obs,
                            x0: x0,
                            P0: P0,
                            parameters: [0.3],
                            dt: 0.01,
                            config: .init(numParticles: 50))

    #expect(pf.state.particles.count == 50)
    let sumw = pf.state.weights.reduce(0, +)
    #expect(abs(sumw - 1.0) < 1e-12)
  }

  @Test("Update produces finite weights and ESS; resamples when threshold high")
  func testUpdateAndResample() {
    let model = Lorenz96Model.standard(stochasticType: .additive)
    let n = model.stateDimension
    let obs = IdentityObservationModel(dimension: n, noiseVariance: 1e-2)

    let pf = ParticleFilter(model: model,
                            observationModel: obs,
                            x0: model.typicalInitialState(),
                            P0: Matrix.identity(size: n) * 0.2,
                            parameters: [0.3],
                            dt: 0.01,
                            config: .init(numParticles: 60, resamplingThreshold: 0.99))

    _ = pf.predict()
    let y = obs.generateObservation(state: pf.state.particles[0])
    let (_, res) = pf.update(y: y)

    let sumw = pf.state.weights.reduce(0, +)
    #expect(abs(sumw - 1.0) < 1e-9)
    #expect(res.ess > 0)
  }
}

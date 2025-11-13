import Foundation
import KalmanCore

@main
struct EnKFBenchMain {
  static func main() {
    // Config from env
    let n = Int(ProcessInfo.processInfo.environment["ENKF_BENCH_N"] ?? "40") ?? 40
    let steps = Int(ProcessInfo.processInfo.environment["ENKF_BENCH_STEPS"] ?? "50") ?? 50
    let ensemblesCSV = ProcessInfo.processInfo.environment["ENKF_BENCH_ENSEMBLES"] ?? "10,20,40"
    let ensembles = ensemblesCSV.split(separator: ",").compactMap { Int($0.trimmingCharacters(in: .whitespaces)) }
    let localize = (ProcessInfo.processInfo.environment["ENKF_BENCH_LOCALIZE"] ?? "false").lowercased() == "true"
    let sqrtMode = (ProcessInfo.processInfo.environment["ENKF_BENCH_SQRT"] ?? "true").lowercased() == "true"

    print("EnKF Bench: n=\(n), steps=\(steps), ensembles=\(ensembles), localize=\(localize), sqrt=\(sqrtMode)")

    // Model/obs
    let model = Lorenz96Model(dimension: n, forcing: 8.0, stochasticType: .additive)
    let obsIdx = Array(stride(from: 0, to: n, by: 2))
    let obs = PartialObservationModel(stateDimension: n, observedIndices: obsIdx, noiseVariance: 0.5)

    // Helpers
    func time(_ block: () -> Void) -> Double {
      let t0 = CFAbsoluteTimeGetCurrent()
      block()
      return (CFAbsoluteTimeGetCurrent() - t0)
    }

    for N in ensembles {
      // Build configs
      let locCfg = localize ? LocalizationConfig(method: .schurGaspariCohn1D(lengthScale: max(4.0, Double(n)/8.0), periodic: true))
      : LocalizationConfig(method: .none)
      let cfg = EnKFConfig(
        ensembleSize: N,
        inflation: 1.03,
        additiveInflation: nil,
        parameterEvolution: .constant,
        useSquareRootAnalysis: sqrtMode,
        usePerturbedObservations: !sqrtMode, // if not sqrt, use stochastic
        localization: locCfg,
        verbose: false
      )
      let enkf = EnsembleKalmanFilter(model: model, observationModel: obs, config: cfg)

      // Initial ensemble
      let x0 = model.typicalInitialState()
      let P0 = Matrix.identity(size: n) * 0.2
      let theta0: [Double] = [0.3]
      let Ptheta0 = Matrix.diagonal([0.05])
      var Z = enkf.initializeEnsemble(x0: x0, P0: P0, theta0: theta0, Ptheta0: Ptheta0)

      // Synthetic truth + obs stream
      var xTruth = x0
      var Y: [[Double]] = []
      Y.reserveCapacity(steps)
      for _ in 0..<steps {
        xTruth = model.transition(state: xTruth, parameters: [0.35], dt: 0.01)
        Y.append(obs.generateObservation(state: xTruth))
      }

      // Run
      let elapsed = time {
        for t in 0..<steps {
          Z = enkf.forecast(ensemble: Z, dt: 0.01)
          Z = enkf.analyze(ensemble: Z, observation: Y[t])
        }
      }

      // Simple output: seconds per step and per member
      let secPerStep = elapsed / Double(steps)
      let secPerMember = secPerStep / Double(N)
      print(String(format: "N=%3d | %.4f s total | %.6f s/step | %.8f s/step/member | sqrt=%@ | loc=%@",
                   N, elapsed, secPerStep, secPerMember, String(sqrtMode), String(localize)))
    }
  }
}

import Foundation

/// Demonstration of Section 2.1: Stochastic Parameterization Framework
/// Using the Lorenz96 model as a test case
public class Lorenz96Demo {
  
  /// Demonstrate basic stochastic system simulation
  public static func demonstrateBasicSimulation() {
    print("\n=== Section 2.1: Basic Stochastic System Simulation ===")
    print("Simulating Lorenz96 model with stochastic parameterization\n")
    
    // Create Lorenz96 model (40 variables, F=8.0)
    let model = Lorenz96Model.standard(stochasticType: .additive)
    print("Model: Lorenz96 (dimension=\(model.stateDimension), F=\(model.forcing))")
    
    // Initial state
    let initialState = model.typicalInitialState()
    print("Initial state (first 5): \(initialState.prefix(5).map { String(format: "%.3f", $0) })")
    
    // Stochastic parameter (noise level)
    let parameters = [0.5]
    print("Stochastic parameter θ₀ = \(parameters[0])")
    
    // Simulate trajectory
    let dt = 0.01
    let steps = 1000
    print("\nSimulating trajectory: dt=\(dt), steps=\(steps)")
    
    let trajectory = model.simulateTrajectory(
      initialState: initialState,
      parameters: parameters,
      dt: dt,
      steps: steps
    )
    
    let finalState = trajectory.last!
    print("Final state (first 5): \(finalState.prefix(5).map { String(format: "%.3f", $0) })")
    
    // Compute statistics
    let mean = Lorenz96Model.climatologicalMean(trajectory: trajectory)
    let meanValue = mean.reduce(0, +) / Double(mean.count)
    print("Mean state value: \(String(format: "%.3f", meanValue))")
    
    print("✓ Basic simulation complete\n")
  }
  
  /// Demonstrate ensemble forecasting
  public static func demonstrateEnsembleForecasting() {
    print("\n=== Section 2.1: Ensemble Forecasting ===")
    print("Generating ensemble of trajectories\n")
    
    let model = Lorenz96Model.standard(stochasticType: .additive)
    let parameters = [0.5]
    
    // Create initial ensemble
    let ensembleSize = 50
    let initialState = model.typicalInitialState()
    let initialCovariance = Matrix.identity(size: model.stateDimension) * 0.1
    
    var ensemble = Ensemble(
      mean: initialState,
      covariance: initialCovariance,
      ensembleSize: ensembleSize
    )
    
    print("Initial ensemble:")
    print("  Size: \(ensemble.ensembleSize)")
    print("  Dimension: \(ensemble.stateDimension)")
    print("  Spread: \(String(format: "%.3f", ensemble.spread))")
    
    // Forecast ensemble forward in time
    let dt = 0.01
    let forecastSteps = 100
    
    for _ in 0..<forecastSteps {
      ensemble = model.forecastEnsemble(ensemble: ensemble, parameters: parameters, dt: dt)
    }
    
    print("\nForecast ensemble (after \(forecastSteps) steps):")
    print("  Spread: \(String(format: "%.3f", ensemble.spread))")
    let forecastMean = ensemble.mean
    print("  Mean (first 5): \(forecastMean.prefix(5).map { String(format: "%.3f", $0) })")
    
    print("✓ Ensemble forecasting complete\n")
  }
  
  /// Demonstrate observation generation
  public static func demonstrateObservationModel() {
    print("\n=== Section 2.1: Observation Model ===")
    print("Generating synthetic observations\n")
    
    let model = Lorenz96Model.standard(stochasticType: .additive)
    let parameters = [0.5]
    
    // Generate true state trajectory
    let trueInitialState = model.typicalInitialState()
    let dt = 0.01
    let steps = 100
    
    let trajectory = model.simulateTrajectory(
      initialState: trueInitialState,
      parameters: parameters,
      dt: dt,
      steps: steps
    )
    
    // Create partial observation model (observe every other variable)
    let observedIndices = stride(from: 0, to: model.stateDimension, by: 2).map { $0 }
    let observationNoise = 0.5
    
    let obsModel = PartialObservationModel(
      stateDimension: model.stateDimension,
      observedIndices: observedIndices,
      noiseVariance: observationNoise
    )
    
    print("Observation model:")
    print("  State dimension: \(obsModel.stateDimension)")
    print("  Observation dimension: \(obsModel.observationDimension)")
    print("  Observed indices: \(observedIndices.prefix(5))...")
    print("  Observation noise: \(observationNoise)")
    
    // Generate observations at regular intervals
    let observationInterval = 10
    var observations = [[Double]]()
    
    for (i, state) in trajectory.enumerated() {
      if i % observationInterval == 0 {
        let obs = obsModel.generateObservation(state: state)
        observations.append(obs)
      }
    }
    
    print("\nGenerated \(observations.count) observations")
    print("First observation (first 5): \(observations[0].prefix(5).map { String(format: "%.3f", $0) })")
    
    print("✓ Observation generation complete\n")
  }
  
  /// Demonstrate different stochastic parameterization types
  public static func compareStochasticTypes() {
    print("\n=== Section 2.1: Comparing Stochastic Parameterization Types ===")
    
    let types: [(Lorenz96Model.StochasticType, String)] = [
      (.additive, "Additive (σ = θ₀·I)"),
      (.stateDependent, "State-dependent (σᵢ = θ₀·|xᵢ|)")
    ]
    
    for (stochasticType, description) in types {
      print("\n\(description)")
      
      let model = Lorenz96Model(dimension: 40, forcing: 8.0, stochasticType: stochasticType)
      let initialState = model.typicalInitialState()
      let parameters = [0.3]
      
      let trajectory = model.simulateTrajectory(
        initialState: initialState,
        parameters: parameters,
        dt: 0.01,
        steps: 1000
      )
      
      let mean = Lorenz96Model.climatologicalMean(trajectory: trajectory)
      
      // Compute average variance
      var totalVariance = 0.0
      for state in trajectory {
        var stateVariance = 0.0
        for (i, val) in state.enumerated() {
          let diff = val - mean[i]
          stateVariance += diff * diff
        }
        totalVariance += stateVariance / Double(state.count)
      }
      let variance = totalVariance / Double(trajectory.count)
      
      print("  Mean climatology: \(String(format: "%.3f", mean.reduce(0, +) / Double(mean.count)))")
      print("  Average variance: \(String(format: "%.3f", variance))")
    }
    
    print("\n✓ Comparison complete\n")
  }
  
  /// Run all demonstrations
  public static func runAll() {
    print("\n" + String(repeating: "=", count: 60))
    print("Section 2.1: Stochastic Parameterization Identification")
    print("Pulido et al. (2018) - Tellus A")
    print(String(repeating: "=", count: 60))
    
    demonstrateBasicSimulation()
    demonstrateEnsembleForecasting()
    demonstrateObservationModel()
    compareStochasticTypes()
    
    print(String(repeating: "=", count: 60))
    print("All demonstrations complete!")
    print(String(repeating: "=", count: 60) + "\n")
  }
}

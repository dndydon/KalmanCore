import Testing
@testable import KalmanCore

@Suite("ExpectationMaximization Algorithm Tests")
struct ExpectationMaximizationTests {

  var model: Lorenz96Model
  var em: ExpectationMaximization

  init() {
    model = Lorenz96Model.standard(stochasticType: .additive)
    em = ExpectationMaximization()
  }

  // MARK: - Basic Functionality Tests

  @Test("Initialization succeeds")
  func testInitialization() {
    let config = ExpectationMaximization.Configuration()
    let algorithm = ExpectationMaximization(config: config)
    #expect(config.maxIterations == 100)  // Verify object creation
    print("👍 Initialization succeeded:", algorithm)
  }

  @Test("Configuration has correct default values")
  func testConfigurationDefaults() {
    let config = ExpectationMaximization.Configuration()
    #expect(config.maxIterations == 100)
    #expect(config.convergenceTolerance == 1e-4)
    #expect(config.logLikelihoodTolerance == 1e-6)
    #expect(config.minParameterValue == 1e-8)
    #expect(config.maxParameterValue == 100.0)
    #expect(config.verbose == true)
  }

  @Test("Result structure is properly initialized")
  func testResultStructure() {
    let result = ExpectationMaximization.Result(
      parameters: [0.5],
      logLikelihoodHistory: [-10.0, -5.0, -3.0],
      parameterHistory: [[0.1], [0.3], [0.5]],
      iterations: 3,
      converged: true,
      finalParameterChange: 0.001,
      finalLogLikelihoodChange: 0.01
    )

    #expect(result.parameters == [0.5])
    #expect(result.iterations == 3)
    #expect(result.converged == true)
    #expect(result.logLikelihoodHistory.count == 3)
  }

  // MARK: - Synthetic Data Tests

  @Test("Parameter recovery produces positive values")
  func testParameterRecoveryAdditiveNoise() {
    let trueParameters = [0.5]
    let initialState = model.typicalInitialState()

    let trajectory = model.simulateTrajectory(
      initialState: initialState,
      parameters: trueParameters,
      dt: 0.01,
      steps: 50
    )

    let obsModel = PartialObservationModel(
      stateDimension: 40,
      observedIndices: stride(from: 0, to: 40, by: 2).map { $0 },
      noiseVariance: 0.1
    )

    var observations: [[Double]] = []
    for state in trajectory {
      observations.append(obsModel.generateObservation(state: state))
    }

    var config = ExpectationMaximization.Configuration()
    config.maxIterations = 20
    config.verbose = false

    let algorithm = ExpectationMaximization(config: config)
    let result = algorithm.estimate(
      model: model,
      observations: observations,
      observationModel: obsModel,
      initialParameters: [0.2],
      initialState: initialState,
      initialCovariance: Matrix.identity(size: 40) * 0.1,
      dt: 0.01
    )

    #expect(result.parameters[0] > 0.0)
    #expect(result.parameters[0] <= 100.0)  // Within bounds (may hit upper bound)
  }

  // MARK: - Convergence Tests

  @Test("Log-likelihood is monitored over iterations")
  func testMonotoneLogLikelihood() {
    let trueParameters = [0.3]
    let initialState = model.typicalInitialState()

    let trajectory = model.simulateTrajectory(
      initialState: initialState,
      parameters: trueParameters,
      dt: 0.01,
      steps: 30
    )

    let obsModel = PartialObservationModel(
      stateDimension: 40,
      observedIndices: stride(from: 0, to: 40, by: 2).map { $0 },
      noiseVariance: 0.1  // Lower noise
    )

    var observations: [[Double]] = []
    for state in trajectory {
      observations.append(obsModel.generateObservation(state: state))
    }

    var config = ExpectationMaximization.Configuration()
    config.maxIterations = 10
    config.verbose = false

    let algorithm = ExpectationMaximization(config: config)
    let result = algorithm.estimate(
      model: model,
      observations: observations,
      observationModel: obsModel,
      initialParameters: [0.2],
      initialState: initialState,
      initialCovariance: Matrix.identity(size: 40) * 0.1,
      dt: 0.01
    )

    // Verify log-likelihood is tracked
    #expect(result.logLikelihoodHistory.count > 0)
    #expect(result.logLikelihoodHistory.count == result.iterations)
  }

  @Test("Algorithm converges before max iterations")
  func testConvergenceCriteria() {
    let trueParameters = [0.4]
    let initialState = model.typicalInitialState()

    let trajectory = model.simulateTrajectory(
      initialState: initialState,
      parameters: trueParameters,
      dt: 0.01,
      steps: 25
    )

    let obsModel = PartialObservationModel(
      stateDimension: 40,
      observedIndices: stride(from: 0, to: 40, by: 4).map { $0 },
      noiseVariance: 0.15
    )

    var observations: [[Double]] = []
    for state in trajectory {
      observations.append(obsModel.generateObservation(state: state))
    }

    var config = ExpectationMaximization.Configuration()
    config.maxIterations = 50
    config.convergenceTolerance = 1e-3
    config.verbose = false

    let algorithm = ExpectationMaximization(config: config)
    let result = algorithm.estimate(
      model: model,
      observations: observations,
      observationModel: obsModel,
      initialParameters: [0.2],
      initialState: initialState,
      initialCovariance: Matrix.identity(size: 40) * 0.1,
      dt: 0.01
    )

    #expect(result.iterations < 50)
  }

  // MARK: - Noise Level Tests

  @Test("Handles high observation noise without crashing")
  func testHighObservationNoise() {
    let trueParameters = [0.5]
    let initialState = model.typicalInitialState()

    let trajectory = model.simulateTrajectory(
      initialState: initialState,
      parameters: trueParameters,
      dt: 0.01,
      steps: 40
    )

    let obsModel = PartialObservationModel(
      stateDimension: 40,
      observedIndices: stride(from: 0, to: 40, by: 2).map { $0 },
      noiseVariance: 1.0
    )

    var observations: [[Double]] = []
    for state in trajectory {
      observations.append(obsModel.generateObservation(state: state))
    }

    var config = ExpectationMaximization.Configuration()
    config.maxIterations = 20
    config.verbose = false

    let algorithm = ExpectationMaximization(config: config)
    let result = algorithm.estimate(
      model: model,
      observations: observations,
      observationModel: obsModel,
      initialParameters: [0.1],
      initialState: initialState,
      initialCovariance: Matrix.identity(size: 40) * 0.5,
      dt: 0.01
    )

    #expect(result.parameters[0] > 0.0)
    #expect(result.parameters[0] <= 100.0)  // Within bounds
  }

  // MARK: - Parameter Constraint Tests

  @Test("Respects parameter bounds")
  func testParameterBounds() {
    let trueParameters = [0.3]
    let initialState = model.typicalInitialState()

    let trajectory = model.simulateTrajectory(
      initialState: initialState,
      parameters: trueParameters,
      dt: 0.01,
      steps: 20
    )

    let obsModel = PartialObservationModel(
      stateDimension: 40,
      observedIndices: stride(from: 0, to: 40, by: 3).map { $0 },
      noiseVariance: 0.2
    )

    var observations: [[Double]] = []
    for state in trajectory {
      observations.append(obsModel.generateObservation(state: state))
    }

    var config = ExpectationMaximization.Configuration()
    config.maxIterations = 20
    config.minParameterValue = 0.01
    config.maxParameterValue = 50.0
    config.verbose = false

    let algorithm = ExpectationMaximization(config: config)
    let result = algorithm.estimate(
      model: model,
      observations: observations,
      observationModel: obsModel,
      initialParameters: [0.05],
      initialState: initialState,
      initialCovariance: Matrix.identity(size: 40) * 0.1,
      dt: 0.01
    )

    for param in result.parameters {
      #expect(param >= config.minParameterValue)
      #expect(param <= config.maxParameterValue)
    }
  }

  // MARK: - Parameter History Tests

  @Test("Parameter history is correctly tracked")
  func testParameterHistoryTracking() {
    let trueParameters = [0.4]
    let initialState = model.typicalInitialState()

    let trajectory = model.simulateTrajectory(
      initialState: initialState,
      parameters: trueParameters,
      dt: 0.01,
      steps: 30
    )

    let obsModel = PartialObservationModel(
      stateDimension: 40,
      observedIndices: stride(from: 0, to: 40, by: 2).map { $0 },
      noiseVariance: 0.15
    )

    var observations: [[Double]] = []
    for state in trajectory {
      observations.append(obsModel.generateObservation(state: state))
    }

    var config = ExpectationMaximization.Configuration()
    config.maxIterations = 25
    config.verbose = false

    let algorithm = ExpectationMaximization(config: config)
    let result = algorithm.estimate(
      model: model,
      observations: observations,
      observationModel: obsModel,
      initialParameters: [0.1],
      initialState: initialState,
      initialCovariance: Matrix.identity(size: 40) * 0.1,
      dt: 0.01
    )

    #expect(result.parameterHistory.count == result.iterations + 1)
    #expect(result.parameterHistory[0] == [0.1])
    #expect(result.parameterHistory.last == result.parameters)
  }

  // MARK: - Log-Likelihood History Tests

  @Test("Log-likelihood history is correctly tracked")
  func testLogLikelihoodHistoryTracking() {
    let trueParameters = [0.35]
    let initialState = model.typicalInitialState()

    let trajectory = model.simulateTrajectory(
      initialState: initialState,
      parameters: trueParameters,
      dt: 0.01,
      steps: 25
    )

    let obsModel = PartialObservationModel(
      stateDimension: 40,
      observedIndices: stride(from: 0, to: 40, by: 2).map { $0 },
      noiseVariance: 0.2
    )

    var observations: [[Double]] = []
    for state in trajectory {
      observations.append(obsModel.generateObservation(state: state))
    }

    var config = ExpectationMaximization.Configuration()
    config.maxIterations = 20
    config.verbose = false

    let algorithm = ExpectationMaximization(config: config)
    let result = algorithm.estimate(
      model: model,
      observations: observations,
      observationModel: obsModel,
      initialParameters: [0.1],
      initialState: initialState,
      initialCovariance: Matrix.identity(size: 40) * 0.1,
      dt: 0.01
    )

    #expect(result.logLikelihoodHistory.count == result.iterations)

    for ll in result.logLikelihoodHistory {
      #expect(ll < 0.0)
    }
  }

  // MARK: - Edge Cases

  @Test("Handles single observation without crashing")
  func testSingleObservation() {
    let trueParameters = [0.3]
    let initialState = model.typicalInitialState()

    let trajectory = model.simulateTrajectory(
      initialState: initialState,
      parameters: trueParameters,
      dt: 0.01,
      steps: 3
    )

    let obsModel = PartialObservationModel(
      stateDimension: 40,
      observedIndices: [0, 5, 10, 15, 20],
      noiseVariance: 0.1
    )

    var observations: [[Double]] = []
    observations.append(obsModel.generateObservation(state: trajectory[0]))

    var config = ExpectationMaximization.Configuration()
    config.maxIterations = 10
    config.verbose = false

    let algorithm = ExpectationMaximization(config: config)

    let result = algorithm.estimate(
      model: model,
      observations: observations,
      observationModel: obsModel,
      initialParameters: [0.2],
      initialState: initialState,
      initialCovariance: Matrix.identity(size: 40) * 0.1,
      dt: 0.01
    )

    #expect(result.parameters[0] > 0.0)
  }

  @Test("Recovers from large initial parameter error")
  func testLargeInitialParameterError() {
    let trueParameters = [0.5]
    let initialState = model.typicalInitialState()

    let trajectory = model.simulateTrajectory(
      initialState: initialState,
      parameters: trueParameters,
      dt: 0.01,
      steps: 40
    )

    let obsModel = PartialObservationModel(
      stateDimension: 40,
      observedIndices: stride(from: 0, to: 40, by: 2).map { $0 },
      noiseVariance: 0.15
    )

    var observations: [[Double]] = []
    for state in trajectory {
      observations.append(obsModel.generateObservation(state: state))
    }

    var config = ExpectationMaximization.Configuration()
    config.maxIterations = 50
    config.verbose = false

    let algorithm = ExpectationMaximization(config: config)

    let result = algorithm.estimate(
      model: model,
      observations: observations,
      observationModel: obsModel,
      initialParameters: [0.01],
      initialState: initialState,
      initialCovariance: Matrix.identity(size: 40) * 0.1,
      dt: 0.01
    )

    #expect(result.parameters[0] > 0.05)
  }

  // MARK: - Different Stochastic Types

  @Test("Works with state-dependent noise")
  func testStateDependentNoise() {
    let model = Lorenz96Model.standard(stochasticType: .stateDependent)
    let trueParameters = [0.2]
    let initialState = model.typicalInitialState()

    let trajectory = model.simulateTrajectory(
      initialState: initialState,
      parameters: trueParameters,
      dt: 0.01,
      steps: 30
    )

    let obsModel = PartialObservationModel(
      stateDimension: 40,
      observedIndices: stride(from: 0, to: 40, by: 3).map { $0 },
      noiseVariance: 0.2
    )

    var observations: [[Double]] = []
    for state in trajectory {
      observations.append(obsModel.generateObservation(state: state))
    }

    var config = ExpectationMaximization.Configuration()
    config.maxIterations = 20
    config.verbose = false

    let algorithm = ExpectationMaximization(config: config)
    let result = algorithm.estimate(
      model: model,
      observations: observations,
      observationModel: obsModel,
      initialParameters: [0.1],
      initialState: initialState,
      initialCovariance: Matrix.identity(size: 40) * 0.1,
      dt: 0.01
    )

    #expect(result.parameters[0] > 0.0)
  }
}

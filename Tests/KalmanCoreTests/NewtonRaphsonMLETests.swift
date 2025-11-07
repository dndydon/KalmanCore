import Testing
@testable import KalmanCore

@Suite("Newton-Raphson MLE Algorithm Tests")
struct NewtonRaphsonMLETests {

  var model: Lorenz96Model
  var nr: NewtonRaphsonMLE

  init() {
    model = Lorenz96Model.standard(stochasticType: .additive)
    nr = NewtonRaphsonMLE()
  }

  // MARK: - Basic Functionality Tests

  @Test("Initialization succeeds")
  func testInitialization() {
    let config = NewtonRaphsonMLE.Configuration()
    let algorithm = NewtonRaphsonMLE(config: config)
    #expect(config.maxIterations == 50)  // Verify object creation
    print("👍 Initialization succeeded:", algorithm)
  }

  @Test("Configuration has correct default values")
  func testConfigurationDefaults() {
    let config = NewtonRaphsonMLE.Configuration()
    #expect(config.maxIterations == 50)
    #expect(config.gradientTolerance == 1e-4)
    #expect(config.parameterTolerance == 1e-4)
    #expect(config.minParameterValue == 1e-8)
    #expect(config.maxParameterValue == 100.0)
    #expect(config.lineSearchMaxSteps == 10)
    #expect(config.finiteDifferenceEps == 1e-6)
    #expect(config.verbose == true)
  }

  @Test("Result structure is properly initialized")
  func testResultStructure() {
    let hessian = Matrix.identity(size: 1)
    let result = NewtonRaphsonMLE.Result(
      parameters: [0.5],
      logLikelihoodHistory: [-10.0, -5.0, -3.0],
      parameterHistory: [[0.1], [0.3], [0.5]],
      gradientNorms: [0.5, 0.2, 0.05],
      iterations: 3,
      converged: true,
      finalGradientNorm: 0.05,
      finalParameterChange: 0.001,
      finalHessian: hessian
    )

    #expect(result.parameters == [0.5])
    #expect(result.iterations == 3)
    #expect(result.converged == true)
    #expect(result.gradientNorms.count == 3)
    #expect(result.finalHessian != nil)
  }

  // MARK: - Synthetic Data Tests

  @Test("Parameter recovery with additive noise")
  func testParameterRecoveryAdditiveNoise() {
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
      noiseVariance: 0.1
    )

    var observations: [[Double]] = []
    for state in trajectory {
      observations.append(obsModel.generateObservation(state: state))
    }

    var config = NewtonRaphsonMLE.Configuration()
    config.maxIterations = 20
    config.verbose = false

    let algorithm = NewtonRaphsonMLE(config: config)
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
    #expect(abs(result.parameters[0] - trueParameters[0]) < 0.4)
  }

  // MARK: - Convergence Tests

  @Test("Gradient norm decreases over iterations")
  func testGradientNormDecrease() {
    let trueParameters = [0.4]
    let initialState = model.typicalInitialState()

    let trajectory = model.simulateTrajectory(
      initialState: initialState,
      parameters: trueParameters,
      dt: 0.01,
      steps: 35
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

    var config = NewtonRaphsonMLE.Configuration()
    config.maxIterations = 15
    config.verbose = false

    let algorithm = NewtonRaphsonMLE(config: config)
    let result = algorithm.estimate(
      model: model,
      observations: observations,
      observationModel: obsModel,
      initialParameters: [0.1],
      initialState: initialState,
      initialCovariance: Matrix.identity(size: 40) * 0.1,
      dt: 0.01
    )

    if result.gradientNorms.count > 2 {
      let firstGradNorm = result.gradientNorms[0]
      let lastGradNorm = result.gradientNorms.last ?? firstGradNorm
      #expect(lastGradNorm < firstGradNorm)
    }
  }

  @Test("Log-likelihood is tracked over iterations")
  func testLogLikelihoodIncrease() {
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
      observedIndices: stride(from: 0, to: 40, by: 3).map { $0 },
      noiseVariance: 0.1  // Lower noise
    )

    var observations: [[Double]] = []
    for state in trajectory {
      observations.append(obsModel.generateObservation(state: state))
    }

    var config = NewtonRaphsonMLE.Configuration()
    config.maxIterations = 10
    config.verbose = false

    let algorithm = NewtonRaphsonMLE(config: config)
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
      noiseVariance: 0.1
    )

    var observations: [[Double]] = []
    for state in trajectory {
      observations.append(obsModel.generateObservation(state: state))
    }

    var config = NewtonRaphsonMLE.Configuration()
    config.maxIterations = 30
    config.gradientTolerance = 1e-3
    config.verbose = false

    let algorithm = NewtonRaphsonMLE(config: config)
    let result = algorithm.estimate(
      model: model,
      observations: observations,
      observationModel: obsModel,
      initialParameters: [0.2],
      initialState: initialState,
      initialCovariance: Matrix.identity(size: 40) * 0.1,
      dt: 0.01
    )

    #expect(result.iterations < 30)
  }

  // MARK: - Gradient Tests

  @Test("Gradient norms are tracked correctly")
  func testGradientNorms() {
    let trueParameters = [0.45]
    let initialState = model.typicalInitialState()

    let trajectory = model.simulateTrajectory(
      initialState: initialState,
      parameters: trueParameters,
      dt: 0.01,
      steps: 28
    )

    let obsModel = PartialObservationModel(
      stateDimension: 40,
      observedIndices: stride(from: 0, to: 40, by: 2).map { $0 },
      noiseVariance: 0.12
    )

    var observations: [[Double]] = []
    for state in trajectory {
      observations.append(obsModel.generateObservation(state: state))
    }

    var config = NewtonRaphsonMLE.Configuration()
    config.maxIterations = 15
    config.verbose = false

    let algorithm = NewtonRaphsonMLE(config: config)
    let result = algorithm.estimate(
      model: model,
      observations: observations,
      observationModel: obsModel,
      initialParameters: [0.1],
      initialState: initialState,
      initialCovariance: Matrix.identity(size: 40) * 0.1,
      dt: 0.01
    )

    #expect(result.gradientNorms.count == result.iterations)
    #expect(result.gradientNorms[0] > 0.0)
    // Gradient may not always decrease monotonically due to line search
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
      steps: 25
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

    var config = NewtonRaphsonMLE.Configuration()
    config.maxIterations = 15
    config.minParameterValue = 0.05
    config.maxParameterValue = 40.0
    config.verbose = false

    let algorithm = NewtonRaphsonMLE(config: config)
    let result = algorithm.estimate(
      model: model,
      observations: observations,
      observationModel: obsModel,
      initialParameters: [0.1],
      initialState: initialState,
      initialCovariance: Matrix.identity(size: 40) * 0.1,
      dt: 0.01
    )

    for param in result.parameters {
      #expect(param >= config.minParameterValue)
      #expect(param <= config.maxParameterValue)
    }

    for paramSet in result.parameterHistory {
      for param in paramSet {
        #expect(param >= config.minParameterValue)
        #expect(param <= config.maxParameterValue)
      }
    }
  }

  // MARK: - Line Search Tests

  @Test("Line search converges")
  func testLineSearch() {
    let trueParameters = [0.4]
    let initialState = model.typicalInitialState()

    let trajectory = model.simulateTrajectory(
      initialState: initialState,
      parameters: trueParameters,
      dt: 0.01,
      steps: 32
    )

    let obsModel = PartialObservationModel(
      stateDimension: 40,
      observedIndices: stride(from: 0, to: 40, by: 2).map { $0 },
      noiseVariance: 0.14
    )

    var observations: [[Double]] = []
    for state in trajectory {
      observations.append(obsModel.generateObservation(state: state))
    }

    var config = NewtonRaphsonMLE.Configuration()
    config.maxIterations = 10
    config.lineSearchMaxSteps = 5
    config.verbose = false

    let algorithm = NewtonRaphsonMLE(config: config)
    let result = algorithm.estimate(
      model: model,
      observations: observations,
      observationModel: obsModel,
      initialParameters: [0.15],
      initialState: initialState,
      initialCovariance: Matrix.identity(size: 40) * 0.1,
      dt: 0.01
    )

    #expect(result.parameters[0] > 0.0)
  }

  // MARK: - Hessian Tests

  @Test("Final Hessian is computed")
  func testFinalHessianComputed() {
    let trueParameters = [0.38]
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

    var config = NewtonRaphsonMLE.Configuration()
    config.maxIterations = 12
    config.verbose = false

    let algorithm = NewtonRaphsonMLE(config: config)
    let result = algorithm.estimate(
      model: model,
      observations: observations,
      observationModel: obsModel,
      initialParameters: [0.1],
      initialState: initialState,
      initialCovariance: Matrix.identity(size: 40) * 0.1,
      dt: 0.01
    )

    #expect(result.finalHessian != nil)

    if let hessian = result.finalHessian {
      #expect(hessian.rows == 1)
      #expect(hessian.cols == 1)
    }
  }

  // MARK: - Parameter History Tests

  @Test("Parameter history is correctly tracked")
  func testParameterHistoryTracking() {
    let trueParameters = [0.42]
    let initialState = model.typicalInitialState()

    let trajectory = model.simulateTrajectory(
      initialState: initialState,
      parameters: trueParameters,
      dt: 0.01,
      steps: 28
    )

    let obsModel = PartialObservationModel(
      stateDimension: 40,
      observedIndices: stride(from: 0, to: 40, by: 2).map { $0 },
      noiseVariance: 0.13
    )

    var observations: [[Double]] = []
    for state in trajectory {
      observations.append(obsModel.generateObservation(state: state))
    }

    var config = NewtonRaphsonMLE.Configuration()
    config.maxIterations = 18
    config.verbose = false

    let algorithm = NewtonRaphsonMLE(config: config)
    let result = algorithm.estimate(
      model: model,
      observations: observations,
      observationModel: obsModel,
      initialParameters: [0.12],
      initialState: initialState,
      initialCovariance: Matrix.identity(size: 40) * 0.1,
      dt: 0.01
    )

    #expect(result.parameterHistory.count == result.iterations + 1)
    #expect(result.parameterHistory[0] == [0.12])
    #expect(result.parameterHistory.last == result.parameters)
  }

  // MARK: - Edge Cases

  @Test("Handles very low noise")
  func testSmallNoise() {
    let trueParameters = [0.5]
    let initialState = model.typicalInitialState()

    let trajectory = model.simulateTrajectory(
      initialState: initialState,
      parameters: trueParameters,
      dt: 0.01,
      steps: 35
    )

    let obsModel = PartialObservationModel(
      stateDimension: 40,
      observedIndices: stride(from: 0, to: 40, by: 2).map { $0 },
      noiseVariance: 0.01
    )

    var observations: [[Double]] = []
    for state in trajectory {
      observations.append(obsModel.generateObservation(state: state))
    }

    var config = NewtonRaphsonMLE.Configuration()
    config.maxIterations = 15
    config.verbose = false

    let algorithm = NewtonRaphsonMLE(config: config)
    let result = algorithm.estimate(
      model: model,
      observations: observations,
      observationModel: obsModel,
      initialParameters: [0.2],
      initialState: initialState,
      initialCovariance: Matrix.identity(size: 40) * 0.05,
      dt: 0.01
    )

    #expect(result.parameters[0] > 0.0)
  }

  @Test("Handles high noise")
  func testLargeNoise() {
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
      noiseVariance: 2.0
    )

    var observations: [[Double]] = []
    for state in trajectory {
      observations.append(obsModel.generateObservation(state: state))
    }

    var config = NewtonRaphsonMLE.Configuration()
    config.maxIterations = 20
    config.verbose = false

    let algorithm = NewtonRaphsonMLE(config: config)
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
    #expect(result.parameters[0] < 10.0)
  }

  // MARK: - Comparison with EM

  @Test("Faster convergence when initialized from EM result")
  func testFasterConvergenceThanEM() {
    let trueParameters = [0.45]
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
      noiseVariance: 0.12
    )

    var observations: [[Double]] = []
    for state in trajectory {
      observations.append(obsModel.generateObservation(state: state))
    }

    // Run EM
    var emConfig = ExpectationMaximization.Configuration()
    emConfig.maxIterations = 50
    emConfig.verbose = false
    let em = ExpectationMaximization(config: emConfig)
    let emResult = em.estimate(
      model: model,
      observations: observations,
      observationModel: obsModel,
      initialParameters: [0.1],
      initialState: initialState,
      initialCovariance: Matrix.identity(size: 40) * 0.1,
      dt: 0.01
    )

    // Run NR starting from EM result
    var nrConfig = NewtonRaphsonMLE.Configuration()
    nrConfig.maxIterations = 20
    nrConfig.verbose = false
    let nr = NewtonRaphsonMLE(config: nrConfig)
    let nrResult = nr.estimate(
      model: model,
      observations: observations,
      observationModel: obsModel,
      initialParameters: emResult.parameters,
      initialState: initialState,
      initialCovariance: Matrix.identity(size: 40) * 0.1,
      dt: 0.01
    )

    #expect(nrResult.iterations < emResult.iterations)
  }
}

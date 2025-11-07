import Foundation

/// Protocol for observation models (Section 2.1)
/// Represents observation equations of the form:
/// y_k = H(x_k) + v_k
/// where:
/// - y_k: observation vector at time k
/// - H: observation operator (can be nonlinear)
/// - x_k: state vector at time k
/// - v_k: observation noise ~ N(0, R)
public protocol ObservationModel {
  /// Dimension of the observation vector
  var observationDimension: Int { get }

  /// Dimension of the state vector
  var stateDimension: Int { get }

  /// Observation noise covariance matrix R
  var observationNoiseCovariance: Matrix { get }

  /// Observation operator: H(x)
  /// - Parameter state: State vector x
  /// - Returns: Observation vector (without noise)
  func observationOperator(state: [Double]) -> [Double]

  /// Generate observation with noise: y = H(x) + v
  /// - Parameter state: State vector x
  /// - Returns: Noisy observation y
  func generateObservation(state: [Double]) -> [Double]

  /// Apply observation operator to ensemble
  /// - Parameter ensemble: Ensemble of states
  /// - Returns: Ensemble of observations
  func observeEnsemble(ensemble: Ensemble) -> Ensemble
}

// MARK: - Default Implementation

public extension ObservationModel {
  /// Default implementation: add Gaussian noise to observation
  func generateObservation(state: [Double]) -> [Double] {
    let observation = observationOperator(state: state)
    let noise = RandomUtils.generateGaussianNoiseWithCovariance(
      dimension: observationDimension,
      covariance: observationNoiseCovariance
    )

    var noisyObservation = [Double](repeating: 0.0, count: observationDimension)
    for i in 0..<observationDimension {
      noisyObservation[i] = observation[i] + noise[i]
    }

    return noisyObservation
  }

  /// Default implementation: observe each ensemble member
  func observeEnsemble(ensemble: Ensemble) -> Ensemble {
    var observedMembers = [[Double]]()
    observedMembers.reserveCapacity(ensemble.ensembleSize)

    for member in ensemble.members {
      let observation = observationOperator(state: member)
      observedMembers.append(observation)
    }

    return Ensemble(members: observedMembers)
  }
}

/// Linear observation model: y = Hx + v
/// Most common case where H is a matrix
public class LinearObservationModel: ObservationModel {
  public let observationDimension: Int
  public let stateDimension: Int
  public let observationNoiseCovariance: Matrix
  public let H: Matrix

  /// Initialize with observation matrix and noise covariance
  /// - Parameters:
  ///   - observationMatrix: Linear observation matrix H
  ///   - observationNoiseCovariance: Observation noise covariance R
  public init(observationMatrix: Matrix, observationNoiseCovariance: Matrix) {
    precondition(observationMatrix.rows == observationNoiseCovariance.rows,
                 "Observation matrix rows must match noise covariance dimension")
    precondition(observationNoiseCovariance.rows == observationNoiseCovariance.cols,
                 "Observation noise covariance must be square")

    self.H = observationMatrix
    self.observationDimension = observationMatrix.rows
    self.stateDimension = observationMatrix.cols
    self.observationNoiseCovariance = observationNoiseCovariance
  }

  /// Convenience initializer with diagonal noise covariance
  /// - Parameters:
  ///   - observationMatrix: Linear observation matrix H
  ///   - noiseVariances: Diagonal elements of R
  public convenience init(observationMatrix: Matrix, noiseVariances: [Double]) {
    let R = Matrix.diagonal(noiseVariances)
    self.init(observationMatrix: observationMatrix, observationNoiseCovariance: R)
  }

  public func observationOperator(state: [Double]) -> [Double] {
    precondition(state.count == stateDimension, "State dimension mismatch")
    return H.multiply(vector: state)
  }
}

/// Identity observation model: observe all state variables directly
/// y = x + v (H = I)
public class IdentityObservationModel: LinearObservationModel {
  /// Initialize identity observation with noise covariance
  /// - Parameters:
  ///   - dimension: State (and observation) dimension
  ///   - observationNoiseCovariance: Observation noise covariance R
  public init(dimension: Int, observationNoiseCovariance: Matrix) {
    let H = Matrix.identity(size: dimension)
    super.init(observationMatrix: H, observationNoiseCovariance: observationNoiseCovariance)
  }

  /// Convenience initializer with uniform noise variance
  /// - Parameters:
  ///   - dimension: State (and observation) dimension
  ///   - noiseVariance: Uniform observation noise variance
  public convenience init(dimension: Int, noiseVariance: Double) {
    let variances = Array(repeating: noiseVariance, count: dimension)
    let R = Matrix.diagonal(variances)
    self.init(dimension: dimension, observationNoiseCovariance: R)
  }
}

/// Partial observation model: observe subset of state variables
/// Useful when only some state components are measured
public class PartialObservationModel: LinearObservationModel {
  /// Initialize with indices of observed state variables
  /// - Parameters:
  ///   - stateDimension: Full state dimension
  ///   - observedIndices: Indices of state variables to observe
  ///   - noiseVariances: Observation noise variances for each observed variable
  public init(stateDimension: Int, observedIndices: [Int], noiseVariances: [Double]) {
    precondition(observedIndices.count == noiseVariances.count,
                 "Number of observed indices must match noise variances")
    precondition(observedIndices.allSatisfy { $0 >= 0 && $0 < stateDimension },
                 "All indices must be valid")

    let observationDim = observedIndices.count
    var H = Matrix(rows: observationDim, cols: stateDimension)

    for (i, idx) in observedIndices.enumerated() {
      H[i, idx] = 1.0
    }

    let R = Matrix.diagonal(noiseVariances)
    super.init(observationMatrix: H, observationNoiseCovariance: R)
  }

  /// Convenience initializer with uniform noise
  /// - Parameters:
  ///   - stateDimension: Full state dimension
  ///   - observedIndices: Indices of state variables to observe
  ///   - noiseVariance: Uniform observation noise variance
  public convenience init(stateDimension: Int, observedIndices: [Int], noiseVariance: Double) {
    let variances = Array(repeating: noiseVariance, count: observedIndices.count)
    self.init(stateDimension: stateDimension, observedIndices: observedIndices, noiseVariances: variances)
  }
}

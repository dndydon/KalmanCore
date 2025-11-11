import Foundation
import Accelerate

/// Utilities for generating random numbers for stochastic processes
public enum RandomUtils {

  /// Generate standard Gaussian noise vector: N(0, I)
  /// - Parameter dimension: The dimension of the noise vector
  /// - Returns: Array of independent standard normal random variables
  public static func generateGaussianNoise(dimension: Int) -> [Double] {
    var noise = [Double](repeating: 0.0, count: dimension)

    for i in stride(from: 0, to: dimension, by: 2) {
      // Box-Muller transform for generating pairs of independent normals
      let u1 = Double.random(in: 0..<1)
      let u2 = Double.random(in: 0..<1)

      let r = sqrt(-2 * log(u1))
      let theta = 2 * .pi * u2

      noise[i] = r * cos(theta)
      if i + 1 < dimension {
        noise[i + 1] = r * sin(theta)
      }
    }

    return noise
  }

  /// Generate Gaussian noise with specified covariance: N(0, Σ)
  /// Uses simple diagonal approximation (for full covariance, use Cholesky decomposition)
  /// - Parameters:
  ///   - dimension: The dimension of the noise vector
  ///   - covariance: Covariance matrix
  /// - Returns: Gaussian random vector with specified covariance
  public static func generateGaussianNoiseWithCovariance(dimension: Int, covariance: Matrix) -> [Double] {
    precondition(covariance.rows == dimension && covariance.cols == dimension,
                 "Covariance matrix must be square and match dimension")

    let standardNoise = generateGaussianNoise(dimension: dimension)

    // For diagonal covariance matrices (common case)
    if isDiagonal(covariance) {
      var result = [Double](repeating: 0.0, count: dimension)
      for i in 0..<dimension {
        result[i] = standardNoise[i] * sqrt(covariance[i, i])
      }
      return result
    }

    // For general covariance, use Cholesky decomposition: Σ = L * L^T
    // Then noise = L * standardNoise
    if let L = choleskyDecomposition(covariance) {
      return L.multiply(vector: standardNoise)
    }

    // Fallback: diagonal approximation
    var result = [Double](repeating: 0.0, count: dimension)
    for i in 0..<dimension {
      result[i] = standardNoise[i] * sqrt(max(0, covariance[i, i]))
    }
    return result
  }

  /// Check if matrix is diagonal
  private static func isDiagonal(_ matrix: Matrix) -> Bool {
    guard matrix.rows == matrix.cols else { return false }

    for i in 0..<matrix.rows {
      for j in 0..<matrix.cols {
        if i != j && abs(matrix[i, j]) > 1e-10 {
          return false
        }
      }
    }
    return true
  }

  /// Cholesky decomposition: A = L * L^T
  /// https://en.wikipedia.org/wiki/Cholesky_decomposition
  /// Returns lower triangular matrix L if A is positive definite, nil otherwise
  private static func choleskyDecomposition(_ matrix: Matrix) -> Matrix? {
    guard matrix.rows == matrix.cols else { return nil }

    let n = matrix.rows
    var L = Matrix(rows: n, cols: n)

    for i in 0..<n {
      for j in 0...i {
        var sum = 0.0

        if i == j {
          for k in 0..<j {
            sum += L[j, k] * L[j, k]
          }
          let val = matrix[j, j] - sum
          if val <= 0 {
            return nil  // Not positive definite
          }
          L[j, j] = sqrt(val)
        } else {
          for k in 0..<j {
            sum += L[i, k] * L[j, k]
          }
          L[i, j] = (matrix[i, j] - sum) / L[j, j]
        }
      }
    }

    return L
  }

  /// Generate uniform random vector in [0, 1]^n
  public static func generateUniformNoise(dimension: Int) -> [Double] {
    return (0..<dimension).map { _ in Double.random(in: 0..<1) }
  }

  /// Resample indices according to weights (for particle filters)
  /// - Parameter weights: Normalized weights (should sum to 1)
  /// - Returns: Array of resampled indices
  public static func resample(weights: [Double], count: Int) -> [Int] {
    let n = weights.count
    var indices = [Int]()
    indices.reserveCapacity(count)

    // Systematic resampling
    let u0 = Double.random(in: 0..<1) / Double(count)
    var cumulativeSum = 0.0
    var j = 0

    for i in 0..<count {
      let u = u0 + Double(i) / Double(count)

      while j < n && cumulativeSum < u {
        cumulativeSum += weights[j]
        j += 1
      }

      indices.append(min(j - 1, n - 1))
    }

    return indices
  }
}

import Foundation

/// Likelihood utilities shared across estimation algorithms.
public enum Likelihood {
  /// Gaussian innovation log-likelihood contribution
  /// ℓ = -0.5 [ m log(2π) + log(det(S)) + (y - Hx)^T S^{-1} (y - Hx) ]
  /// - Parameters:
  ///   - innovation: y - Hx (length m)
  ///   - covariance: Innovation covariance S (m×m, SPD)
  /// - Returns: Log-likelihood contribution (double); returns -infinity if det(S) <= 0
  public static func gaussianInnovationLogLikelihood(innovation: [Double], covariance S: Matrix) -> Double {
    let m = innovation.count
    let detS = matrixDeterminant(S)
    guard detS > 0 else { return -Double.infinity }
    let Sinv = matrixInverse(S)
    let mahal = dotProduct(innovation, Sinv.multiply(vector: innovation))
    return -0.5 * (Double(m) * log(2 * .pi) + log(detS) + mahal)
  }
}

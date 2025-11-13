import Foundation
import Accelerate

// Matrix utilities with dual paths:
// - For small matrices, use simplified educational implementations (readable, dependency-light)
// - For larger matrices, prefer Accelerate/LAPACK for robustness and performance
// NOTE: These routines operate on the custom row-major Matrix; LAPACK expects column-major.
//       Conversions are handled internally when using the Accelerate path.
//
// Numerical caveats and guidance
// - Pivoting and sign: LU factorization uses partial pivoting (dgetrf) which introduces a
//   permutation. det(A) = sign(P) * Π diag(U). We compute the sign from pivot indices.
// - Near-singular matrices: when a pivot is nearly zero in simplified paths, we add a tiny
//   regularization (1e-6) or return a small positive determinant (1e-10) as a guard to avoid
//   division by zero. This signals ill-conditioning; downstream code should handle gracefully.
// - SPD matrices: for symmetric positive definite matrices (e.g., innovation covariance S),
//   prefer Cholesky-based routines (dpotrf) for inversion and log-det (sum(log(diag(L))) * 2)
//   to preserve symmetry and improve numerical stability. Current code uses general LU for
//   simplicity; future optimization could specialize SPD paths.
// - Log-determinant: if you need log|A|, prefer accumulating logs during factorization rather
//   than forming det and taking log, to reduce overflow/underflow risk.
// - Reproducibility: Accelerate may exhibit small platform-specific numeric differences.
//   For bitwise stability in tests, consider lowering accelerateThreshold or forcing the
//   simplified path.

private let accelerateThreshold = 3 // use Accelerate when n >= this size

// MARK: - Public API

public func matrixInverse(_ matrix: Matrix) -> Matrix {
  precondition(matrix.rows == matrix.cols, "Matrix must be square")
  let n = matrix.rows
  if n == 1 { return Matrix(rows: 1, cols: 1, data: [1.0 / matrix[0, 0]]) }

  // Prefer Accelerate for moderate/large matrices; fall back if it fails
  if n >= accelerateThreshold {
    if let inv = inverseAccelerate(matrix) { return inv }
    // Fall through to simplified path if Accelerate fails
  }

  return inverseSimplified(matrix)
}

public func matrixDeterminant(_ matrix: Matrix) -> Double {
  precondition(matrix.rows == matrix.cols, "Matrix must be square")
  let n = matrix.rows
  if n == 1 { return matrix[0, 0] }
  if n == 2 { return matrix[0, 0] * matrix[1, 1] - matrix[0, 1] * matrix[1, 0] }

  if n >= accelerateThreshold {
    if let det = determinantAccelerate(matrix) { return det }
  }
  return determinantSimplified(matrix)
}

// MARK: - Simplified (educational) implementations
// These are preserved for readability and as a robust fallback for tiny matrices

private func inverseSimplified(_ matrix: Matrix) -> Matrix {
  let n = matrix.rows
  var augmented = Matrix(rows: n, cols: 2 * n)
  for i in 0..<n {
    for j in 0..<n {
      augmented[i, j] = matrix[i, j]
      augmented[i, j + n] = (i == j) ? 1.0 : 0.0
    }
  }
  for i in 0..<n {
    var pivot = augmented[i, i]
    if abs(pivot) < 1e-10 { augmented[i, i] += 1e-6; pivot = augmented[i, i] }
    for j in 0..<(2 * n) { augmented[i, j] /= pivot }
    for k in 0..<n where k != i {
      let factor = augmented[k, i]
      for j in 0..<(2 * n) { augmented[k, j] -= factor * augmented[i, j] }
    }
  }
  var inverse = Matrix(rows: n, cols: n)
  for i in 0..<n { for j in 0..<n { inverse[i, j] = augmented[i, j + n] } }
  return inverse
}

private func determinantSimplified(_ matrix: Matrix) -> Double {
  let n = matrix.rows
  var det = 1.0
  var A = matrix
  for i in 0..<n {
    if abs(A[i, i]) < 1e-10 { return 1e-10 }
    det *= A[i, i]
    for k in (i + 1)..<n {
      let factor = A[k, i] / A[i, i]
      for j in i..<n { A[k, j] -= factor * A[i, j] }
    }
  }
  return det
}

// MARK: - Accelerate/LAPACK-backed implementations

private func toColumnMajor(_ A: Matrix) -> [Double] {
  var colMajor = [Double](repeating: 0.0, count: A.rows * A.cols)
  for j in 0..<A.cols {
    for i in 0..<A.rows {
      colMajor[j * A.rows + i] = A[i, j]
    }
  }
  return colMajor
}

private func fromColumnMajor(rows: Int, cols: Int, _ colData: [Double]) -> Matrix {
  var M = Matrix(rows: rows, cols: cols)
  for j in 0..<cols {
    for i in 0..<rows {
      M[i, j] = colData[j * rows + i]
    }
  }
  return M
}

private func inverseAccelerate(_ A: Matrix) -> Matrix? {
  var n32 = __CLPK_integer(A.rows)
  var m = n32
  var nn = n32
  var lda = n32
  var pivots = [__CLPK_integer](repeating: 0, count: Int(n32))
  var info: __CLPK_integer = 0

  // Copy to column-major
  var a = toColumnMajor(A)

  // LU factorization
  dgetrf_(&m, &nn, &a, &lda, &pivots, &info)
  if info != 0 { return nil }

  // Query optimal work size
  var lwork: __CLPK_integer = -1
  var wkopt: Double = 0.0
  dgetri_(&n32, &a, &lda, &pivots, &wkopt, &lwork, &info)
  if info != 0 && info != -7 { return nil }

  lwork = __CLPK_integer(wkopt)
  var work = [Double](repeating: 0.0, count: Int(lwork))
  dgetri_(&n32, &a, &lda, &pivots, &work, &lwork, &info)
  if info != 0 { return nil }

  return fromColumnMajor(rows: A.rows, cols: A.cols, a)
}

private func determinantAccelerate(_ A: Matrix) -> Double? {
  let n32 = __CLPK_integer(A.rows)
  var m = n32
  var nn = n32
  var lda = n32
  var pivots = [__CLPK_integer](repeating: 0, count: Int(n32))
  var info: __CLPK_integer = 0
  var a = toColumnMajor(A)

  dgetrf_(&m, &nn, &a, &lda, &pivots, &info)
  if info != 0 { return nil }

  // det(A) = sign * Π diag(U)
  var det = 1.0
  for i in 0..<Int(n32) {
    det *= a[i * Int(lda) + i]
  }
  // Sign from pivot permutations: count of row interchanges
  var sign: Double = 1.0
  for i in 0..<Int(n32) {
    if pivots[i] != __CLPK_integer(i + 1) { sign *= -1.0 }
  }
  return sign * det
}

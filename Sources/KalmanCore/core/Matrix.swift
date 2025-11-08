import Foundation
import Accelerate

/// A matrix structure optimized for numerical computations using Accelerate framework
public struct Matrix {
  public let rows: Int
  public let cols: Int
  public var data: [Double]
  
  /// Initialize matrix with explicit data
  public init(rows: Int, cols: Int, data: [Double]) {
    precondition(data.count == rows * cols, "Data size must match matrix dimensions")
    self.rows = rows
    self.cols = cols
    self.data = data
  }
  
  /// Initialize matrix with repeated value
  public init(rows: Int, cols: Int, repeating value: Double = 0.0) {
    self.rows = rows
    self.cols = cols
    self.data = Array(repeating: value, count: rows * cols)
  }
  
  /// Initialize identity matrix
  public static func identity(size: Int) -> Matrix {
    var matrix = Matrix(rows: size, cols: size)
    for i in 0..<size {
      matrix[i, i] = 1.0
    }
    return matrix
  }
  
  /// Initialize diagonal matrix from array
  public static func diagonal(_ values: [Double]) -> Matrix {
    let n = values.count
    var matrix = Matrix(rows: n, cols: n)
    for i in 0..<n {
      matrix[i, i] = values[i]
    }
    return matrix
  }
  
  /// Access matrix elements
  public subscript(row: Int, col: Int) -> Double {
    get {
      precondition(row >= 0 && row < rows && col >= 0 && col < cols, "Index out of bounds")
      return data[row * cols + col]
    }
    set {
      precondition(row >= 0 && row < rows && col >= 0 && col < cols, "Index out of bounds")
      data[row * cols + col] = newValue
    }
  }
  
  /// Matrix multiplication: A * B
  public static func * (lhs: Matrix, rhs: Matrix) -> Matrix {
    precondition(lhs.cols == rhs.rows, "Matrix dimensions incompatible for multiplication")
    var result = Matrix(rows: lhs.rows, cols: rhs.cols)
    
    vDSP_mmulD(lhs.data, 1, rhs.data, 1, &result.data, 1,
               vDSP_Length(lhs.rows), vDSP_Length(rhs.cols), vDSP_Length(lhs.cols))
    
    return result
  }
  
  /// Matrix-vector multiplication: A * x
  public func multiply(vector: [Double]) -> [Double] {
    precondition(cols == vector.count, "Vector dimension must match matrix columns")
    var result = [Double](repeating: 0.0, count: rows)
    
    vDSP_mmulD(data, 1, vector, 1, &result, 1,
               vDSP_Length(rows), 1, vDSP_Length(cols))
    
    return result
  }
  
  /// Matrix addition: A + B
  public static func + (lhs: Matrix, rhs: Matrix) -> Matrix {
    precondition(lhs.rows == rhs.rows && lhs.cols == rhs.cols, "Matrix dimensions must match")
    var result = Matrix(rows: lhs.rows, cols: lhs.cols)
    
    vDSP_vaddD(lhs.data, 1, rhs.data, 1, &result.data, 1, vDSP_Length(lhs.data.count))
    
    return result
  }
  
  /// Matrix subtraction: A - B
  public static func - (lhs: Matrix, rhs: Matrix) -> Matrix {
    precondition(lhs.rows == rhs.rows && lhs.cols == rhs.cols, "Matrix dimensions must match")
    var result = Matrix(rows: lhs.rows, cols: lhs.cols)
    
    vDSP_vsubD(rhs.data, 1, lhs.data, 1, &result.data, 1, vDSP_Length(lhs.data.count))
    
    return result
  }
  
  /// Scalar multiplication: α * A
  public static func * (scalar: Double, matrix: Matrix) -> Matrix {
    var result = Matrix(rows: matrix.rows, cols: matrix.cols)
    var scalarValue = scalar
    
    vDSP_vsmulD(matrix.data, 1, &scalarValue, &result.data, 1, vDSP_Length(matrix.data.count))
    
    return result
  }
  
  /// Scalar multiplication: A * α
  public static func * (matrix: Matrix, scalar: Double) -> Matrix {
    return scalar * matrix
  }
  
  /// Matrix transpose
  public var transposed: Matrix {
    var result = Matrix(rows: cols, cols: rows)
    vDSP_mtransD(data, 1, &result.data, 1, vDSP_Length(cols), vDSP_Length(rows))
    return result
  }
  
  /// Matrix trace (sum of diagonal elements)
  public var trace: Double {
    precondition(rows == cols, "Trace is only defined for square matrices")
    var sum = 0.0
    for i in 0..<rows {
      sum += self[i, i]
    }
    return sum
  }
  
  /// Frobenius norm
  public var frobeniusNorm: Double {
    var result = 0.0
    vDSP_svesqD(data, 1, &result, vDSP_Length(data.count))
    return sqrt(result)
  }
}


// MARK: - CustomStringConvertible for debug output
extension Matrix: CustomStringConvertible {
  public var description: String {
    // Helper to format numbers consistently.
    func format(_ x: Double) -> String {
      // Choose a compact but readable default. Adjust as needed.
      return String(format: "%.6g", x)
      //return String(format: "%.6f", x)
      // Alternatives:
      // return String(format: "%.1f", x)
      // return String(format: "% .6e", x)
    }

    // Handle empty shapes gracefully
    guard rows > 0 && cols > 0 else { return "\n  ⎡ ⎤\n" }

    // Compute per-column widths based on formatted content
    var colWidths = [Int](repeating: 0, count: cols)
    for j in 0..<cols {
      var w = 0
      for i in 0..<rows {
        let s = format(self[i, j])
        if s.count > w { w = s.count }
      }
      colWidths[j] = w
    }

    // Build each row string with right-aligned cols
    var lines: [String] = []
    lines.reserveCapacity(rows)
    for i in 0..<rows {
      var parts: [String] = []
      parts.reserveCapacity(cols)
      for j in 0..<cols {
        let s = format(self[i, j])
        let pad = max(0, colWidths[j] - s.count)
        let cell = String(repeating: " ", count: pad) + s
        parts.append(cell)
      }
      // Two spaces between cols for readability
      let rowBody = parts.joined(separator: "  ")
      lines.append(rowBody)
    }

    // Determine the inner width to align brackets consistently
    let innerWidth = lines.map { $0.count }.max() ?? 0

    func wrap(_ i: Int, _ body: String) -> String {
      let padded = body + String(repeating: " ", count: max(0, innerWidth - body.count))
      switch (i, rows) {
        case (0, 1):
          // Single-row matrix
          return "  ( \(padded) )"
        case (0, _):
          return "  ⎛ \(padded) ⎞"
        case (rows - 1, _):
          return "  ⎝ \(padded) ⎠"
        default:
          return "  ⎜ \(padded) ⎥"
      }
    }
    
    var result = "Matrix(\(rows)×\(cols)):\n"
    for i in 0..<rows {
      result += wrap(i, lines[i]) + "\n"
    }
    return result
  }
}

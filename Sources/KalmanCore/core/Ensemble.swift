import Foundation
import Accelerate

/// Represents an ensemble of state vectors for Ensemble Kalman Filter
/// Section 2.1: Used in stochastic parameterization identification
public struct Ensemble {
    public let ensembleSize: Int
    public let stateDimension: Int
    public var members: [[Double]]
    
    /// Initialize ensemble with specified size and dimension
    public init(ensembleSize: Int, stateDimension: Int) {
        precondition(ensembleSize > 0, "Ensemble size must be positive")
        precondition(stateDimension > 0, "State dimension must be positive")
        
        self.ensembleSize = ensembleSize
        self.stateDimension = stateDimension
        self.members = Array(repeating: Array(repeating: 0.0, count: stateDimension),
                            count: ensembleSize)
    }
    
    /// Initialize ensemble with explicit members
    public init(members: [[Double]]) {
        precondition(!members.isEmpty, "Ensemble must have at least one member")
        
        let dimension = members[0].count
        precondition(members.allSatisfy { $0.count == dimension },
                     "All ensemble members must have the same dimension")
        
        self.ensembleSize = members.count
        self.stateDimension = dimension
        self.members = members
    }
    
    /// Initialize ensemble from mean and covariance with random perturbations
    public init(mean: [Double], covariance: Matrix, ensembleSize: Int) {
        self.init(ensembleSize: ensembleSize, stateDimension: mean.count)
        
        for i in 0..<ensembleSize {
            let perturbation = RandomUtils.generateGaussianNoiseWithCovariance(
                dimension: stateDimension,
                covariance: covariance
            )
            
            for j in 0..<stateDimension {
                members[i][j] = mean[j] + perturbation[j]
            }
        }
    }
    
    /// Compute ensemble mean: x̄ = (1/N) Σᵢ xᵢ
    public var mean: [Double] {
        var meanState = [Double](repeating: 0.0, count: stateDimension)
        
        for member in members {
            for i in 0..<stateDimension {
                meanState[i] += member[i]
            }
        }
        
        let scale = 1.0 / Double(ensembleSize)
        for i in 0..<stateDimension {
            meanState[i] *= scale
        }
        
        return meanState
    }
    
    /// Compute ensemble covariance: P = (1/(N-1)) Σᵢ (xᵢ - x̄)(xᵢ - x̄)ᵀ
    public var covariance: Matrix {
        let meanState = self.mean
        var cov = Matrix(rows: stateDimension, cols: stateDimension)
        
        let scale = 1.0 / Double(ensembleSize - 1)
        
        for i in 0..<stateDimension {
            for j in 0..<stateDimension {
                var sum = 0.0
                for member in members {
                    let devI = member[i] - meanState[i]
                    let devJ = member[j] - meanState[j]
                    sum += devI * devJ
                }
                cov[i, j] = sum * scale
            }
        }
        
        return cov
    }
    
    /// Compute anomaly matrix: A = [x₁-x̄, x₂-x̄, ..., xₙ-x̄]
    /// Used in EnKF analysis step
    public var anomalyMatrix: Matrix {
        let meanState = self.mean
        var anomalies = Matrix(rows: stateDimension, cols: ensembleSize)
        
        for (col, member) in members.enumerated() {
            for row in 0..<stateDimension {
                anomalies[row, col] = member[row] - meanState[row]
            }
        }
        
        return anomalies
    }
    
    /// Apply a transformation to all ensemble members
    public mutating func transform(_ transformation: ([Double]) -> [Double]) {
        for i in 0..<ensembleSize {
            members[i] = transformation(members[i])
        }
    }
    
    /// Inflate ensemble around mean (covariance inflation)
    /// x'ᵢ = x̄ + α(xᵢ - x̄), where α > 1 increases spread
    public mutating func inflate(by factor: Double) {
        precondition(factor > 0, "Inflation factor must be positive")
        
        let meanState = self.mean
        
        for i in 0..<ensembleSize {
            for j in 0..<stateDimension {
                let deviation = members[i][j] - meanState[j]
                members[i][j] = meanState[j] + factor * deviation
            }
        }
    }
    
    /// Add random perturbations to ensemble members (additive inflation)
    public mutating func addPerturbations(covariance: Matrix) {
        for i in 0..<ensembleSize {
            let perturbation = RandomUtils.generateGaussianNoiseWithCovariance(
                dimension: stateDimension,
                covariance: covariance
            )
            
            for j in 0..<stateDimension {
                members[i][j] += perturbation[j]
            }
        }
    }
    
    /// Get specific ensemble member
    public subscript(index: Int) -> [Double] {
        get {
            precondition(index >= 0 && index < ensembleSize, "Index out of bounds")
            return members[index]
        }
        set {
            precondition(index >= 0 && index < ensembleSize, "Index out of bounds")
            precondition(newValue.count == stateDimension, "Member dimension must match")
            members[index] = newValue
        }
    }
    
    /// Compute spread (average standard deviation across dimensions)
    public var spread: Double {
        let meanState = self.mean
        var totalVariance = 0.0
        
        for i in 0..<stateDimension {
            var variance = 0.0
            for member in members {
                let dev = member[i] - meanState[i]
                variance += dev * dev
            }
            variance /= Double(ensembleSize - 1)
            totalVariance += variance
        }
        
        return sqrt(totalVariance / Double(stateDimension))
    }
}

// MARK: - CustomStringConvertible

extension Ensemble: CustomStringConvertible {
    public var description: String {
        let meanState = mean
        let spreadValue = spread
        
        return """
        Ensemble(size: \(ensembleSize), dimension: \(stateDimension))
        Mean: [\(meanState.prefix(5).map { String(format: "%.3f", $0) }.joined(separator: ", "))\
        \(stateDimension > 5 ? "..." : "")]
        Spread: \(String(format: "%.3f", spreadValue))
        """
    }
}

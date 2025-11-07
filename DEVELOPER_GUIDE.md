# KalmanCore Developer Guide

Quick reference for continuing development on KalmanCore framework.

## Project Structure

```
KalmanCore/
├── Sources/KalmanCore/
│   ├── core/                    # Matrix, Random, Ensemble utilities
│   ├── models/                  # Stochastic systems (Lorenz96, etc.)
│   ├── observations/            # Observation operators
│   ├── filters/                 # Kalman variants
│   ├── estimation/              # EM, NR-MLE algorithms
│   ├── algorithms/              # Combined algorithms (Section 3)
│   ├── protocols/               # Abstract interfaces
│   └── examples/                # Demonstrations
├── Tests/KalmanCoreTests/       # Test suite
├── README.md                    # User-facing documentation
├── SECTION_2_1.md               # Framework documentation
├── SECTION_2_2.md               # EM algorithm documentation
├── SECTION_2_3.md               # NR-MLE documentation
├── IMPLEMENTATION_ROADMAP.md    # Path to v1.0
└── COMPLETION_SUMMARY.md        # Current status
```

## Current Status (v0.5)

✅ **Complete**: Sections 2.1, 2.2, 2.3
- 3,500+ lines of production code
- 1,100+ lines of tests (28 tests, all passing)
- 2,200+ lines of documentation

⏳ **Next**: Section 3 (Sequential Parameter Estimation)
- Estimated 2-3 weeks
- See `IMPLEMENTATION_ROADMAP.md` for details

## Building & Testing

### Build
```bash
cd /Users/donsleeter/Library/Mobile\ Documents/com~apple~CloudDocs/Developer/SwiftUI/KalmanCore
swift build
```

### Run Tests
```bash
swift test
```

### Run Specific Test Suite
```bash
swift test --filter ExpectationMaximizationTests
swift test --filter NewtonRaphsonMLETests
```

## Code Patterns to Follow

### 1. Algorithm Structure (Configuration + Result)

**Configuration struct:**
```swift
public struct Configuration {
    public var maxIterations: Int = 50
    public var tolerance: Double = 1e-4
    // ...
    public init() {}
}
```

**Result struct:**
```swift
public struct Result {
    public let parameters: [Double]
    public let history: [[Double]]
    public let converged: Bool
    public let iterations: Int
    // ...
}
```

**Main API:**
```swift
public func estimate<Model: StochasticDynamicalSystem>(
    model: Model,
    observations: [[Double]],
    observationModel: ObservationModel,
    initialParameters: [Double],
    initialState: [Double],
    initialCovariance: Matrix,
    dt: Double
) -> Result
```

### 2. Documentation Pattern

Every new algorithm needs:
1. **SECTION_X_Y.md** with:
   - Mathematical framework
   - Implementation details
   - Usage examples
   - Convergence analysis
   - Performance characteristics

2. **In-code documentation**:
   - Comprehensive docstrings
   - Usage examples in comments
   - References to paper sections

3. **Tests**:
   - 10-15 tests per algorithm
   - Unit + integration + convergence tests
   - Edge case handling

### 3. Test Structure (Swift Testing)

```swift
@Suite("Algorithm Name Tests")
struct AlgorithmTests {
    var model: Lorenz96Model
    var algorithm: Algorithm
    
    init() {
        model = Lorenz96Model.standard(stochasticType: .additive)
        algorithm = Algorithm()
    }
    
    @Test("Test description")
    func testSomething() {
        // Setup
        // Execute
        // Verify with #expect()
    }
}
```

## Key Classes & Protocols

### Protocols (Extensible)
- **StochasticDynamicalSystem**: Model interface
- **ObservationModel**: Observation operator interface
- **Filter**: Base filter interface
- **EnsembleFilter**: Ensemble-specific interface

### Core Classes
- **Matrix**: Accelerate-based matrix operations
- **Ensemble**: Ensemble representation
- **Lorenz96Model**: Chaotic system example

### Algorithm Classes
- **ExpectationMaximization**: EM algorithm
- **NewtonRaphsonMLE**: Newton-Raphson MLE

## Implementing New Components

### Adding a New Filter

1. Create `sources/KalmanCore/filters/NewFilter.swift`
2. Implement `Filter` protocol:
   ```swift
   public class NewFilter {
       public struct Configuration { /* ... */ }
       public struct Result { /* ... */ }
       public func filter(...) -> Result { /* ... */ }
   }
   ```
3. Create `Tests/KalmanCoreTests/NewFilterTests.swift`
4. Document in `SECTION_X_Y.md`
5. Add usage examples to README

### Adding a New Model

1. Create `Sources/KalmanCore/models/NewModel.swift`
2. Inherit from/implement `StochasticDynamicalSystem`
3. Implement required methods:
   - `deterministicDynamics(...)`
   - `stochasticParameterization(...)`
   - `transition(...)`
4. Add convenience factory methods
5. Document with examples

## Common Tasks

### Run Full Test Suite
```bash
swift test 2>&1 | grep -E "passed|failed"
```

### Check for Compiler Warnings
```bash
swift build 2>&1 | grep warning
```

### Build Documentation
No special build needed - markdown files are documentation.

### Profile Performance
```bash
swift build -c release
time swift run
```

## Key References

### Paper Sections Implemented
- **Section 2.1**: Stochastic Parameterization Framework
- **Section 2.2**: Expectation-Maximization Algorithm
- **Section 2.3**: Newton-Raphson MLE

### Paper Sections Pending
- **Section 3.1**: Sequential EnKF with Parameter Augmentation
- **Section 3.2**: Combined EnKF-EM
- **Section 3.3**: Combined EnKF-NR

### Supporting Papers
- Kalman (1960) - Kalman filter foundations
- Rauch et al. (1965) - RTS smoother
- Dempster et al. (1977) - EM algorithm
- Evensen (1994) - Ensemble Kalman filter
- Nocedal & Wright (2006) - Optimization

## Development Checklist (for new features)

- [ ] Implementation in appropriate source file
- [ ] Configuration struct for parameters
- [ ] Result struct for output
- [ ] 10+ comprehensive tests
- [ ] SECTION_X_Y.md documentation
- [ ] Usage examples in README
- [ ] In-code docstrings
- [ ] Tests pass: `swift test`
- [ ] Clean build: `swift build`
- [ ] No compiler warnings
- [ ] Cross-platform tested

## Useful Commands

```bash
# View git history
git log --oneline

# Check current changes
git status

# Run tests with verbose output
swift test --verbose

# Build release version
swift build -c release

# Clean build
swift build --clean

# Show test names
swift test --list
```

## Performance Guidelines

- EM per iteration: ~500ms for Lorenz96 (n=40)
- NR-MLE per iteration: ~1-2s for Lorenz96
- Target: <30s for full parameter estimation

## Memory Guidelines

- O(T·n²) for observations
- Test with T=1000, n=40 minimum
- Check for memory leaks with Instruments if needed

## Code Style

- Use Swift 6 idioms
- Follow existing naming conventions
- Prefer clarity over brevity
- Document public APIs thoroughly
- Add inline comments for complex logic

## Next Steps

1. Read `IMPLEMENTATION_ROADMAP.md` for full v1.0 plan
2. Review `SECTION_2_1.md`, `SECTION_2_2.md`, `SECTION_2_3.md` for theory
3. Start with Phase 1: Sequential Kalman Filter
4. Follow the patterns established in EM and NR-MLE implementations

## Questions?

1. Check SECTION_X_Y.md documentation
2. Review test cases for usage examples
3. Look at existing code patterns
4. Consult main paper (Pulido et al., 2018)

---

**Last Updated**: November 7, 2025
**Version**: v0.5
**Next Phase**: Section 3 - Sequential Parameter Estimation

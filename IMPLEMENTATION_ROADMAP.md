# KalmanCore Swift Package Implementation Roadmap

## Overview

This roadmap describes the path to completing KalmanCore v1.0, implementing all sections of:

- Pulido, M., Tandeo, P., Bocquet, M., Carrassi, A., & Lucini, M. (2018).  
- Stochastic parameterization identification using ensemble Kalman filtering combined with maximum likelihood methods.  
- *Tellus A: Dynamic Meteorology and Oceanography*, 70:1, 1-17.

## Completed (✅)

### Section 2.1: Stochastic Parameterization Framework
- ✅ Matrix operations with Accelerate framework
- ✅ Random utilities for noise generation
- ✅ Ensemble structures
- ✅ StochasticDynamicalSystem protocol
- ✅ Lorenz96Model implementation (additive, state-dependent, diagonal noise)
- ✅ Observation models (linear, identity, partial)
- ✅ SECTION_2_1.md documentation

### Section 2.2: Expectation-Maximization Algorithm
- ✅ ExpectationMaximization class
- ✅ E-step (Kalman filter + RTS smoother)
- ✅ M-step (closed-form solution for additive noise)
- ✅ Convergence criteria and monitoring
- ✅ Parameter constraints and bounds
- ✅ SECTION_2_2.md documentation
- ✅ 14 comprehensive tests

### Section 2.3: Newton-Raphson MLE
- ✅ NewtonRaphsonMLE class
- ✅ Gradient computation via finite differences
- ✅ Hessian computation and regularization
- ✅ Line search with Armijo conditions
- ✅ Convergence criteria
- ✅ SECTION_2_3.md documentation
- ✅ 14 comprehensive tests

---

## Remaining Implementation (⏳)

### Phase 1: Section 3 - Sequential Parameter Estimation
**Estimated effort: 2-3 weeks**

Current status (in progress)
- Augmented-state EnKF implemented (initial version):
  - Forecast step with parameter evolution: `.constant`, `.randomWalk(Qθ)`, `.ar1(ρ,Qθ)`
  - Analysis step (perturbed/deterministic) with augmented gain K using ensemble anomalies
  - Analysis modes: stochastic (perturbed obs) and square-root (deterministic)
  - Inflation: multiplicative (state-only), additive (state-only)
  - Localization: Schur (Gaspari–Cohn), 1D periodic, for Identity/Partial observation models
  - File: `Sources/KalmanCore/filters/EnsembleKalmanFilter.swift`
- Combined algorithms:
  - EnKF–EM: windowed run with tight-loop additive M-step (re-assimilates each EM iter)
    - File: `Sources/KalmanCore/estimation/EnKFEM.swift`
  - EnKF–NR: windowed likelihood evaluator scaffold
    - File: `Sources/KalmanCore/estimation/EnKFNewtonRaphson.swift`
- Tests added:
  - `Tests/KalmanCoreTests/EnKFParameterAugmentationTests.swift` (analysis updates θ)
  - `Tests/KalmanCoreTests/EnKFEMBasicTests.swift` (runWindow basics)
  - `Tests/KalmanCoreTests/EnKFEMParameterRecoveryTests.swift` (parameter moves toward truth)

Next actions
- EnKF (augmented):
  - [x] Add additive inflation support and optional simple localization
  - [ ] Add deterministic square-root variant (optional)
- EnKF–EM:
  - [x] Compute E-step statistics from EnKF window (filtered proxies: mean and cov trace)
  - [x] Reuse existing EM M-step for additive σ = θ₀ I to update θ (tight loop with re-assimilation)
  - [x] Add parameter-recovery tests (L96, identity obs)
- EnKF–NR:
  - [ ] Implement windowed innovation-based log-likelihood
  - [ ] Finite-difference gradients/Hessian with cached randomness
  - [ ] Newton loop with line search and bounds

Documentation
- [x] `SECTION_3_1.md` (EnKF with parameter augmentation)
- [x] `SECTION_3_2.md` (EnKF–EM)
- [x] `SECTION_3_3.md` (EnKF–NR, stub)

### Phase 2: Additional Filters (High Priority)
**Estimated effort: 2-3 weeks**

Progress: KF and EKF implemented with tests; UKF and Particle remain.

#### Kalman Filter (Linear)
- Basic linear Kalman filter
- Basis for other filters
- Used in validation

Status / files
- Implemented: `Sources/KalmanCore/filters/KalmanFilter.swift`
- [x] Tests: `Tests/KalmanCoreTests/KalmanFilterTests.swift`
- [ ] Documentation

#### Extended Kalman Filter (EKF)
- Nonlinear system support
- Jacobian computation
- Used in comparisons

Status / files
- Implemented: `Sources/KalmanCore/filters/ExtendedKalmanFilter.swift`
- [x] Tests: `Tests/KalmanCoreTests/ExtendedKalmanFilterTests.swift`
- [ ] Documentation

#### Unscented Kalman Filter (UKF)
- Sigma point approach
- Better nonlinear handling than EKF
- Performance comparison

Status / files
- Implementation target: `Sources/KalmanCore/filters/UnscentedKalmanFilter.swift` (exists as placeholder; fill in)
- [ ] Documentation
- [ ] Tests

Next up (UKF)
- [ ] Sigma point generation (scaled UT): configure α, β, κ; ensure reproducible ordering
- [ ] Time update: propagate sigma points through model.transition; compute mean/cov (with process noise)
- [ ] Measurement update: propagate sigma points through observation; compute innovation stats and cross-covariance; gain
- [ ] Numerical stability: use Cholesky (SPD) for square roots; add small jitter to keep covariances SPD
- [ ] Optional: Square-root UKF variant for better stability/performance
- [ ] Tests: shapes and SPD checks, partial observation case, linear-consistency vs KF, LL finiteness, basic recovery on Lorenz96 small dt
- [ ] Docs: brief SECTION_3_x note and README snippet

### Phase 3: Advanced Features (Medium Priority)
**Estimated effort: 1-2 weeks**

#### Particle Filter
- Non-Gaussian representation
- Resampling methods
- Comparison with Kalman variants

**Files to create:**
- `filters/ParticleFilter.swift` (~350 lines)
- Documentation
- Tests

#### Multi-model Support
- Adaptive model selection
- Model switching framework
- Demonstration examples

**Files to create:**
- `models/MultiModelFramework.swift` (~200 lines)
- Example models

### Phase 4: Examples & Demonstrations (Low Priority, but Important for Adoption)
**Estimated effort: 2-3 weeks**

#### Tutorial Examples
- Step-by-step parameter identification workflow
- Using Section 2.2 (EM) algorithm
- Using Section 2.3 (NR-MLE) algorithm

**Files to create:**
- `examples/TutorialEM.swift` (~150 lines)
- `examples/TutorialNRMLE.swift` (~150 lines)

#### Scientific Applications
- Atmospheric modeling scenario
- Financial time series example
- Biological system identification

**Files to create:**
- `examples/AtmosphericModel.swift` (~200 lines)
- `examples/FinancialData.swift` (~200 lines)
- `examples/BiologicalSystem.swift` (~200 lines)

#### Benchmark Suite
- Performance comparisons (EM vs NR-MLE)
- Convergence analysis
- Scalability testing

Progress
- Initial micro-benchmark implemented: EnKFBench executable target (`swift run enkf-bench`)
  - Files: `benchmarks/EnKFBenchMain.swift`, `benchmarks/README.md`

Planned files (later)
- `benchmarks/PerformanceBenchmark.swift` (~300 lines)
- `benchmarks/ConvergenceAnalysis.swift` (~250 lines)

### Phase 5: Production Readiness
**Estimated effort: 1 week**

- [ ] Comprehensive README update
- [ ] API documentation review
- [ ] Example code verification
- [ ] Test coverage analysis (target: >90%)
- [ ] Performance profiling
- [ ] Memory leak checks
- [ ] Cross-platform testing (macOS, iOS, Linux)
- [ ] License selection and setup
- [ ] Contributing guidelines
- [ ] Changelog preparation

---

## Implementation Priority

### Must-Have for v1.0 (Critical Path)
1. ✅ Section 2.1 - Stochastic framework (DONE)
2. ✅ Section 2.2 - EM algorithm (DONE)
3. ✅ Section 2.3 - NR-MLE algorithm (DONE)
4. Section 3.1-3.3 - Sequential parameter estimation (in progress)
5. ✅ Basic filters (KF, EKF)
6. Comprehensive examples
7. Production readiness

### Should-Have for v1.0 (High Value)
- UKF and Particle Filter
- Tutorial examples
- Performance benchmarks
- Scientific application examples

### Nice-to-Have for v1.0+ (Post-release)
- Advanced localization techniques
- GPU acceleration investigation
- Async/await variants
- Additional stochastic models

---

## Testing Strategy

Each new component should include:

- **Unit tests**: Basic functionality (5-10 tests per class)
- **Integration tests**: With existing components (3-5 tests)
- **Convergence tests**: Verify mathematical properties (2-3 tests)
- **Performance tests**: Benchmark comparisons (1-2 tests)

**Target test coverage for v1.0**: >90%

---

## Documentation Requirements

Each major section should include:

1. **Mathematical documentation** (SECTION_X_Y.md)
   - Problem formulation
   - Algorithm description
   - Key equations
   - Convergence properties

2. **API documentation** (in-code)
   - Class and function descriptions
   - Parameter documentation
   - Return value documentation
   - Usage examples

3. **Usage examples**
   - Basic usage
   - Advanced usage
   - Edge cases
   - Common pitfalls

---

## Timeline Estimate

| Phase | Work Items | Effort | Target Date |
|-------|-----------|--------|-------------|
| ✅ 0 | Sections 2.1-2.3 | Done | Completed |
| 1 | Sequential Est. (3.1-3.3) | 2-3 weeks | In progress |
| 2 | Filters (KF, EKF, UKF) | 2-3 weeks | Week 6 |
| 3 | Advanced (PF, Multi-model) | 1-2 weeks | Week 8 |
| 4 | Examples & Demos | 2-3 weeks | Week 11 |
| 5 | Production Ready | 1 week | Week 12 |

**Total estimated time to v1.0: 10-14 weeks**

---

## Success Criteria for v1.0

- [ ] All sections 2.1-3.3 implemented
- [ ] 95%+ test coverage
- [ ] All algorithms documented with examples
- [ ] No compiler warnings
- [ ] Cross-platform tested (macOS, iOS)
- [ ] README complete with quick-start guide
- [ ] License selected and included
- [ ] Contributing guidelines written
- [ ] Changelog prepared
- [ ] Ready for public release

---

## Post-v1.0 Enhancements (estimated)

### v1.1 (2-3 months later)
- Async/await variants for NR-MLE Hessian computation
- GPU acceleration investigation
- Performance optimizations

### v1.2 (6 months later)
- Machine learning integration (e.g., PyTorch interop)
- Advanced localization techniques
- Time-varying parameter support

### v2.0 (1+ year)
- Bayesian inference framework
- Variational methods
- Deep learning parameter estimation

---

## Key Milestones

### Milestone 1: Core Math (✅ ACHIEVED)
Sections 2.1-2.3 fully implemented with comprehensive tests

### Milestone 2: Sequential Estimation (In Progress)
Section 3.1-3.3: EnKF augmented-state and basic tests in place; next: connect EnKF–EM M-step and implement EnKF–NR likelihood

### Milestone 3: Filter Suite (Target: Week 6)
KF, EKF, UKF all functional and tested

### Milestone 4: Publication Ready (Target: Week 12)
v1.0 released with full documentation and examples

---

## Resource Requirements

### Development
- Estimated 350-400 hours of development work
- Primary developer: 2-3 months full-time equivalent

### Testing
- Automated test suite (Swift Testing framework)
- Cross-platform CI/CD pipeline

### Documentation
- ~3000-4000 lines of markdown documentation
- Code examples and tutorials

---

## Notes for Contributors

When implementing new sections:

1. **Follow existing patterns**
   - Use same configuration struct pattern (see EM, NR-MLE)
   - Use same Result struct pattern for output
   - Keep protocols consistent

2. **Maintain consistency**
   - Error handling approach
   - Naming conventions
   - Documentation style

3. **Prioritize clarity over optimization**
   - This is v1.0, not production HPC code
   - Async/await can be added later
   - Readability > micro-optimizations

4. **Test thoroughly**
   - Unit tests for each component
   - Integration tests with other pieces
   - Convergence property verification

---

## References

- Pulido et al. (2018) - Main paper
- Kalman (1960) - Kalman filter foundations
- Rauch et al. (1965) - RTS smoother
- Dempster et al. (1977) - EM algorithm
- Evensen (1994) - Ensemble Kalman filter
- Nocedal & Wright (2006) - Optimization methods

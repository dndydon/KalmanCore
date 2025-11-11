# Session Notes – 2025-11-11

This document captures today’s changes and next actions.

## Summary of changes
- EnKF (augmented): added additive inflation (state-only) and simple localization (global scalar taper); improved headers/comments.
- EnKF–EM: implemented tight EM loop (re-assimilates each M-step), added diagnostics summary; added parameter-recovery test.
- Shared utilities: introduced Likelihood.gaussianInnovationLogLikelihood; moved matrixInverse/matrixDeterminant into core/MatrixUtils with Accelerate-backed paths and numerical caveats; removed duplicates from EM/NR.
- Filters: implemented KalmanFilter (linear) and ExtendedKalmanFilter (nonlinear) with Config/State/StepResult; added shared Linearization utilities (finite-diff F/H); added 10 new tests (5 each).
- Docs: README “Features” and “Using KF/EKF” sections updated; added SECTION_3_1.md (EnKF), SECTION_3_2.md (EnKF–EM, earlier), SECTION_3_3.md (EnKF–NR stub); added reference headers to KF/EKF files.
- Roadmap: marked KF/EKF done; EnKF inflation/localization done; documentation checklist updated; added “Next up (UKF)” actionable list.
- Tests: now 41 tests across 7 suites, all passing.

## Open tasks (priority)
1) UKF
   - Implement per roadmap “Next up (UKF)” checklist (sigma points, weights, predict/update, SPD safeguards, optional square-root variant)
   - Add 5–8 tests; brief doc note + README snippet
2) EnKF–NR
   - Implement windowed innovation-based log-likelihood
   - Finite-difference gradients/Hessian with cached randomness
   - Newton loop with line search and bounds; expand SECTION_3_3.md beyond stub
3) EnKF (augmented)
   - Optional deterministic square-root variant
4) Docs
   - Add brief API docs for KF/EKF and EnKF in code headers and SECTION_3_x

## Suggested next steps (tomorrow)
- Scaffold UnscentedKalmanFilter.swift (scaled UT): sigma points/weights, predict/update; start with non–square-root version; add tests.
- Implement EnKFNewtonRaphson windowed likelihood evaluator and minimal tests.

## Useful commands
- Build: `swift build`
- Run all tests: `swift test`
- Run specific suites:
  - `swift test --filter KalmanFilter`
  - `swift test --filter ExtendedKalmanFilter`
  - `swift test --filter EnKFEM`

## Pointers
- EnKF: `Sources/KalmanCore/filters/EnsembleKalmanFilter.swift`
- EnKF–EM: `Sources/KalmanCore/estimation/EnKFEM.swift`
- EnKF–NR scaffold: `Sources/KalmanCore/estimation/EnKFNewtonRaphson.swift`
- KF: `Sources/KalmanCore/filters/KalmanFilter.swift`
- EKF: `Sources/KalmanCore/filters/ExtendedKalmanFilter.swift`
- Linearization: `Sources/KalmanCore/core/Linearization.swift`
- Likelihood/Matrix utils: `Sources/KalmanCore/core/Likelihood.swift`, `Sources/KalmanCore/core/MatrixUtils.swift`
- Docs: `SECTION_3_1.md`, `SECTION_3_2.md`, `SECTION_3_3.md`
- Roadmap: `IMPLEMENTATION_ROADMAP.md`

# Session Notes – 2025-11-08

This document captures the context and next actions from today’s work so it’s easy to resume tomorrow.

## Summary of changes
- Added WARP.md with commands and architecture overview; added integration section (Xcode + SwiftPM).
- Implemented augmented-state EnKF scaffold and analysis update:
  - Forecast with parameter evolution: .constant, .randomWalk(Qθ), .ar1(ρ,Qθ)
  - Analysis (stochastic/deterministic) with augmented gain using ensemble anomalies
  - Optional multiplicative inflation on state
  - File: `Sources/KalmanCore/filters/EnsembleKalmanFilter.swift`
- EnKF–EM scaffold: windowed run with placeholder E/M steps
  - File: `Sources/KalmanCore/estimation/EnKFEM.swift`
- EnKF–NR scaffold: windowed likelihood evaluator placeholder
  - File: `Sources/KalmanCore/estimation/EnKFNewtonRaphson.swift`
- Tests added:
  - `Tests/KalmanCoreTests/EnKFParameterAugmentationTests.swift` — verifies analyze() updates θ
  - `Tests/KalmanCoreTests/EnKFEMBasicTests.swift` — exercises EnKFEM.runWindow basics
- Updated roadmap to reflect “in progress” Section 3 and new files/tests:
  - `IMPLEMENTATION_ROADMAP.md`

## Open tasks (priority)
1) EnKF (augmented):
   - [ ] Add additive inflation support and simple (optional) localization
   - [ ] Consider a deterministic square-root variant (optional)
2) EnKF–EM wiring:
   - [ ] Compute E-step stats from EnKF window (filtered/smoothed proxies)
   - [ ] Reuse existing EM M-step for additive σ = θ₀ I to update θ
   - [ ] Add parameter-recovery test (Lorenz96, partial obs)
3) EnKF–NR:
   - [ ] Implement windowed innovation-based log-likelihood
   - [ ] Finite-difference gradients/Hessian with cached randomness
   - [ ] Newton loop with line search and bounds
4) Filters (Phase 2, after Section 3):
   - [ ] Fill in `KalmanFilter.swift`, `ExtendedKalmanFilter.swift`, `UnscentedKalmanFilter.swift`

## Suggested next steps (tomorrow)
- Implement additive inflation and a trivial block localization (state only) in EnKF.analyze().
- Wire EnKFEM.runWindow to call the existing EM M-step (additive noise) using EnKF-derived window stats.
- Add a short parameter-recovery test that should pass once the M-step wiring is in.

## Useful commands
- Build:
  - `swift build`
- Run all tests:
  - `swift test`
- List tests:
  - `swift test --list-tests`
- Run the new tests:
  - `swift test --filter EnKFParameterAugmentationTests`
  - `swift test --filter EnKFEMBasicTests`

## Pointers
- EnKF implementation: `Sources/KalmanCore/filters/EnsembleKalmanFilter.swift`
- EnKF–EM scaffold: `Sources/KalmanCore/estimation/EnKFEM.swift`
- EnKF–NR scaffold: `Sources/KalmanCore/estimation/EnKFNewtonRaphson.swift`
- Tests: `Tests/KalmanCoreTests/EnKFParameterAugmentationTests.swift`, `Tests/KalmanCoreTests/EnKFEMBasicTests.swift`
- Roadmap: `IMPLEMENTATION_ROADMAP.md`

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

# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

Project overview
- KalmanCore is a Swift Package (swift-tools-version 6.1) providing numerical primitives, stochastic state-space models, observation models, and parameter-estimation algorithms (EM and Newton–Raphson MLE). Filters (KF/EKF/UKF/EnKF/Particle) are scaffolded and in development.
- Targets: a single library target KalmanCore and a test target KalmanCoreTests.
- Platforms: macOS 13+, iOS 16+, tvOS 16+, watchOS 9+.
- External deps: none declared in Package.swift; numerics rely on Apple’s Accelerate framework.

Common commands
- Resolve dependencies (first-time or after manifest changes):
  - swift package resolve
- Build (debug/release):
  - swift build
  - swift build -c release
- Run tests (Swift Testing):
  - swift test
  - swift test -v   # verbose
  - swift test --list-tests   # list discovered tests/suites
  - Run a suite: swift test --filter ExpectationMaximizationTests
  - Run a single test: swift test --filter ExpectationMaximizationTests/testConvergenceCriteria
- Lint/format:
  - No linter/formatter config is present in this repo (no SwiftLint/SwiftFormat files found).

High-level architecture
- Core numerics (Sources/KalmanCore/core)
  - Matrix: a value-type matrix with BLAS/LAPACK-backed ops via Accelerate (multiplication, transpose, trace, Frobenius norm, scalar ops, matrix–vector product). Implements small utility ops locally (determinant/inverse) for algorithms; prefer Accelerate for production-grade routines.
  - Ensemble: represents an ensemble of state vectors (mean, covariance, anomalies, inflation, perturbations). Used by ensemble-based filters and model forecasting.
  - RandomUtils: Gaussian noise utilities (including sampling with covariance). Used throughout for process/observation noise.
- Modeling layer
  - StochasticDynamicalSystem protocol (+ BaseStochasticSystem): defines state dimension, parameter dimension, deterministic dynamics M(x, θ), stochastic parameterization σ(x, θ), and a default Euler–Maruyama transition/trajectory/ensemble forecast implementation.
  - Lorenz96Model: concrete chaotic system supporting three stochastic parameterization modes (additive, diagonal, state-dependent) with helpers (standard(), typicalInitialState(), spinUp, climatology stats, Lyapunov approximation).
- Observation models (Sources/KalmanCore/observations)
  - ObservationModel protocol with defaults for noisy observations and applying H(·) to ensembles.
  - LinearObservationModel (y = Hx + v), IdentityObservationModel (H = I), PartialObservationModel (select indices with per-variable noise). Provides H extraction for linear models; nonlinear models are supported via finite-difference linearization where needed.
- Filters (Sources/KalmanCore/filters)
  - Types for KF/EKF/UKF/EnKF/Particle are present and will integrate with the modeling and observation layers. Some implementations are placeholders; parameter estimation currently uses internally implemented Kalman filtering/smoothing steps where needed.
- Estimation (Sources/KalmanCore/estimation)
  - ExpectationMaximization: EM loop with E-step RTS-style smoothing, M-step updates (closed-form for additive-noise case), history tracking, bounds, and convergence checks. Utilities approximate Jacobians and process-noise covariances from σ(x, θ).
  - NewtonRaphsonMLE: direct likelihood optimization using finite-difference gradients/Hessians, Newton steps with Armijo line search, Hessian regularization, and convergence criteria.
- Examples (Sources/KalmanCore/examples)
  - Sample scenarios (e.g., Lorenz96Demo, LinearTrackingExample, NonlinearPendulum) demonstrate usage patterns. These aren’t separate executables; copy/invoke from client apps or tests.
- Tests (Tests/KalmanCoreTests)
  - Uses Swift’s Testing framework (import Testing, @Suite/@Test macros). Current suites validate EM behavior (config defaults, parameter history, convergence, noise robustness, different stochastic types). Use swift test --list-tests to discover names, then --filter to target specific cases.

Important notes from README
- Feature focus areas: matrix ops, random utilities, ensemble structures; stochastic models (Pulido et al. 2018, Sec. 2.1); observation models; filters (in development); estimation (EM, Newton–Raphson MLE); Lorenz96 demos.
- Stated requirements in README mention Swift 5.9+, but the package manifest requires swift-tools-version 6.1 and platforms macOS 13+/iOS 16+. Prefer the manifest as the source of truth when selecting the toolchain.

Integration (Xcode app + Swift script)
- Xcode (SPM):
  - File → Add Packages…
    - Local: select this folder to develop against the local package; or
    - Remote: https://github.com/dndydon/KalmanCore.git (choose a version/branch)
  - Add KalmanCore to your app target, then use in code:
  ```swift path=null start=null
  import KalmanCore
  // Example: run demo
  Lorenz96Demo.runAll()
  ```
- Swift script / small executable (SwiftPM):
  - Create an executable package and add KalmanCore as a dependency (local or Git):
  ```sh path=null start=null
  mkdir MyApp && cd MyApp
  swift package init --type executable
  ```
  - Update Package.swift target/dependency:
  ```swift path=null start=null
  // Package.swift (excerpt)
  dependencies: [
      .package(path: "../KalmanCore")
      // or: .package(url: "https://github.com/dndydon/KalmanCore.git", from: "1.0.0")
  ],
  targets: [
      .executableTarget(name: "MyApp", dependencies: ["KalmanCore"]) 
  ]
  ```
  - In Sources/MyApp/main.swift:
  ```swift path=null start=null
  import KalmanCore

  // Run a demo or minimal usage
  Lorenz96Demo.runAll()
  // Or:
  let model = Lorenz96Model.standard(stochasticType: .additive)
  let x0 = model.typicalInitialState()
  let params = [0.3]
  let traj = model.simulateTrajectory(initialState: x0, parameters: params, dt: 0.01, steps: 10)
  print("First state:", traj.first ?? [])
  ```

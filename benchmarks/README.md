# EnKF Micro-Benchmarks

This executable measures basic performance characteristics of the Ensemble Kalman Filter (EnKF)
in stochastic vs square-root analysis modes, with and without Schur (Gaspari–Cohn) localization,
on Lorenz-96. It is a coarse benchmark intended to spot obvious regressions or large gains.

Usage
- swift run enkf-bench
- Environment variables (optional):
  - ENKF_BENCH_N: state dimension (default 40)
  - ENKF_BENCH_STEPS: number of forecast/analysis steps (default 50)
  - ENKF_BENCH_ENSEMBLES: comma-separated list of ensemble sizes (default 10,20,40)
  - ENKF_BENCH_LOCALIZE: true/false (default false)
  - ENKF_BENCH_SQRT: true/false (default true)

Notes
- These numbers are not meant to be publication-grade. For detailed profiling,
  instrument at the matrix-kernel level and run with a fixed CPU/GPU affinity.

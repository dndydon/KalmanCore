# Section 3.4 — Unscented Kalman Filter (UKF)

This brief note outlines the UKF implementation included in KalmanCore.

Overview
- Model: x_{k+1} = f(x_k, θ, dt) + w_k, y_k = h(x_k) + v_k
- Process noise: w_k ~ N(0, Q_k) with Q_k = dt · σ(x_k, θ) σ(x_k, θ)^T (evaluated at current mean)
- Measurement noise: v_k ~ N(0, R)
- Variant: scaled unscented transform (non–square-root). Cholesky + jitter used for sigma point generation.

Configuration (UnscentedKalmanFilter.Config)
- alpha: spread of sigma points around the mean (typ. 1e-3)
- beta: incorporates prior knowledge about the distribution (2 for Gaussian)
- kappa: secondary scaling parameter (0 by default)
- jitter: small diagonal added to covariances for SPD robustness (default 1e-9)

Algorithm sketch
1) Sigma points: {x^(i)} from (x, P) using scaled UT with λ = α^2 (n + κ) − n
2) Predict: x^(i) → f(x^(i), θ, dt); aggregate mean/cov with weights (w_m, w_c) and add Q
3) Update: propagate sigma points through observation h(·); compute ȳ, innovation covariance S, cross-cov P_xy; K = P_xy S^{-1}
4) Posterior: x ← x + K (y − ȳ), P ← P − K S K^T

Usage
```swift
let model = Lorenz96Model.standard(stochasticType: .additive)
let n = model.stateDimension
let obs = IdentityObservationModel(dimension: n, noiseVariance: 1e-2)
var ukf = UnscentedKalmanFilter(model: model,
                                observationModel: obs,
                                initialState: model.typicalInitialState(),
                                initialCovariance: Matrix.identity(size: n) * 0.2,
                                parameters: [0.3],
                                dt: 0.01)
let y = obs.generateObservation(state: ukf.state.x)
let (_, res) = ukf.step(y: y)
```

Notes
- SPD safeguards: add small jitter to P before Cholesky; retry with larger jitter once if needed
- Square-root UKF is a potential follow-up for improved stability

References
- Julier, S.J., & Uhlmann, J.K. (1997, 2004)
- Wan, E.A., & Van der Merwe, R. (2000)

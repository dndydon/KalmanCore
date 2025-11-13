# Section 3.5 — Particle Filter (SIR)

This brief note outlines the bootstrap particle filter (Sequential Importance Resampling) in KalmanCore.

Overview
- Model: x_{k+1} = f(x_k, θ, dt) + w_k; y_k = h(x_k) + v_k
- Proposal: prior/transition (bootstrap filter): x_i ← f(x_i)
- Weights: w_i ∝ p(y_k | x_i) with Gaussian likelihood N(h(x_i), R)
- Resampling: systematic (default) or multinomial; triggered by ESS/N below threshold

Configuration (ParticleFilter.Config)
- numParticles: number of particles (≥ 1)
- resamplingThreshold: trigger when ESS/N falls below this value (default 0.5)
- resamplingMethod: .systematic (default) or .multinomial

Algorithm sketch
1) Initialization: draw x_i ~ N(x0, P0), set weights w_i = 1/N
2) Predict: x_i ← f(x_i, θ, dt)
3) Update: w_i ← w_i · N(y; h(x_i), R); normalize in log-space
4) ESS = 1 / ∑ w_i^2; if ESS/N < threshold: resample and reset w_i = 1/N

Usage
```swift
let model = Lorenz96Model.standard(stochasticType: .additive)
let n = model.stateDimension
let obs = IdentityObservationModel(dimension: n, noiseVariance: 1e-2)
var pf = ParticleFilter(model: model,
                        observationModel: obs,
                        x0: model.typicalInitialState(),
                        P0: Matrix.identity(size: n) * 0.2,
                        parameters: [0.3],
                        dt: 0.01,
                        config: .init(numParticles: 200, resamplingThreshold: 0.5))
let y = obs.generateObservation(state: pf.state.particles[0])
let (_, res) = pf.step(y: y)
```

Notes
- Likelihood computation uses the ObservationModel’s R and h(·)
- For heavy-tailed noise or non-Gaussian likelihoods, replace the weight update accordingly

References
- Gordon, N.J., Salmond, D.J., & Smith, A.F.M. (1993)
- Doucet, A., de Freitas, N., & Gordon, N. (2001)

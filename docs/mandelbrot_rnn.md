# Mandelbrot RNN

A recurrent neural network trained to approximate Mandelbrot set iteration via BPTT.

## Task

Given c ∈ ℂ, learn to simulate the trajectory z_{t+1} = z_t² + c starting at z_0 = c, predicting the magnitude |z_T| after T steps.

The Mandelbrot set consists of all c for which this trajectory remains bounded (|z_T| ≤ escape_radius = 2).

## Architecture

### Fourier encoding

Raw coordinates are numerically chaotic for BPTT. We lift c = x + yi into a smooth high-dimensional space:

```
encode(c) = [x/S, y/S,
             sin(π x/S), cos(π x/S), sin(π y/S), cos(π y/S),
             sin(2π x/S), cos(2π x/S), sin(2π y/S), cos(2π y/S), ...]
```

where S = 2 (the escape radius / sampling range). This produces enc_dim = 2 + 4K features for K frequency levels (default K=4 → dim 18).

### Decoding (pool estimator)

The final hidden state h_T is decoded back to a complex number by pooling:
- Direct estimate: z₀ = h[0] + i h[1] (scaled by S)
- Per-band estimates: zₖ = atan2(sₓₖ, cₓₖ) / (2^k π) + i atan2(s_yₖ, c_yₖ) / (2^k π) (scaled by S)

Pool: z_out = mean(z₀, z₁, ..., z_K)

Higher frequency bands provide finer phase resolution for small |c| values; the mean combines them.

### RNN cell

A 2-layer tanh FFN:
```
(h_t, x) -> concat -> Linear(2d, hidden) -> tanh -> Linear(hidden, d) -> tanh -> h_{t+1}
```

x = encode(c) is fixed across all T steps; h_0 = x.

### Full model

```
h_0 = encode(c)
for t in 1..T:
    h_t = cell(h_{t-1}, h_0)
output: |decode(h_T)|
```

Default: hidden_dim=128, T=50.

## Training

- **Loss**: MSE between normalised predicted magnitude and normalised reference magnitude (÷ escape_radius)
- **Optimiser**: AdamW (weight_decay=0.01), peak lr=3e-4
- **Schedule**: single-cycle cosine decay with 5% warmup
- **Gradient clipping**: global norm clip = 1.0 (critical for BPTT stability)
- **Data split**: train on Im(c) ≤ 0, test on Im(c) ≥ 0
- **Reference**: z is clamped to escape_radius at each iteration step to prevent overflow

## Design notes

- The train/test split is deliberately hard: |z_T(c̄)| = |z_T(c)| by conjugate symmetry, but the Fourier encoding treats sin(π y/S) as odd in y, so the model cannot trivially generalise. Test error measures whether the model discovers this symmetry.
- h_0 = encode(c) rather than encode(0) ensures the model's initial state matches the reference iteration (z_0 = c).
- Clamping the reference at each step is essential: divergent points grow super-exponentially and produce NaN/Inf targets within ~20 steps without clamping.

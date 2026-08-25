# Hypernetwork (HNN)

A hypernetwork that generates all weights of a target RNN from a noise-driven reparameterised Gaussian encoder, using coordinate-based weight generation.

## Motivation

Direct MLP training for the Mandelbrot set fails to generalise despite low test loss being achievable in principle. The optimisation landscape traps gradient descent in poor local minima. The HNN reframes the problem: instead of optimising network weights directly, we optimise a latent distribution that is decoded into weights through a structured generative process.

The target network is a recurrent cell (matching the standalone RNN architecture) applied for `rnn_num_steps` steps, enabling a fair comparison between the two approaches. Recurrence provides an inductive bias that reduces spectral bias compared to static FFNs.

## Architecture

```
x ~ Uniform[-1, 1]^n_stimulus  (sampled each training step)
       │
       ▼
 GaussianEncoder (reparameterisation trick)
   stimulus = x · σ · τ + μ       σ = exp(0.5 · log_var)
   learnable: μ (32,), log_var (32,)
       │
       ▼
stimulus (32,)
       │
       ▼
 stimulus_to_coord_params   ← learnable: MLP(32 → 128 → 128 → 1153)
       │
       ▼
coord net weights (W1, b1, W2, b2)
       │
       ▼  vmap over all 7,058 parameter embeddings
 coord net (34 → 32 → 1)    ← dynamically instantiated per forward pass
       │
       ▼
flat target params (7,058,)
       │
       ▼
 target RNN cell (36 → 128 → 18, tanh)   ← applied rnn_num_steps=10 times
 h_0 = c_enc;  h_{t+1} = cell(h_t, c_enc)
       │
       ▼
h_T[0]  →  predicted |z_T| / ESCAPE_RADIUS ∈ (-1, 1)
```

## Parameter Embeddings

Each node in the target RNN at layer `l`, position `j` gets a fixed frozen random vector of dimension `node_vec_dim=16`, concatenated with `sin(l)`:

```
node_embed(l, j) = cat(frozen_random_vec_j, [sin(l)])   ∈ R^17
```

Each parameter's embedding is the concatenation of its connected node embeddings:

- **Weight W\_l[i, j]** (connects node `j` at layer `l` to node `i` at layer `l+1`):
  `embed = cat(node_embed(l, j), node_embed(l+1, i))   ∈ R^34`

- **Bias b\_l[i]** (at node `i` in layer `l+1`):
  `embed = cat(node_embed(l+1, i), zeros(17))           ∈ R^34`

These embeddings are computed once at model construction and frozen for the entire training run via `jax.lax.stop_gradient`.

## Gaussian Encoder (reparameterised)

Maps external noise `x ∈ [-1, 1]^n` through a learned Gaussian distribution:

```
σ = exp(0.5 · log_var)
stimulus = x · σ · temperature + μ
```

`log_var` parameterises the variance; `exp(log_var) = σ² > 0` is always positive.

At evaluation (mean prediction), `x = 0` so `stimulus = μ`.

**KL regularisation:** Instead of an entropy penalty, the encoder is regularised by its KL divergence to the standard normal prior `N(0, I)`:

```
KL(N(μ, diag(σ²)) ‖ N(0, I)) = 0.5 · ∑(μ² + σ² − 1 − log σ²)
```

This is always ≥ 0 (zero iff `μ = 0, σ = 1` everywhere) and bounds entropy both above and below — preventing both bandwidth collapse (`σ → 0`) and explosion (`σ → ∞`).

## Coord Net

A small 2-layer MLP (34 → 32 → 1) with `tanh` activation and linear output. Its weights are generated dynamically for each forward pass by `stimulus_to_coord_params`:

```
stimulus  →  MLP(32 → 128 → 128 → 1153)  →  (W1, b1, W2, b2)
```

Applied via `vmap` over all 7,058 parameter embeddings to produce the flat target network parameter vector in one batched matmul pass.

## Target Network (RNN)

A recurrent cell matching the standalone `MandelbrotRNN` architecture:

- **Cell:** `Linear(36 → 128) → tanh → Linear(128 → 18) → tanh`
  Input is `cat(h_t, c_enc) ∈ R^36` (hidden + encoded input)
- **Recurrence:** applied `rnn_num_steps=10` times via `jax.lax.scan`
- **Readout:** headless fixed-index projection — `h_T[0]` (same convention as standalone RNN)

During each forward pass on input `c_enc`:
1. A noise vector `x ~ Uniform[-1, 1]^n` is sampled and passed through the encoder.
2. The full weight-generation pipeline runs (`stimulus → coord net → flat params`).
3. The generated RNN cell is applied recurrently for `rnn_num_steps` steps with `h_0 = c_enc`.
4. `h_T[0]` is returned as the scalar prediction.

## Parameter Counts

| Component | Count |
|---|---|
| Target RNN cell parameters (generated) | 7,058 |
| GaussianEncoder: μ + log\_var | 64 |
| stimulus\_to\_coord\_params (MLP 32→128→128→1153) | 169,473 |
| **Total trainable** | **169,537** |
| param\_embeddings (frozen) | 240,772 |

## Oscillatory Training

Training alternates between two phases in a cycle of length `n_gaussian + 1`:

**Gaussian phase** (`n_gaussian` consecutive steps, default 3):
- Update: `μ`, `log_var` only
- Optimizer: Adam (no weight decay — KL already regularises scale)
- Loss: `MSE(target, pred) + λ_kl · KL(N(μ, σ²) ‖ N(0, I))`

**Rest phase** (1 step):
- Update: `stimulus_to_coord_params`
- Optimizer: AdamW with cosine LR schedule
- Loss: `MSE(target, pred) + λ_target_decay · mean(flat_target_params²)`

The `λ_target_decay` term regularises the L2 norm of the generated target network weights (distinct from the AdamW weight decay on `stim_ffn` weights). This prevents the coord net from generating arbitrarily large weights.

## Training Configuration (defaults)

| Param | Value |
|---|---|
| `n_stimulus` | 32 |
| `node_vec_dim` | 16 |
| `coord_net_hidden` | 32 |
| `stim_ffn_hidden` | 128 |
| `stim_ffn_depth` | 2 |
| `target_is_rnn` | True |
| `rnn_hidden_dim` | 128 |
| `rnn_depth` | 2 |
| `rnn_num_steps` | 10 |
| `temperature` | 1.0 |
| `n_steps` | 20,000 |
| `batch_size` | 64 |
| `lr` / `lr_gauss` | 1e-3 |
| `weight_decay` | 0.01 |
| `lambda_kl` | 0.01 |
| `lambda_target_decay` | 1e-3 |
| `n_gaussian` | 3 |

## Data Split

Identical to other experiments:
- **Train:** `Im(c) ≤ 0` (lower half-plane), 10,000 points
- **Test:** `Im(c) ≥ 0` (upper half-plane), 2,000 points

The test split tests conjugate symmetry generalisation.

## Running

```bash
god-hnn
```

Outputs to `data/mandel/hnn/`.

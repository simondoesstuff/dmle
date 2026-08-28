# Hypernetwork (HNN)

A hypernetwork that generates all weights of a target RNN from a noise stimulus using coordinate-based weight generation.

## Motivation

Direct MLP training for the Mandelbrot set fails to generalise despite low test loss being achievable in principle. The optimisation landscape traps gradient descent in poor local minima. The HNN reframes the problem: instead of optimising network weights directly, we optimise a stimulus FFN that decodes noise into weights through a structured generative process.

The target network is a recurrent cell (matching the standalone RNN architecture) applied for `rnn_num_steps` steps, enabling a fair comparison between the two approaches.

## Architecture

```
x ~ Uniform[-1, 1]^n_stimulus  (sampled each training step)
       │
       ▼
 stimulus_to_coord_params   ← learnable: MLP(32 → 128 → 128 → n_coord_params)
       │
       ▼
coord net weights (W1, b1, W2, b2)
       │
       ▼  vmap over all n_params parameter embeddings
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

The stimulus `x` is passed directly into the stimulus FFN — there is no learned encoder. Different `x` values generate different target networks.

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

These embeddings are computed once at model construction and frozen via `jax.lax.stop_gradient`.

## Coord Net

A small 2-layer MLP (34 → 32 → 1) with `tanh` activation and linear output. Its weights are generated dynamically for each forward pass by `stimulus_to_coord_params`:

```
stimulus  →  MLP(32 → 128 → 128 → n_coord_params)  →  (W1, b1, W2, b2)
```

Applied via `vmap` over all parameter embeddings to produce the flat target network parameter vector.

## Target Network (RNN)

A recurrent cell matching the standalone `MandelbrotRNN` architecture:

- **Cell:** `Linear(36 → 128) → tanh → Linear(128 → 18) → tanh`
  Input is `cat(h_t, c_enc) ∈ R^36`
- **Recurrence:** applied `rnn_num_steps=10` times via `jax.lax.scan`
- **Readout:** headless fixed-index projection — `h_T[0]`

## Loss

Per training step, `n_mc_train` independent noise vectors `x_1, ..., x_k` are drawn. Each generates a different target network, but all are evaluated on the same (c_enc, target) batch for a fair comparison:

```
loss = mean_k[ MSE(targets, preds_k) ] + λ_target_decay · mean_k[ mean(||params_k||²) ]
```

The `λ_target_decay` term regularises the L2 norm of generated target weights.

## Parameter Counts

| Component | Count |
|---|---|
| Target RNN cell parameters (generated) | 7,058 |
| `stimulus_to_coord_params` (MLP 32→128→128→1153) | 169,473 |
| **Total trainable** | **169,473** |
| `param_embeddings` (frozen) | 240,772 |

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
| `n_steps` | 150,000 |
| `batch_size` | 64 |
| `n_mc_train` | 8 |
| `n_mc_eval` | 32 |
| `lr` | 3e-4 |
| `warmup_frac` | 0.3 |
| `weight_decay` | 0.01 |
| `lambda_target_decay` | 1e-3 |

## Data Split

Identical to other experiments:
- **Train:** `Im(c) ≤ 0` (lower half-plane), 10,000 points
- **Test:** `Im(c) ≥ 0` (upper half-plane), 2,000 points

## Running

```bash
god-hnn
```

Outputs to `data/mandel/hnn/`. Auto-detects the latest checkpoint and resumes from there.

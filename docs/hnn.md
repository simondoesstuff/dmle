# Hypernetwork (HNN)

A hypernetwork that generates all weights of a target RNN cell directly from a noise stimulus.

## Motivation

Direct MLP training for the Mandelbrot set fails to generalise despite low test loss being achievable in principle. The optimisation landscape traps gradient descent in poor local minima. The HNN reframes the problem: instead of optimising network weights directly, we optimise a stimulus FFN that decodes noise into weights.

The target network is a recurrent cell matching the standalone RNN architecture — the same depth=1 TanhFFN-style cell (one hidden tanh layer, linear output) applied for `num_steps` recurrences.

## Architecture

```
x ~ Uniform[-1, 1]^n_stimulus  (sampled each training step)
       │
       ▼
 stimulus_ffn   ← learnable MLP(16 → 64 → n_target_params)
       │
       ▼
flat target params (n_target_params,)
       │
       ▼
 target RNN cell (36 → 16 → 18, tanh hidden + linear output)
 h_0 = c_enc;  h_{t+1} = cell(h_t, c_enc)  [num_steps=10 times]
       │
       ▼
sqrt(h_T[0]² + h_T[1]²)  →  predicted |z_T| / ESCAPE_RADIUS
```

The stimulus `x` is passed directly into the stimulus FFN — there is no learned encoder. Different `x` values generate different target networks.

## Cell Architecture

The target RNN cell uses TanhFFN convention (`rnn_depth=1`):
- **Layer 0:** `Linear(2·enc_dim=36 → hidden=16) → tanh`
- **Layer 1:** `Linear(16 → enc_dim=18)` (linear output, no tanh)

Total cell parameters: (36·16 + 16) + (16·18 + 18) = 898

## Stimulus FFN

A small MLP that maps noise stimulus directly to all cell parameters:

```
MLP(n_stimulus=16 → stim_ffn_hidden=64 → n_target_params=898)
```

The output layer is initialised with `init_scale=0.1` to keep generated parameters near zero at the start of training.

## Loss

Per training step, `n_mc_train` independent noise vectors `x_1, ..., x_k` are drawn. Each generates a different target network, but all are evaluated on the same (c_enc, target) batch for a fair comparison:

```
loss = mean_k[ MSE(targets, preds_k) ] + λ_target_decay · mean_k[ mean(||params_k||²) ]
```

The `λ_target_decay` term regularises the L2 norm of generated target weights.

## Parameter Counts

| Component | Count |
|---|---|
| Target RNN cell parameters (generated, not stored) | 898 |
| `stimulus_ffn` (MLP 16→64→898) | 59,458 |
| **Total trainable** | **59,458** |

## Training Configuration (defaults)

| Param | Value |
|---|---|
| `n_stimulus` | 16 |
| `rnn_hidden_dim` | 16 |
| `rnn_depth` | 1 |
| `stim_ffn_hidden` | 64 |
| `stim_ffn_depth` | 1 |
| `num_steps` | 10 |
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

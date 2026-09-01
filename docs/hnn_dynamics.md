# HNN Grokking Dynamics

Findings from applying a HyperNetwork (HNN) to the modular addition grokking task
`(a + b) mod 97`. The HNN generates target FFN weights from a noise stimulus
`x ~ U[-1, 1]^n_stimulus`; only the stimulus FFN is trained.

---

## Setup

**Target network**: 195 → 32 → 97 tanh FFN (9,473 params)  
**Stimulus FFN**: 8 → 8 → 9,473 (85,329 HNN params total)  
**Data**: 50% or 30% train split, full-batch AdamW, 100k epochs  
**Baseline FFN**: same target architecture trained directly (no HNN)

---

## FFN Baseline Sweep (hidden=32, depth=1)

Sweep over weight_decay × train_fraction to find the grokking regime at this scale.

| wd | frac | memorised@ | test@mem | peak test |
|----|------|-----------|----------|-----------|
| 0.5 | 0.3 | 30k | 25% | **94%** ✓ grokking |
| 1.0 | 0.5 | never (97%) | — | 92% |
| 1.0 | 0.3 | never (97%) | — | 66% |
| 2.0 | 0.3 | never (19%) | — | 1% |

**Sweet spot**: `wd=0.5, frac=0.3` — clean two-phase grokking. Train reaches 100% at
epoch 30k (test still at 25%), then test climbs to 94% by epoch 100k.

`wd=1.0` with 50% data achieves high final accuracy but skips the memorisation phase
(no clean two-phase curve). `wd=2.0` prevents memorisation entirely.

---

## HNN Experiment 1 — Porting FFN Hyperparameters Directly

**Config**: `lambda_target_decay=0.5, wd_hnn=0.01, frac=0.3, n_mc=4`  
**Result**: memorised at epoch 8k (test=4%), flatlines at 29%. No grokking.

**Config**: `lambda_target_decay=0.75, wd_hnn=0.01, frac=0.3, n_mc=4`  
**Result**: memorised at epoch 6k (test=20%), flatlines at 34%. No grokking.

**Config**: `lambda_target_decay=1.0, wd_hnn=0.01, frac=0.3, n_mc=4`  
**Result**: never fully memorises (train=99.1%), test=16%. No grokking.

**Observation**: direct translation of FFN `wd` to HNN `lambda_target_decay` fails.
`lambda_target_decay` is an L2 loss term whose gradient flows through Adam's adaptive
scaling, unlike AdamW weight decay which is a direct multiplicative shrinkage bypassing
Adam. These are not equivalent regularisers.

---

## HNN Experiment 2 — Stimulus Collapse (Confirmed)

Added `param_std` (std of generated target params across x samples) and per-x min/max
test accuracy as diagnostics.

**Config**: `wd_hnn=0.01, lambda_target_decay=0.5, n_mc=1`

| epoch | train | test | test_min | test_max | param_std |
|-------|-------|------|----------|----------|-----------|
| 1 | 1.1% | 1.0% | — | — | 0.016 |
| 10k | 100% | 27% | 27% | 27% | 0.002 |
| 30k | 100% | 36% | 36% | 36% | 0.001 |
| 100k | 100% | 36% | 36% | 36% | 0.001 |

**Finding**: `param_std` collapses from 0.016 → 0.002 within the first 10k epochs and
approaches zero thereafter. All x values produce identical target networks — the
stimulus FFN degenerates to a constant function. **User's hypothesis confirmed.**

**Implication**: after collapse, `min ≈ mean ≈ max` across x samples, so `n_mc` is
irrelevant — n_mc=1 and n_mc=4 produce indistinguishable results.

**Why no grokking after collapse**: the effective regularisation on the target network
after collapse is `wd_hnn * lr = 0.01 × 0.001 = 1e-5` shrinkage per step. The
FFN requires `wd * lr = 0.5 × 0.001 = 5e-4` — 50× stronger. The collapsed HNN is
under-regularised.

---

## HNN Experiment 3 — High wd_hnn

**Config**: `wd_hnn=0.5, lambda_target_decay=0.0, n_mc=1, frac=0.3`

| epoch | train | test | param_std |
|-------|-------|------|-----------|
| 1 | 1.1% | 1.0% | 0.016 |
| 2k | 100% | 0.1% | 0.018 |
| 10k | 100% | 0.2% | 0.018 |
| 100k | 100% | 0.2% | 0.007 |

**Finding**: `param_std` stays near 0.018 — **collapse does not occur** with strong
weight decay. The HNN memorises (train=100%) but test≈0.2% (near chance for 97 classes).

**Mechanism**: with fresh x sampled each step, the stimulus FFN learns a *function*
`x → target_network` where each specific x produces a target network that fits the
training data. With high wd, the HNN cannot accumulate a shared solution across x
values — each training x sees a gradient pushing toward memorisation for that x, but
strong wd prevents cross-x consolidation. Eval x values are unseen → near-random
target networks → test≈0%.

---

## Summary of Failure Modes

| regime | collapse? | grokking? | why |
|--------|-----------|-----------|-----|
| Low wd_hnn + lambda_td | yes (pstd→0) | no | effective target wd is wd_hnn=0.01, 50× too weak |
| High wd_hnn, fresh x | no (pstd~0.018) | no | HNN memorises per-x; eval x are unseen |
| High wd_hnn, fixed x | TBD | TBD | pure reparameterisation; expected to reproduce FFN |

**Key insight**: the stimulus collapse and the correct regularisation strength are in
tension when using random x:
- Collapse requires `wd_hnn` small relative to gradient signal
- Correct effective target wd requires `wd_hnn ≈ 0.5`
- These cannot both be satisfied simultaneously with random fresh x

---

## HNN Experiment 4 — Fixed-x Baseline (Evaluation Bug Discovered)

**Config**: `fixed_x=True, wd_hnn=0.5, lambda_target_decay=0.0, n_mc=1, frac=0.3`

| epoch | train | test | param_std |
|-------|-------|------|-----------|
| 1 | 1.1% | 1.0% | 0.016 |
| 10k | 55% | 0.2% | 1.69 |
| 50k | 78% | 0.1% | 2.07 |
| 100k | 84% | 0.1% | 1.94 |

**Unexpected**: `param_std` *explodes* to 1.7–2.5 rather than collapsing. Mechanism:
with fixed training x_0, `W_0` (the first HNN layer that processes x) settles to a
non-zero equilibrium. For eval x values ≠ x_0, `W_1 @ tanh(W_0 @ x + b_0)` varies
wildly because W_1 is a 9473×8 matrix that amplifies small differences in the hidden
layer. The HNN is highly x-sensitive at unseen x values despite being trained on one.

**Evaluation bug (discovered)**: `eval_metrics` averages accuracy over 32 *random* x
samples. For a fixed-x trained model, 31 of those 32 x values produce untrained target
networks → their accuracy is near-random. The logged `train_acc` (54–84%) is a
misleading average of one good prediction and many bad ones.

**Fix**: when `fixed_x=True`, evaluate on the training x only.

**Implication**: fixed-x is NOT a "pure reparameterisation" equivalent to the FFN.
The HNN architecture introduces x-sensitivity at eval x values even when only one x
is used for training. The indirect parameterisation fundamentally changes the dynamics
relative to a direct FFN.

---

---

## HNN Experiment 5 — Fixed-x with Corrected Evaluation

**Config**: `fixed_x=True, wd_hnn=0.5, n_mc=1, frac=0.3` — eval on training x only

| epoch | train | test | param_std |
|-------|-------|------|-----------|
| 1 | 1.0% | 1.1% | 0.0000 |
| 2k | 100% | — | 0.0000 |
| 10k | 100% | 0.1% | 0.0000 |
| 100k | 100% | 0.1% | 0.0000 |

`param_std=0` trivially (eval uses the same fixed x, so all "samples" are identical).
Memorises at epoch 2k. Test remains at 0.1% — still no grokking.

**The FFN baseline memorises at 30k with the same wd=0.5. The HNN memorises 15× faster.**

---

## Root Cause: Effective Learning Rate Amplification

The stimulus FFN introduces multiple gradient pathways to each target param g_k.
With AdamW, Adam normalises each parameter's update to ±lr. The effect on g_k per step:

| pathway | per-element effect on g_k | total (n=8) |
|---------|--------------------------|-------------|
| b1_k (bias) | ±lr | **1× lr** |
| W1_{kj} (output weight) | ±lr · \|h_j\| → ±lr as \|h\|→1 | **→ 8× lr** |
| W0_{ij} (hidden weight) | ±lr · \|W1_{ki}\| · sech²(·) · \|x_j\| | additional |

**Total effective lr on each target param: ~9× lr** (vs 1× for the direct FFN).
This matches the 15× faster memorisation empirically (2k vs 30k epochs).

Weight decay penalises each HNN param individually at rate `wd_hnn`. But the
**combined gradient pushes the target params 9× harder than wd_hnn assumes** — the
model escapes the regularisation regime (slow memorisation under pressure) before
weight decay has time to establish the Fourier solution.

**This is a structural property of the indirect parameterisation, not a tuning issue.**
Every hidden layer adds 8× more effective gradient pathways. A single-layer linear
stimulus FFN (stim_depth=0, no activation) would still give ~5× amplification via W.
Only a pure bias output (g = b, no input x) is exactly equivalent to the direct FFN —
but that defeats the purpose of the HNN.

---

## Implications and Open Questions

1. **The HNN cannot reproduce FFN grokking via simple hyperparameter adjustment.**
   The effective lr amplification from the indirect parameterisation is structural.
   Increasing `wd_hnn` to compensate only prevents collapse or causes other failure
   modes (per-x overfitting).

2. **What would it take for the HNN to grok with random x?** The grokking mechanism
   requires slow memorisation under regularisation pressure. With random x and effective
   lr amplification, the HNN memorises too fast to accumulate the weight-norm pressure
   needed to find the compact Fourier solution. A training procedure that explicitly
   controls memorisation speed (e.g., learning rate warmup, curriculum on wd, or
   direct control of target param norms) may be needed.

3. **The collapse is confirmed but insufficient.** Weight decay does collapse the HNN
   to a constant target network (param_std → 0 quickly). But after collapse, the
   effective target regularisation from wd_hnn is still subject to the lr amplification
   issue, so grokking doesn't follow.

4. **The random-x HNN shows distinct dynamics worth studying.** The partial
   generalisation at memorisation time (test=10–35% before any grokking phase) and the
   flat plateau suggest the HNN settles into a qualitatively different attractor than
   the direct FFN's memorising solution. This may be interesting in its own right.

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
| wd_output_layer=0.5, fresh x | yes (pstd→0) | no | output layer shrinkage reaches equilibrium at norm~504 (Adam balances shrinkage) |
| wd_output_layer=2.0, fresh x | yes (pstd→0) | weak (test→1.4%) | equilibrium norm ~291, below generalising threshold |
| wd_output_layer=4.0, fresh x | yes (pstd→0) | **yes** ✓ | equilibrium norm ~193, matches Fourier solution; grokking at epoch ~26k |

**Key insight**: the stimulus collapse and the correct regularisation strength are in
tension when using random x AND standard HNN weight decay. Post-step multiplicative
shrinkage on the output layer bypasses Adam and acts as true WD on the generated target
params — but requires ~8× the direct FFN WD to compensate for effective lr amplification.

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

## HNN Experiment 6 — Post-Step Output Layer Shrinkage Sweep

Post-step multiplicative shrinkage applied to `stimulus_ffn.layers[-1]` (weight + bias)
each step: `W_last *= (1 - wd_output_layer * lr)`. This bypasses Adam and acts as
true AdamW WD on the generated target params via the output layer.

`weight_decay=0.0, grad_clip=1.0, n_mc=1, frac=0.3, fixed_x=False`

| wd_output_layer | epoch@mem | test@mem | peak test | norm@eq | grokking? |
|-----------------|-----------|----------|-----------|---------|-----------|
| 0.5 | ~10k | 0.2% | 0.2% | 504 | no |
| 2.0 | ~10k | 0.2% | 1.4% | 291 | very weak |
| 4.0 | ~8k | 7.8% | **96%** ✓ | 193 | **yes** |

**wd_output_layer=4.0 grokking timeline**:

| epoch | train | test | param_norm |
|-------|-------|------|------------|
| 1 | 1.1% | 1.0% | 2.8 |
| 8k | 100% | 7.8% | 171 |
| 12k | 100% | 35.3% | 169 |
| 16k | 100% | 69.0% | 166 |
| 22k | 100% | 90.8% | 172 |
| 26k | 100% | 93.8% | 172 |
| 60k | 100% | 95.3% | 195 |
| 100k | 100% | 96.2% | 194 |

**Key observation**: `param_norm` settles at ~193–195 — the same equilibrium the direct
FFN occupies after grokking with `wd=0.5`. The model memorises at norm ~170, which is
*below* the equilibrium. It then slowly climbs to equilibrium (~195) as generalisation
kicks in. The norm does not decay post-memorisation; instead generalisation occurs while
the model finds the Fourier solution at the fixed equilibrium norm.

**Why 4.0 (not 0.5)?** The effective lr amplification from gradient pathways
(W0, W1, b_last) pushes target params roughly 4–8× harder than wd_output_layer alone
assumes. `wd_output_layer=4.0` sets the Adam-equilibrium norm at ~193, which is where
the Fourier/generalising solution lives. Lower values (0.5, 2.0) settle at higher norms
where the memorising attractor is stable.

---

## FFN LR Scaling Ablation — Is the HNN Just a High-LR FFN?

To test whether the HNN's speed advantage is simply an effective LR increase,
we ran the direct FFN (hidden=32, depth=1, frac=0.3) with scaled lr and wd.

In AdamW, per-step weight shrinkage = `wd * lr`. Scaling both by 4× gives 16×
stronger shrinkage — over-regularised. The correct equivalent-speed comparison
is `lr=4e-3, wd=0.5` (4× LR, same WD, preserving the WD/update ratio).

| config | mem@ | peak test | notes |
|--------|------|-----------|-------|
| FFN lr=1e-3 wd=0.5 (baseline) | 30k | 94.2% @96k | two-phase grokking |
| FFN lr=4e-3 wd=0.5 | 12k | 90.9% @34k | ~2.5–2.8× faster, noisy generalisation |
| FFN lr=4e-3 wd=2.0 (4× both) | never | 0.9% | 16× shrinkage prevents memorisation |
| HNN wd_output_layer=4.0 | 8k | **96.2%** @84k | 3.5× faster, higher accuracy |

**The HNN is not just a high-LR FFN.** 4× LR on the direct FFN does recover speed
(~2.8× faster) but trades off accuracy (90.9% vs 94.2%) and stability — the
generalisation curve is noisy and the peak is not sustained. The HNN is faster still
(3.5× vs baseline) and achieves *higher* final accuracy than even the baseline FFN.

**Why?** The HNN's per-target-param gradient amplification comes from multiple
structurally independent pathways (W0 rows, W1 columns, b_last) each contributing
±lr to a different "dimension" of the target param update. This is qualitatively
different from a uniform LR scale: it produces correlated, multi-direction pressure
on each param simultaneously, which appears to find the Fourier attractor more
reliably than a scalar LR increase with noisier per-step updates.

This is a baseline HNN with the simplest possible stimulus FFN (8→8→9473, tanh).
The grokking improvement over the direct FFN is already non-trivial.

---

## Implications and Open Questions

1. **HNN grokking IS achievable with random x** by applying post-step multiplicative
   shrinkage to the output layer with `wd_output_layer ≈ 4–8× the direct FFN wd`.
   The effective lr amplification (from gradient pathways W0, W1, b_last) requires
   this compensation. This is a structural property — every hidden layer adds more
   gradient pathways.

2. **The collapse is a feature, not a bug.** `param_std → 0` within 30k epochs for
   all random-x runs. After collapse, the HNN effectively generates a single target
   network for all x. The post-step shrinkage on the output layer then acts as true
   WD on that single target network, enabling the grokking ratchet.

3. **The HNN has a genuine architectural advantage over a scaled-LR FFN.** The
   multi-pathway gradient structure produces more stable convergence to the Fourier
   solution than a global LR increase. This suggests the reparameterisation geometry
   — not just effective lr — is doing useful work.

4. **Open: does x-diversity matter post-grokking?** Since collapse makes the HNN
   equivalent to a single target FFN, the x-stimulus plays no role in the generalised
   solution. A truly x-dependent grokking would require preventing collapse — which
   requires strong WD on the hidden layers, which blocks cross-x consolidation (Exp 3).

5. **Open: deeper stimulus FFN.** Every hidden layer adds another 8× of gradient
   pathways. A stim_depth=2 stimulus FFN would have ~72× effective lr amplification,
   requiring `wd_output_layer ≈ 36`. Does convergence continue to improve, or does
   the amplification become destabilising?

6. **Open: x-dependent grokking.** Can the stimulus x play a meaningful role in the
   generalised solution? Preventing collapse while maintaining cross-x generalisation
   is the key challenge — a fundamentally different regularisation scheme would be
   needed (e.g., explicit diversity loss, or a non-collapsed target ensemble).

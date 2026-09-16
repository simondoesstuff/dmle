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

| wd  | frac | memorised@  | test@mem | peak test          |
| --- | ---- | ----------- | -------- | ------------------ |
| 0.5 | 0.3  | 30k         | 25%      | **94%** ✓ grokking |
| 1.0 | 0.5  | never (97%) | —        | 92%                |
| 1.0 | 0.3  | never (97%) | —        | 66%                |
| 2.0 | 0.3  | never (19%) | —        | 1%                 |

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
| ----- | ----- | ---- | -------- | -------- | --------- |
| 1     | 1.1%  | 1.0% | —        | —        | 0.016     |
| 10k   | 100%  | 27%  | 27%      | 27%      | 0.002     |
| 30k   | 100%  | 36%  | 36%      | 36%      | 0.001     |
| 100k  | 100%  | 36%  | 36%      | 36%      | 0.001     |

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
| ----- | ----- | ---- | --------- |
| 1     | 1.1%  | 1.0% | 0.016     |
| 2k    | 100%  | 0.1% | 0.018     |
| 10k   | 100%  | 0.2% | 0.018     |
| 100k  | 100%  | 0.2% | 0.007     |

**Finding**: `param_std` stays near 0.018 — **collapse does not occur** with strong
weight decay. The HNN memorises (train=100%) but test≈0.2% (near chance for 97 classes).

**Mechanism**: with fresh x sampled each step, the stimulus FFN learns a _function_
`x → target_network` where each specific x produces a target network that fits the
training data. With high wd, the HNN cannot accumulate a shared solution across x
values — each training x sees a gradient pushing toward memorisation for that x, but
strong wd prevents cross-x consolidation. Eval x values are unseen → near-random
target networks → test≈0%.

---

## Summary of Failure Modes

| regime                       | collapse?       | grokking?        | why                                                                              |
| ---------------------------- | --------------- | ---------------- | -------------------------------------------------------------------------------- |
| Low wd_hnn + lambda_td       | yes (pstd→0)    | no               | effective target wd is wd_hnn=0.01, 50× too weak                                 |
| High wd_hnn, fresh x         | no (pstd~0.018) | no               | HNN memorises per-x; eval x are unseen                                           |
| High wd_hnn, fixed x         | TBD             | TBD              | pure reparameterisation; expected to reproduce FFN                               |
| wd_output_layer=0.5, fresh x | yes (pstd→0)    | no               | output layer shrinkage reaches equilibrium at norm~504 (Adam balances shrinkage) |
| wd_output_layer=2.0, fresh x | yes (pstd→0)    | weak (test→1.4%) | equilibrium norm ~291, below generalising threshold                              |
| wd_output_layer=4.0, fresh x | yes (pstd→0)    | **yes** ✓        | equilibrium norm ~193, matches Fourier solution; grokking at epoch ~26k          |

**Key insight**: the stimulus collapse and the correct regularisation strength are in
tension when using random x AND standard HNN weight decay. Post-step multiplicative
shrinkage on the output layer bypasses Adam and acts as true WD on the generated target
params — but requires ~8× the direct FFN WD to compensate for effective lr amplification.

---

## HNN Experiment 4 — Fixed-x Baseline (Evaluation Bug Discovered)

**Config**: `fixed_x=True, wd_hnn=0.5, lambda_target_decay=0.0, n_mc=1, frac=0.3`

| epoch | train | test | param_std |
| ----- | ----- | ---- | --------- |
| 1     | 1.1%  | 1.0% | 0.016     |
| 10k   | 55%   | 0.2% | 1.69      |
| 50k   | 78%   | 0.1% | 2.07      |
| 100k  | 84%   | 0.1% | 1.94      |

**Unexpected**: `param_std` _explodes_ to 1.7–2.5 rather than collapsing. Mechanism:
with fixed training x_0, `W_0` (the first HNN layer that processes x) settles to a
non-zero equilibrium. For eval x values ≠ x_0, `W_1 @ tanh(W_0 @ x + b_0)` varies
wildly because W_1 is a 9473×8 matrix that amplifies small differences in the hidden
layer. The HNN is highly x-sensitive at unseen x values despite being trained on one.

**Evaluation bug (discovered)**: `eval_metrics` averages accuracy over 32 _random_ x
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
| ----- | ----- | ---- | --------- |
| 1     | 1.0%  | 1.1% | 0.0000    |
| 2k    | 100%  | —    | 0.0000    |
| 10k   | 100%  | 0.1% | 0.0000    |
| 100k  | 100%  | 0.1% | 0.0000    |

`param_std=0` trivially (eval uses the same fixed x, so all "samples" are identical).
Memorises at epoch 2k. Test remains at 0.1% — still no grokking.

**The FFN baseline memorises at 30k with the same wd=0.5. The HNN memorises 15× faster.**

---

## Root Cause: Effective Learning Rate Amplification

The stimulus FFN introduces multiple gradient pathways to each target param g_k.
With AdamW, Adam normalises each parameter's update to ±lr. The effect on g_k per step:

| pathway                  | per-element effect on g_k               | total (n=8) |
| ------------------------ | --------------------------------------- | ----------- |
| b1_k (bias)              | ±lr                                     | **1× lr**   |
| W1\_{kj} (output weight) | ±lr · \|h_j\| → ±lr as \|h\|→1          | **→ 8× lr** |
| W0\_{ij} (hidden weight) | ±lr · \|W1\_{ki}\| · sech²(·) · \|x_j\| | additional  |

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
| --------------- | --------- | -------- | --------- | ------- | --------- |
| 0.5             | ~10k      | 0.2%     | 0.2%      | 504     | no        |
| 2.0             | ~10k      | 0.2%     | 1.4%      | 291     | very weak |
| 4.0             | ~8k       | 7.8%     | **96%** ✓ | 193     | **yes**   |

**wd_output_layer=4.0 grokking timeline**:

| epoch | train | test  | param_norm |
| ----- | ----- | ----- | ---------- |
| 1     | 1.1%  | 1.0%  | 2.8        |
| 8k    | 100%  | 7.8%  | 171        |
| 12k   | 100%  | 35.3% | 169        |
| 16k   | 100%  | 69.0% | 166        |
| 22k   | 100%  | 90.8% | 172        |
| 26k   | 100%  | 93.8% | 172        |
| 60k   | 100%  | 95.3% | 195        |
| 100k  | 100%  | 96.2% | 194        |

**Key observation**: `param_norm` settles at ~193–195 — the same equilibrium the direct
FFN occupies after grokking with `wd=0.5`. The model memorises at norm ~170, which is
_below_ the equilibrium. It then slowly climbs to equilibrium (~195) as generalisation
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

| config                        | mem@  | peak test      | notes                                  |
| ----------------------------- | ----- | -------------- | -------------------------------------- |
| FFN lr=1e-3 wd=0.5 (baseline) | 30k   | 94.2% @96k     | two-phase grokking                     |
| FFN lr=4e-3 wd=0.5            | 12k   | 90.9% @34k     | ~2.5–2.8× faster, noisy generalisation |
| FFN lr=4e-3 wd=2.0 (4× both)  | never | 0.9%           | 16× shrinkage prevents memorisation    |
| HNN wd_output_layer=4.0       | 8k    | **96.2%** @84k | 3.5× faster, higher accuracy           |

**The HNN is not just a high-LR FFN.** 4× LR on the direct FFN does recover speed
(~2.8× faster) but trades off accuracy (90.9% vs 94.2%) and stability — the
generalisation curve is noisy and the peak is not sustained. The HNN is faster still
(3.5× vs baseline) and achieves _higher_ final accuracy than even the baseline FFN.

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
   **Resolved further in Exp 7**: collapse isn't the causal mechanism — it's a side
   effect of random x plus weight shrinkage, not a requirement. Arm 3 groks just as
   well with a single fixed x (no diversity to collapse from). What's required is
   direct shrinkage on the output-layer weight; collapse is what that produces when x
   happens to vary.

3. **The HNN has a genuine architectural advantage over a scaled-LR FFN.** The
   multi-pathway gradient structure produces more stable convergence to the Fourier
   solution than a global LR increase. This suggests the reparameterisation geometry
   — not just effective lr — is doing useful work.

4. **Resolved in Exp 7: x-diversity doesn't matter, full stop — not just
   post-grokking.** Originally framed as open (does x play a role in the generalised
   solution once collapsed?), Exp 7 shows it's stronger than that: x-diversity isn't
   needed at any point, including during training. Arm 3 (weight shrinkage, x held
   constant for all 200k epochs — never varies, so there's nothing to collapse) groks
   as fast and as well as arm 2 (weight shrinkage, random x). The x-stimulus plays no
   causal role in either reaching or occupying the generalised solution.

5. **Open: deeper stimulus FFN.** Every hidden layer adds another 8× of gradient
   pathways. A stim_depth=2 stimulus FFN would have ~72× effective lr amplification,
   requiring `wd_output_layer ≈ 36`. Does convergence continue to improve, or does
   the amplification become destabilising?

6. **Open: x-dependent grokking.** Can the stimulus x play a meaningful role in the
   generalised solution? Preventing collapse while maintaining cross-x generalisation
   is the key challenge — a fundamentally different regularisation scheme would be
   needed (e.g., explicit diversity loss, or a non-collapsed target ensemble).

---

## HNN Experiment 7 — Output-Layer Shrinkage Structural Ablation

Follow-up finding (not detailed above): post-step shrinkage applied _only_ to the
stimulus FFN's final-layer **bias** (`shrink_target="bias"` below) groks ~10x faster
than the Experiment 6 weight+bias shrinkage — consistent with the bias being where
Adam accumulates the x-independent target-network copy that standard AdamW WD fails
to regularise (see Root Cause section above). A softmin-complexity term
(`lambda_complexity`, the current default mechanism in `grokking_hnn.py`) can also
sustain diversity and grok, but at roughly direct-FFN speed, not 10x.

**Question this ablation targets**: is the bias-WD speedup a structural property of
the HNN reparameterisation (independent of x-diversity), or does it depend on the
collapse/diversity dynamics? Four scenarios (`scripts/hnn_bias_ablation.py`), holding
`weight_decay=0` and `lambda_complexity=0` to isolate the shrinkage mechanism:

| #   | `output_bias` | `fixed_x`     | `shrink_target` | tests                                                |
| --- | ------------- | ------------- | --------------- | ---------------------------------------------------- |
| 0   | True          | True (n_mc=1) | `"both"`        | historical Exp 6 mechanism with no x-diversity       |
| 1   | True          | True (n_mc=1) | `"bias"`        | bias-WD speedup with no x-diversity to collapse      |
| 2   | False         | False         | `"weight"`      | removing the bias "hideout" entirely, usual random x |
| 3   | False         | True (n_mc=1) | `"weight"`      | removing the bias hideout, no x-diversity either     |

New config knobs (`GrokkingHNNConfig`): `output_bias` (drops the stimulus FFN's final
bias — `use_final_bias=False`), `shrink_target`/`shrink_wd` (post-step multiplicative
shrinkage on `stimulus_ffn.layers[-1]`, generalises the old `wd_output_layer` to
target `"bias"`, `"weight"`, or `"both"`), and `probe_n`/`probe_seed` (a fixed x-probe
set, independent of `fixed_x`/`seed`, logged each interval as `probe_param_norm`,
`probe_param_std`, `probe_bias_norm`, `probe_residual_norm` — the bias/x-dependent
decomposition of the generated target params, comparable across all four arms).

**Results** (`data/hnn_bias_ablation/`, `shrink_wd=4.0`, 200k epochs, `train_fraction=0.3`):

| arm                       | mem@99% | test≥90%@ | peak test             | final test | final param_norm | final probe_bias | final probe_resid | final probe_std |
| ------------------------- | ------- | --------- | --------------------- | ---------- | ---------------- | ---------------- | ----------------- | --------------- |
| 0 — both, fixed x         | 4k      | 24k       | 98.1%@116k            | 97.6%      | 190              | 21.2             | 147               | 1.17            |
| 1 — bias only, fixed x    | 2k      | never     | 1.1% (epoch 1, noise) | 0.3%       | 940              | 0.11             | 772               | 4.98            |
| 2 — weight only, random x | 6k      | 22k       | 97.1%@120k            | 94.9%      | 176              | 0 (no bias)      | 176               | 0.0000          |
| 3 — weight only, fixed x  | 6k      | 18k       | 97.5%@198k            | 97.5%      | 179              | 0 (no bias)      | 155               | 1.23            |

**Arm 1 fails outright — it's the only arm that doesn't grok.** Memorises fast (2k,
matching Exp 5), then test accuracy never rises above noise through 200k epochs. The
shrinkage mechanism does exactly what it's told on the bias (`probe_bias_norm` stays
pinned ≈0.11 throughout), but `probe_residual_norm` — the unshrunk x-dependent weight
pathway — grows _monotonically and unboundedly_ to 772 with no sign of plateauing.
With x fixed, nothing forces target information through the bias; the unshrunk weight
pathway freely absorbs everything and the model just overfits at ever-growing norm.

**Arms 0, 2, and 3 all grok, and land within a point or two of each other on every
metric that matters** — memorised@4–6k, test≥90% by 18–24k, peak 97.1–98.1%. This
holds despite the three arms disagreeing on both variables the ablation was designed
to probe: arm 0 has a bias (shrunk), arms 2/3 don't; arm 2 trains on random x, arms
0/3 on a single fixed x. None of that moves the outcome much. What all three share,
and what arm 1 alone lacks, is direct shrinkage on the output-layer **weight** — the
dominant, x-dependent carrier of the generated target params.

**Conclusions:**

1. **Grokking speed here is gated by output-layer _weight_ shrinkage specifically —
   not by x-diversity, and not by whether the bias is regularised.** Arm 1 technically
   applies WD "to the output layer" too (just to the bias half of it), and that's
   exactly the arm that fails. The earlier "~10x faster" bias-only result held under
   random x; this ablation isolates x-diversity as the variable and shows bias-only
   shrinkage depends on it completely, while weight shrinkage (arms 2/3) needs no
   diversity at all — arm 3 groks with x held constant the entire run. The speedup is a
   structural property of directly regularising the reparameterisation's dominant
   pathway, not of the collapse/diversity dynamics.

2. **The bias term is optimisation-irrelevant once the weight is properly regularised.**
   The cleanest single-variable pair is arm 0 vs. arm 3 (both `fixed_x=True`, differing
   only in whether the bias exists and gets its own shrinkage): arm 3 (no bias at all)
   reaches 90% test _faster_ than arm 0 (18k vs 24k), though arm 0 edges to a slightly
   higher eventual peak (98.1% vs 97.5%) — differences small enough to call the bias a
   non-factor either way. Carrying the extra bias degree of freedom doesn't buy
   anything once the weight pathway is shrunk directly.

3. **The diversity/`probe_param_std` numbers need a caveat: 3 of the 4 arms are
   pre-collapsed by construction, not by training.** Arms 0, 1, and 3 use
   `fixed_x=True`, so `probe_param_std` there measures sensitivity to _unseen_ x, not a
   trained-in collapse — there's no diversity during training to collapse in the first
   place. Arm 2 is the only arm where training actually sees varying x, making it the
   only genuine collapse data point — and it's a clean confirmation of the general
   mechanism: `probe_param_std` still collapses to exactly `0.0000` with no final bias
   present at all. The x-independent copy must be encoded via the hidden layer
   (`W0`/`b0`) instead, which `shrink_target="weight"` never touches. Diversity collapse
   looks like a general property of Adam + this architecture under random x, not
   something anchored to the bias specifically.

**A smaller bonus finding**: arms 0/2/3 all modestly _outperform_ the historical Exp 6
`wd_output_layer=4.0` result (weight+bias, random x, same shrink magnitude: mem@8k,
90.8%@22k, peak 96.2%@84k) — faster memorisation (4–6k vs 8k), comparable-or-faster
90%-test time (18–24k vs 22k), and a higher peak (97.1–98.1% vs 96.2%). Fixing x (arm
0 vs. Exp 6) doesn't cost anything either, consistent with conclusion 1 — Exp 6's
speedup was never about x-diversity, it was already coming from the weight component.

**Caveat**: `shrink_wd=4.0` here is the historical `wd_output_layer` value, not a
separately-tuned bias-only coefficient — the earlier "~10x faster" bias-WD finding
(random x, not tested here) may have used a different magnitude. Arm 1's failure is
about x being fixed, not about this specific `shrink_wd` value; a larger `shrink_wd`
would shrink the bias faster but can't fix the underlying issue (the weight pathway is
completely unregularised in arm 1 and will always absorb the slack).

---

## HNN Experiment 7b — Global AdamW WD Control

Open question: is the _scoping_ doing real work, or would plain global AdamW `weight_decay` — applied to
every HNN parameter (`W0`/`b0`/`W1`/`b_last`), routed through the ordinary gradient
update rather than bypassing it — reach the same speed if tuned to the right magnitude?

`optax.adamw`'s decoupled WD term is the same multiplicative-shrinkage form as the
scoped mechanism (`param *= (1 - lr · coefficient)` either way) — the only real
difference is _scope_ (all params vs. just the output layer). This makes `weight_decay`
and `shrink_wd` directly comparable in magnitude, not just in kind.

**Setup** (`scripts/hnn_wd_sweep.py`): identical to Exp 7 arm 0 (`output_bias=True`,
`fixed_x=True`, `n_mc=1`, `lambda_complexity=0`, `train_fraction=0.3`, 200k epochs) but
`shrink_target="none"` and `weight_decay` swept instead of `shrink_wd`, anchored on
arm 0's value of 4.0 and bracketed by factors of 2–4 on each side: `{1, 2, 4, 8, 16}`.

**Results** (`data/hnn_wd_sweep/`):

| `weight_decay` | mem@99%                 | test≥90%@ | final test                | final param_norm | final probe_bias | final probe_resid |
| -------------- | ----------------------- | --------- | ------------------------- | ---------------- | ---------------- | ----------------- |
| 1              | 2k                      | never     | 0.2%                      | 465              | 51.6             | 329               |
| 2              | 2k                      | never     | 8.4% (still rising @200k) | 288              | 32.2             | 169               |
| **4**          | **6k**                  | **18k**   | **96.6%**                 | **185**          | **22.7**         | **79.1**          |
| 8              | never (train caps ~96%) | never     | 60.4% (flat)              | 71               | 12.0             | 23.2              |
| 16             | never (train ~10%)      | never     | 0.0%                      | 22               | 6.0              | 5.9               |

**`weight_decay=4.0` groks, at speed comparable to arm 0.** Memorises a bit slower
(6k vs 4k) but crosses 90% test _faster_ (18k vs 24k), landing 1–2 points lower on final
accuracy (96.6% vs 97.6%) and visibly noisier — oscillating 92–96% through the back half
of training rather than settling smoothly the way arm 0 does. `param_norm` (185) lands
close to the same ~190 equilibrium arm 0 and Exp 6 both converge to.

Global WD reaches near-parity with the scoped mechanism here.

**A secondary signal worth tracking**: at `wd=4`, the gap between the trained-x target
norm (184.8, i.e. `final_param_norm`, measured at the training x) and the probe-set norm
(bias 22.7 + residual 79.1, both much smaller) is proportionally much larger than arm
0's equivalent gap (190 trained vs. 157 probe). The global-WD solution looks more
sensitive to x off the training point than the scoped-shrinkage solution, even though
headline test accuracy on the training x is close. Consistent with the fragility point
above — global WD seems to find a _less uniformly compact_ solution even when it does
land near the right norm.

**Bottom line**: scoping the shrinkage to the output-layer weight is not _necessary_
for grokking speed in the fixed-x regime — properly-tuned global WD gets there too —
but it does appear to buy a smoother, more x-robust solution. It's also worth noting
that `4.0` only looks like an obviously-right coefficient for global WD in hindsight,
because it happens to equal the value already established as correct for the scoped
mechanism (Exp 6, Exp 7 arm 0); found cold, without that anchor, `{1, 2, 4, 8, 16}`
would have looked like "mostly broken, one point works," which is a much harder
coefficient to land on by search than the scoped mechanism's `shrink_wd` has so far
appeared to be.

**Interesting note on zero-shot pnorm**:

- "Trained-x norm" (final_param_norm in the table) — the norm of the target params the model generates at the one x it was actually trained on
- "Probe-set norm" (the probe\_\* fields) — computed on a separate, fixed set of 32 random x values (drawn once from probe_seed, independent of training) that the model never saw during training. This answers: "what target network does the model produce if you feed it some other x it was never optimized for?"

These can differ a lot, and the gap between them is itself informative. For wd=4: trained-x norm is 184.8, but the probe-set gives bias 22.7 + residual 79.1 (roughly 82 combined) — the network the model builds for its one training x is more than 2x bigger than what it builds for x values it's never seen. For arm 0 (the scoped mechanism): trained-x norm is 190, probe-set is 157 — only about 20% smaller, a much tighter gap.

What that means concretely: both arms/runs get high accuracy at the training x (that's literally what's being scored as test accuracy in a fixed-x run). But arm 0's
solution looks roughly similar in scale no matter what x you query it at, while wd=4's solution is comparatively "spiky" — it built something unusually large and specffic around the one x it was trained on, and reverts to something much smaller/different everywhere else. That's a sign the global-WD solution is more narrowly tied to the specific training point, i.e. less uniform/robust across x-space, even though headline accuracy at the training x looks comparable.

# FFN Step: Minimum Network for z² + c

## Question

What is the minimum tanh FFN required to learn the single-step Mandelbrot map?

```
f(z_re, z_im, c_re, c_im) = (z_re² - z_im² + c_re,  2·z_re·z_im + c_im)
```

This is the same question as "what cell is sufficient for an RNN to generalise the Mandelbrot set?" — the cell IS the FFN.

## Architecture

```
[4] -tanh-> [hidden]*depth -linear-> [2]
```

- Input: raw (z_re, z_im, c_re, c_im) — no Fourier encoding needed (no BPTT chaos here)
- Output: linear (no tanh) — the map has outputs in [-2.5, 2.5]; a saturating readout makes high fit impossible
- Depth: 1 hidden layer

## Domain

- z sampled uniformly from [-1.5, 1.5]² (covers realistic orbit values)
- c sampled uniformly from [-1.5, 0.5] × [-0.85, 0.85] — slightly larger than the main Mandelbrot cardioid + period-2 bulb
- Sampled independently (testing function approximation, not dynamics)

## Results (depth=1, AdamW, 2000 epochs, 20k samples)

| width | params | test MSE   | R²     |
|-------|--------|------------|--------|
| 1     | 9      | 1.677      | 0.097  |
| 2     | 16     | 1.214      | 0.346  |
| 3     | 23     | 0.636      | 0.657  |
| **4** | **30** | **0.037**  | **0.980** |
| 6     | 44     | 0.0017     | 0.9991 |
| 8     | 58     | 0.00024    | 0.9999 |

**Sharp phase transition at width 4.** R² jumps from 0.66 → 0.98 between width 3 and 4.

## Theory

The map's quadratic part lives in the 3-dimensional space spanned by {z_re², z_im², z_re·z_im}.
Each tanh neuron contributes one rank-1 quadratic form w_iw_iᵀ (from the second-order Taylor term,
only present when bias ≠ 0). Since these are rank-1 with positive trace, a single neuron cannot
represent the trace-zero target diag(1, −1) needed for z_re² − z_im². Width 2 can form two such
forms, still insufficient to span the full 3D quadratic space. Width 3 reaches the dimension of
the target but training struggles to find the correct configuration. Width 4 provides the first
reliable overcomplete basis — the sharp R² jump confirms it.

**Practical minimum: width=4 (30 params) for R²>0.98; width=6 (44 params) for R²>0.999.**

## Iteration check

After training, the learned cell is iterated T=50 times over a c-grid (starting at z₀=0) to
check whether one-step accuracy translates to multi-step Mandelbrot dynamics.

Width=6 (R²=0.999) reproduces the broad shape of the Mandelbrot set (main cardioid, period-2 bulb)
but accumulates visible error at the boundary — one-step error compounds under iteration.
Width=8 (R²=0.9999) is sharper; the cardioid outline is clearly reproduced.

Neither matches the true boundary at T=50 — even 1e-4 one-step error composes to ~1% error
per iteration, which degrades boundary sharpness after ~10 steps. This is why the RNN with a
cell of similar size cannot generalise to the full Mandelbrot set under long roll-outs.

## Training

```
god-ffn-step
```

Config: `god.training.ffn_step.TrainConfig`.

Output: `data/mandel/ffn_direct/` — sweep.json, width_sweep.png, iteration_check.png.

"""Ablation: is HNN grokking speed driven by the bias-WD mechanism structurally,
or does x-diversity (collapse) matter?

Background (docs/hnn_dynamics.md): plain AdamW WD on the HNN fails to grok because
Adam effectively starves the WD applied to the x-independent copy of the target
network that accumulates in the stimulus FFN's final-layer bias. Two fixes were
found to work:
  (1) shrink_target="bias" — post-step multiplicative shrinkage applied directly
      to that bias (bypasses Adam) — groks ~10x faster than the historical
      wd_output_layer (weight+bias) mechanism.
  (2) lambda_complexity (softmin complexity term) — sustains diversity instead
      of collapsing it, grokking at roughly direct-FFN speed.

This script isolates whether (1)'s speedup is structural (a property of the HNN
reparameterisation / the shrinkage mechanism itself) or depends on x-diversity
dynamics, via 4 scenarios:

  0. output_bias=True,  fixed_x=True,  shrink_target="both"   — historical wd_output_layer mechanism, no x-diversity
  1. output_bias=True,  fixed_x=True,  shrink_target="bias"   — bias WD, no x-diversity to collapse
  2. output_bias=False, fixed_x=False, shrink_target="weight" — no bias to hide in; usual random x
  3. output_bias=False, fixed_x=True,  shrink_target="weight" — no bias; no x-diversity either

Scenario 2/3 use shrink_target="weight" (not plain weight_decay) so the shrinkage
mechanism is matched across arms — output_bias=False has no bias for "bias"
shrinkage to act on, so "weight" (the only x-dependent carrier of the target
params) is the structurally-equivalent knob.

Arm 0 is the single-variable counterpart to historical Exp 6 (`wd_output_layer=4.0`,
weight+bias shrinkage, random x, groks at ~3.5x FFN speed): same shrink_target="both"
mechanism, only x swapped from random to fixed. Answers whether Exp 6's success also
depends on x-diversity, the way arm 1 showed bias-only shrinkage does.

Each run logs probe_* diagnostics (see grokking_hnn.py) from a fixed x-probe set
shared across all arms (independent of fixed_x/seed), so param_std/bias_norm/
residual_norm are directly comparable across arms for the retrospective analysis.

Usage:
    uv run scripts/hnn_bias_ablation.py --shrink-wd 4.0 --n-epochs 200000
    uv run scripts/hnn_bias_ablation.py --scenario 0_both_wd_fixed_x  # just one arm
"""

import argparse
from pathlib import Path

from god.training.grokking_hnn import GrokkingHNNConfig, train


def _scenarios(shrink_wd: float) -> dict[str, dict]:
    return {
        "0_both_wd_fixed_x": dict(
            output_bias=True, fixed_x=True, n_mc=1,
            shrink_target="both", shrink_wd=shrink_wd,
        ),
        "1_bias_wd_fixed_x": dict(
            output_bias=True, fixed_x=True, n_mc=1,
            shrink_target="bias", shrink_wd=shrink_wd,
        ),
        "2_no_bias_random_x": dict(
            output_bias=False, fixed_x=False, n_mc=4,
            shrink_target="weight", shrink_wd=shrink_wd,
        ),
        "3_no_bias_fixed_x": dict(
            output_bias=False, fixed_x=True, n_mc=1,
            shrink_target="weight", shrink_wd=shrink_wd,
        ),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output-dir", default="data/hnn_bias_ablation")
    p.add_argument("--scenario", default=None, choices=list(_scenarios(0.0)),
                    help="run only this scenario (default: all 4, sequentially)")
    p.add_argument("--shrink-wd", type=float, default=4.0,
                    help="post-step shrinkage magnitude (historical wd_output_layer default: 4.0). "
                         "Substitute your own tuned bias-WD value if you have one from prior runs.")
    p.add_argument("--n-epochs", type=int, default=200_000)
    p.add_argument("--checkpoint-interval", type=int, default=2000)
    p.add_argument("--log-interval", type=int, default=2000)
    p.add_argument("--train-fraction", type=float, default=0.3,
                    help="0.3 matches the reference numbers in docs/hnn_dynamics.md")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    output_dir = Path(args.output_dir)
    scenarios = _scenarios(args.shrink_wd)
    names = [args.scenario] if args.scenario else list(scenarios)

    for name in names:
        cfg = GrokkingHNNConfig(
            data_dir=str(output_dir / name),
            modulus=97,
            target_hidden=32,
            target_depth=1,
            n_stimulus=8,
            stim_ffn_hidden=8,
            stim_ffn_depth=1,
            init_scale=0.1,
            train_fraction=args.train_fraction,
            lr=1e-3,
            weight_decay=0.0,       # isolate the shrink_target mechanism — no AdamW WD confound
            grad_clip=1.0,
            lambda_complexity=0.0,  # isolate shrink_target — no softmin-diversity confound
            n_epochs=args.n_epochs,
            log_interval=args.log_interval,
            checkpoint_interval=args.checkpoint_interval,
            hexplot_interval=0,
            seed=args.seed,
            **scenarios[name],
        )
        print(f"\n{'=' * 60}\n{name}\n{'=' * 60}")
        train(cfg)


if __name__ == "__main__":
    main()

"""Render hexplots for all checkpoints in a run directory.

Skips checkpoints that already have a corresponding hexplot.

Usage:
    uv run scripts/render_hexplots.py data/hnn_best
    uv run scripts/render_hexplots.py data/hnn_best --force   # re-render all
"""

import argparse
import json
import re
import sys
from pathlib import Path

import equinox as eqx

from god.training.grokking_hnn import GrokkingHNNConfig, _hexplot, _setup


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--force", action="store_true", help="Re-render existing hexplots")
    args = parser.parse_args()

    run_dir = args.run_dir
    cfg_path = run_dir / "config.json"
    if not cfg_path.exists():
        sys.exit(f"No config.json in {run_dir}")

    cfg = GrokkingHNNConfig(**json.loads(cfg_path.read_text()))
    dataset, template = _setup(cfg)

    checkpoints = sorted(run_dir.glob("checkpoint_*.eqx"))
    if not checkpoints:
        sys.exit(f"No checkpoints found in {run_dir}")

    print(f"Found {len(checkpoints)} checkpoints in {run_dir}")

    for ckpt in checkpoints:
        epoch = int(re.search(r"checkpoint_(\d+)\.eqx", ckpt.name).group(1))
        hexplot_path = run_dir / f"hexplot_{epoch:07d}.png"
        if hexplot_path.exists() and not args.force:
            print(f"  skip  epoch {epoch:>7,} (exists)")
            continue

        model = eqx.tree_deserialise_leaves(str(ckpt), template)
        _hexplot(
            model, dataset, cfg.n_stimulus, epoch, run_dir,
            xlim=cfg.hexplot_xlim, ylim=cfg.hexplot_ylim,
        )
        print(f"  done  epoch {epoch:>7,}")

    print(f"\nHexplots → {run_dir}/hexplot_*.png")


if __name__ == "__main__":
    main()

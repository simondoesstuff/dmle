"""Per-checkpoint UMAP of HNN target network parameters.

For each sampled checkpoint, generates a batch of target networks, evaluates
each on the validation set, and embeds the parameter vectors with UMAP.
Points are coloured by log validation loss normalized within that batch.
The actual min/max log-loss values are annotated as text so plots are
comparable across checkpoints despite the per-batch normalization.

One PNG is written per checkpoint to <out-dir>/step_XXXXX.png.

Usage:
    god-hnn-umap [--data-dir PATH] [--n-ckpts N] [--n-networks N] [--n-eval N] [--out-dir PATH]
"""

import argparse
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from umap import UMAP

from god.datasets.mandelbrot import make_mandelbrot_dataset
from god.training.hnn import HNNConfig, _make_model_template


def _find_checkpoints(ckpt_dir: Path) -> list[tuple[int, Path]]:
    return sorted(
        (int(p.stem.split("_")[1]), p)
        for p in ckpt_dir.glob("step_*.eqx")
        if "_opt" not in p.name
    )


def _plot_umap(
    embedding: np.ndarray,
    log_losses: np.ndarray,
    step: int,
    out_path: Path,
) -> None:
    p5, p95 = np.percentile(log_losses, [5, 95])
    lo, hi = log_losses.min(), log_losses.max()
    normalized = (log_losses - lo) / (hi - lo + 1e-10)

    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(
        embedding[:, 0], embedding[:, 1],
        c=normalized, cmap="plasma", vmin=0, vmax=1,
        s=30, alpha=0.85, linewidths=0,
    )
    plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label="log val loss (batch-normalized)")
    ax.set_title(f"HNN target network UMAP — step {step:,}")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(
        0.02, 0.02,
        f"log loss  p5 {p5:.3f}  p95 {p95:.3f}",
        transform=ax.transAxes,
        fontsize=8, va="bottom", ha="left",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7, ec="none"),
    )
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=120)
    plt.close(fig)


def run(
    data_dir: Path,
    n_ckpts: int,
    n_networks: int,
    n_eval: int,
    out_dir: Path,
) -> None:
    ckpt_dir = data_dir / "checkpoints"
    all_ckpts = _find_checkpoints(ckpt_dir)
    all_steps = [s for s, _ in all_ckpts]
    ckpt_map = {s: p for s, p in all_ckpts}

    step_set = set(all_steps)
    targets = np.linspace(all_steps[0], all_steps[-1], n_ckpts)
    sampled_steps = sorted({min(step_set, key=lambda s: abs(s - t)) for t in targets})

    print(f"Checkpoints: {len(all_steps)} total → {len(sampled_steps)} evenly spaced")
    print(f"Networks per checkpoint: {n_networks} | Val points: {n_eval}")

    cfg = HNNConfig(data_dir=str(data_dir))
    model_template = _make_model_template(cfg)

    key = jax.random.PRNGKey(0)
    k_data, k_stim = jax.random.split(key)
    dataset = make_mandelbrot_dataset(cfg.n_train, cfg.n_test, cfg.num_steps, k_data, K=cfg.K)
    c_eval = dataset.test_inputs[:n_eval]
    t_eval = dataset.test_targets[:n_eval]

    @eqx.filter_jit
    def collect(model, x_samples):
        param_vecs = jax.vmap(model.target_params)(x_samples)
        losses = jax.vmap(
            lambda x: jnp.mean((jax.vmap(model, in_axes=(0, None))(c_eval, x) - t_eval) ** 2)
        )(x_samples)
        return param_vecs, losses

    reducer = UMAP(n_components=2, n_neighbors=min(15, n_networks - 1), random_state=0, verbose=False)

    for step in tqdm(sampled_steps, desc="checkpoints"):
        model = eqx.tree_deserialise_leaves(str(ckpt_map[step]), model_template)

        k_stim, subkey = jax.random.split(k_stim)
        x_samples = jax.random.uniform(subkey, (n_networks, cfg.n_stimulus), minval=-1.0, maxval=1.0)

        param_vecs, losses = collect(model, x_samples)
        param_vecs = np.array(param_vecs)
        log_losses = np.log(np.array(losses))

        embedding = reducer.fit_transform(param_vecs)
        _plot_umap(embedding, log_losses, step, out_dir / f"step_{step:06d}.png")

    print(f"\nSaved {len(sampled_steps)} plots → {out_dir}/")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default="data/mandel/hnn", help="HNN output directory")
    parser.add_argument("--n-ckpts", type=int, default=20, help="Number of evenly spaced checkpoints (default: 20)")
    parser.add_argument("--n-networks", type=int, default=64, help="Stimulus draws per checkpoint (default: 64)")
    parser.add_argument("--n-eval", type=int, default=500, help="Val points for loss evaluation (default: 500)")
    parser.add_argument("--out-dir", default=None, help="Output directory (default: <data-dir>/umap_per_ckpt/)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir) if args.out_dir else data_dir / "umap_per_ckpt"
    run(data_dir, args.n_ckpts, args.n_networks, args.n_eval, out_dir)


if __name__ == "__main__":
    main()

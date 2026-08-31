"""Separability of HNN target networks by validation loss over training.

For each checkpoint, draws a batch of target networks and computes the
silhouette score measuring how well median-split val-loss groups (good vs bad
networks) separate in raw parameter space. Plots the score over training steps.

A score near +1 means good and bad networks occupy distinct regions of param
space (separable). Near 0 means they overlap. Negative means the split is
inverted.

Usage:
    god-hnn-sep [--data-dir PATH] [--n-networks N] [--n-eval N] [--out PATH]
"""

import argparse
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from god.datasets.mandelbrot import make_mandelbrot_dataset
from god.training.hnn import HNNConfig, _make_model_template


def _find_checkpoints(ckpt_dir: Path) -> list[tuple[int, Path]]:
    return sorted(
        (int(p.stem.split("_")[1]), p)
        for p in ckpt_dir.glob("step_*.eqx")
        if "_opt" not in p.name
    )


def run(
    data_dir: Path,
    n_networks: int,
    n_eval: int,
    out_path: Path,
) -> None:
    ckpt_dir = data_dir / "checkpoints"
    all_ckpts = _find_checkpoints(ckpt_dir)
    all_steps = [s for s, _ in all_ckpts]
    ckpt_map = {s: p for s, p in all_ckpts}
    print(f"Checkpoints: {len(all_steps)} | Networks per checkpoint: {n_networks} | Val points: {n_eval}")

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

    steps_out = []
    silhouettes = []
    param_stds = []

    for step in tqdm(all_steps, desc="checkpoints"):
        model = eqx.tree_deserialise_leaves(str(ckpt_map[step]), model_template)

        k_stim, subkey = jax.random.split(k_stim)
        x_samples = jax.random.uniform(subkey, (n_networks, cfg.n_stimulus), minval=-1.0, maxval=1.0)

        param_vecs, losses = collect(model, x_samples)
        param_vecs = np.array(param_vecs)
        losses = np.array(losses)

        # median split: good = lower half, bad = upper half
        median = np.median(losses)
        good = param_vecs[losses < median]
        bad  = param_vecs[losses >= median]
        sep = float(np.linalg.norm(good.mean(axis=0) - bad.mean(axis=0)))

        # random baseline: shuffle labels 100 times
        rng = np.random.default_rng(42)
        null_seps = []
        n_good = len(good)
        for _ in range(100):
            idx = rng.permutation(len(param_vecs))
            g, b = param_vecs[idx[:n_good]], param_vecs[idx[n_good:]]
            null_seps.append(np.linalg.norm(g.mean(axis=0) - b.mean(axis=0)))
        null_mean, null_std = np.mean(null_seps), np.std(null_seps)
        sep_z = (sep - null_mean) / (null_std + 1e-10)

        steps_out.append(step)
        silhouettes.append(sep_z)
        param_stds.append(float(np.mean(np.std(param_vecs, axis=0))))

    steps_out = np.array(steps_out)
    silhouettes = np.array(silhouettes)
    param_stds = np.array(param_stds)

    _plot(steps_out, silhouettes, param_stds, out_path)
    print(f"Saved → {out_path}")


def _smooth(x: np.ndarray, w: int = 10) -> np.ndarray:
    kernel = np.ones(w) / w
    return np.convolve(x, kernel, mode="same")


def _plot(steps, silhouettes, param_stds, out_path: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    ax1.plot(steps, silhouettes, alpha=0.3, color="steelblue", linewidth=0.8)
    ax1.plot(steps, _smooth(silhouettes), color="steelblue", linewidth=1.8)
    ax1.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax1.set_ylabel("Separation Z-score\n(vs random label shuffle)")
    ax1.set_title("HNN target-network separability over training")

    ax2.plot(steps, param_stds, alpha=0.3, color="coral", linewidth=0.8)
    ax2.plot(steps, _smooth(param_stds), color="coral", linewidth=1.8)
    ax2.set_ylabel("Param std\n(network diversity)")
    ax2.set_xlabel("Training step")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default="data/mandel/hnn", help="HNN output directory")
    parser.add_argument("--n-networks", type=int, default=64, help="Stimulus draws per checkpoint (default: 64)")
    parser.add_argument("--n-eval", type=int, default=500, help="Val points for loss evaluation (default: 500)")
    parser.add_argument("--out", default=None, help="Output PNG path (default: <data-dir>/separability.png)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_path = Path(args.out) if args.out else data_dir / "separability.png"
    run(data_dir, args.n_networks, args.n_eval, out_path)


if __name__ == "__main__":
    main()

"""Shared training utilities: visualization, metrics plotting, checkpointing."""

from collections.abc import Callable
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from god.datasets.mandelbrot import ESCAPE_RADIUS, K_DEFAULT, encode, mandelbrot_mag

_LR_FLOOR = 1e-9


def checkpoint_interval(
    current_lr: float,
    lr_peak: float,
    min_interval: int,
    max_interval: int,
) -> int:
    effective_lr = max(current_lr, _LR_FLOOR)
    raw = round(min_interval * lr_peak / effective_lr)
    return int(np.clip(raw, min_interval, max_interval))


def visualize_mandelbrot(
    predict_fn: Callable[[jax.Array], jax.Array],
    out_path: Path,
    num_steps: int,
    K: int = K_DEFAULT,
    title: str | None = None,
) -> None:
    """Render a 3-panel Mandelbrot heatmap: true | predicted | absolute error.

    Args:
        predict_fn: maps c_encs (N, enc_dim) → pred_mags (N,) in Mandelbrot scale.
        out_path: where to save the PNG.
        num_steps: Mandelbrot iteration count for the ground truth.
        K: Fourier frequency bands used when encoding the grid.
        title: optional figure suptitle.
    """
    import matplotlib.pyplot as plt

    res = 300
    re_vals = np.linspace(-2.0, 2.0, res)
    im_vals = np.linspace(-2.0, 2.0, res)
    RE, IM = np.meshgrid(re_vals, im_vals)
    c_r = jnp.array(RE.ravel())
    c_i = jnp.array(IM.ravel())

    true_mags = jax.vmap(mandelbrot_mag, in_axes=(0, 0, None, None))(
        c_r, c_i, num_steps, ESCAPE_RADIUS
    )
    c_encs = jax.vmap(encode, in_axes=(0, 0, None))(c_r, c_i, K)
    pred_mags = predict_fn(c_encs)

    true_grid = np.array(true_mags).reshape(res, res)
    pred_grid = np.array(pred_mags).reshape(res, res)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    kw = dict(origin="lower", extent=[-2, 2, -2, 2], vmin=0, vmax=ESCAPE_RADIUS, cmap="inferno")
    axes[0].imshow(true_grid, **kw)
    axes[0].set_title("True |z_T|")
    axes[1].imshow(pred_grid, **kw)
    axes[1].set_title("Predicted |z_T|")
    err = np.abs(true_grid - pred_grid)
    axes[2].imshow(err, origin="lower", extent=[-2, 2, -2, 2], cmap="hot")
    axes[2].set_title("Absolute error")
    for ax in axes:
        ax.axhline(0, color="white", lw=0.5, ls="--")
        ax.set_xlabel("Re(c)")
        ax.set_ylabel("Im(c)")
    if title is not None:
        fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=150)
    plt.close(fig)


def plot_metrics_standard(
    metrics: list[dict[str, float | int | bool]],
    baseline: float,
    out_path: Path,
    title: str = "Training (log–log)",
) -> None:
    """Log-log loss curve with LR overlay; compatible with MLP and PCN metrics."""
    import matplotlib.pyplot as plt

    epochs = [m["epoch"] for m in metrics]
    train = [m["train_loss"] for m in metrics]
    test = [m["test_loss"] for m in metrics]
    lrs = [m["lr"] for m in metrics]
    ckpt_epochs = [m["epoch"] for m in metrics if m.get("checkpoint", False)]

    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax2 = ax1.twinx()

    ax1.plot(epochs, train, label="train MSE", lw=1.5, color="tab:blue")
    ax1.plot(epochs, test, label="test MSE", lw=1.5, color="tab:orange")
    ax1.axhline(baseline, color="gray", ls="--", lw=1, label=f"const baseline ({baseline:.3f})")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("epoch (log)")
    ax1.set_ylabel("MSE (log)")

    ax2.plot(epochs, lrs, color="tab:green", lw=0.8, alpha=0.35, label="LR")
    ax2.set_ylabel("learning rate")
    ax2.set_ylim(bottom=0)

    for e in ckpt_epochs:
        ax1.axvline(e, color="lightgray", lw=0.3, alpha=0.5)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8)
    ax1.set_title(title)
    ax1.grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=150)
    plt.close(fig)

"""MLP training for Mandelbrot magnitude prediction.

Two preset configurations:
  TrainConfig() — default: small-data, high-reg, long training (primary grokking study)
  simple_config() — large-data, cosine LR, standard regularisation (baseline)
"""

import dataclasses
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from tqdm import trange

from god.data import ESCAPE_RADIUS, make_dataset
from god.encoding import K_DEFAULT, enc_dim
from god.model import MandelbrotRNN

_LR_FLOOR = 1e-9


@dataclass
class TrainConfig:
    # Paths
    data_dir: str = "data/mandel/mlp_grokk"

    # Model
    K: int = K_DEFAULT
    hidden_dim: int = 128
    depth: int = 2
    num_steps: int = 10
    linear_head: bool = False

    # Data
    n_train: int = 500
    n_test: int = 2_000

    # Training
    batch_size: int = 64
    n_epochs: int = 12_000
    lr_peak: float = 1e-4
    warmup_frac: float = 0.0   # 0 → constant LR; >0 → warmup then cosine decay
    lr_end: float = 1e-4       # equal to lr_peak → constant; 0.0 → cosine-to-zero
    weight_decay: float = 0.5
    grad_clip: float = 1.0

    # Checkpointing: interval ∝ lr_peak / current_lr
    checkpoint_min_interval: int = 50
    checkpoint_max_interval: int = 50

    # Reproducibility
    seed: int = 0


def simple_config() -> TrainConfig:
    """Baseline MLP: large dataset, cosine LR with warmup, standard regularisation."""
    return TrainConfig(
        data_dir="data/mandel/mlp",
        n_train=10_000,
        batch_size=256,
        n_epochs=3_000,
        lr_peak=1e-3,
        warmup_frac=0.05,
        lr_end=0.0,
        weight_decay=0.01,
        checkpoint_min_interval=10,
        checkpoint_max_interval=300,
    )


def _checkpoint_interval(current_lr: float, cfg: TrainConfig) -> int:
    effective_lr = max(current_lr, _LR_FLOOR)
    raw = round(cfg.checkpoint_min_interval * cfg.lr_peak / effective_lr)
    return int(np.clip(raw, cfg.checkpoint_min_interval, cfg.checkpoint_max_interval))


def _loss_fn(
    model: MandelbrotRNN,
    c_encs: jax.Array,
    targets: jax.Array,
) -> jax.Array:
    h_T = jax.vmap(model)(c_encs)
    pred_mags = jax.vmap(model.predict_magnitude)(h_T)
    return jnp.mean((pred_mags / ESCAPE_RADIUS - targets) ** 2)


def _visualize_model(
    model: MandelbrotRNN,
    cfg: TrainConfig,
    out_path: Path,
    epoch: int | None = None,
) -> None:
    import matplotlib.pyplot as plt

    from god.data import mandelbrot_mag
    from god.encoding import encode

    res = 300
    re_vals = np.linspace(-2.0, 2.0, res)
    im_vals = np.linspace(-2.0, 2.0, res)
    RE, IM = np.meshgrid(re_vals, im_vals)
    c_r = jnp.array(RE.ravel())
    c_i = jnp.array(IM.ravel())

    true_mags = jax.vmap(mandelbrot_mag, in_axes=(0, 0, None, None))(
        c_r, c_i, cfg.num_steps, ESCAPE_RADIUS
    )
    c_encs = jax.vmap(encode, in_axes=(0, 0, None))(c_r, c_i, cfg.K)
    h_T = jax.vmap(model)(c_encs)
    pred_mags = jax.vmap(model.predict_magnitude)(h_T)

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

    fig.suptitle(f"Epoch {epoch}" if epoch is not None else "Final", fontsize=12)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=150)
    plt.close(fig)


def plot_metrics(
    metrics: list[dict[str, Any]],
    baseline: float,
    out_path: Path,
) -> None:
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
    ax1.set_title("Mandelbrot RNN — training (log–log)")
    ax1.grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=150)
    plt.close(fig)


def load_checkpoint(path: str | Path, cfg: TrainConfig) -> MandelbrotRNN:
    d = enc_dim(cfg.K)
    template = MandelbrotRNN(
        d, cfg.hidden_dim, cfg.depth, cfg.num_steps,
        jax.random.PRNGKey(0), linear_head=cfg.linear_head,
    )
    return eqx.tree_deserialise_leaves(str(path), template)


def train(cfg: TrainConfig) -> MandelbrotRNN:
    key = jax.random.PRNGKey(cfg.seed)
    k_model, k_train, k_test = jax.random.split(key, 3)

    d = enc_dim(cfg.K)
    model = MandelbrotRNN(
        d, cfg.hidden_dim, cfg.depth, cfg.num_steps, k_model,
        linear_head=cfg.linear_head,
    )

    data_dir = Path(cfg.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "config.json").write_text(json.dumps(dataclasses.asdict(cfg), indent=2))

    print("Generating datasets...")
    c_train, t_train = make_dataset(cfg.n_train, cfg.num_steps, k_train, train=True, K=cfg.K)
    c_test, t_test = make_dataset(cfg.n_test, cfg.num_steps, k_test, train=False, K=cfg.K)
    const_baseline = float(jnp.mean((t_train - jnp.mean(t_train)) ** 2))
    print(f"Constant-mean MSE baseline: {const_baseline:.4f}")

    steps_per_epoch = max(1, cfg.n_train // cfg.batch_size)
    total_steps = steps_per_epoch * cfg.n_epochs
    warmup_steps = int(cfg.warmup_frac * total_steps)

    if warmup_steps > 0:
        schedule: optax.Schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=cfg.lr_peak,
            warmup_steps=warmup_steps,
            decay_steps=total_steps,
            end_value=cfg.lr_end,
        )
    else:
        schedule = optax.constant_schedule(cfg.lr_peak)

    optimizer = optax.chain(
        optax.clip_by_global_norm(cfg.grad_clip),
        optax.adamw(learning_rate=schedule, weight_decay=cfg.weight_decay),
    )
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    def step(
        model: MandelbrotRNN,
        opt_state: optax.OptState,
        c_batch: jax.Array,
        t_batch: jax.Array,
    ) -> tuple[MandelbrotRNN, optax.OptState, jax.Array]:
        loss, grads = eqx.filter_value_and_grad(_loss_fn)(model, c_batch, t_batch)
        updates, new_opt_state = optimizer.update(
            grads, opt_state, eqx.filter(model, eqx.is_array)
        )
        return eqx.apply_updates(model, updates), new_opt_state, loss

    @eqx.filter_jit
    def eval_loss(
        model: MandelbrotRNN,
        c_encs: jax.Array,
        targets: jax.Array,
    ) -> jax.Array:
        return _loss_fn(model, c_encs, targets)

    rng = np.random.default_rng(cfg.seed)
    c_train_np = np.array(c_train)
    t_train_np = np.array(t_train)

    n_params = sum(x.size for x in jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array)))
    head_str = "linear" if cfg.linear_head else "direct"
    print(
        f"Model: enc_dim={d}, hidden={cfg.hidden_dim}, depth={cfg.depth}, "
        f"steps={cfg.num_steps}, head={head_str}, params={n_params:,}"
    )
    print(f"Training: {cfg.n_train} samples, {steps_per_epoch} steps/epoch, {cfg.n_epochs} epochs")

    metrics: list[dict[str, Any]] = []
    ckpt_dir = data_dir / "checkpoints"
    last_checkpoint_epoch = 0

    pbar = trange(cfg.n_epochs, desc="epochs")
    for epoch in pbar:
        t0 = time.perf_counter()
        idx = rng.permutation(cfg.n_train)
        epoch_loss = 0.0
        for i in range(steps_per_epoch):
            batch_idx = idx[i * cfg.batch_size : (i + 1) * cfg.batch_size]
            c_b = jnp.array(c_train_np[batch_idx])
            t_b = jnp.array(t_train_np[batch_idx])
            model, opt_state, loss = step(model, opt_state, c_b, t_b)
            epoch_loss += float(loss)

        train_loss = epoch_loss / steps_per_epoch
        test_loss = float(eval_loss(model, c_test, t_test))
        elapsed = time.perf_counter() - t0

        completed = epoch + 1
        current_lr = float(schedule(completed * steps_per_epoch))

        record: dict[str, Any] = {
            "epoch": completed,
            "train_loss": train_loss,
            "test_loss": test_loss,
            "lr": current_lr,
        }

        interval = _checkpoint_interval(current_lr, cfg)
        is_last = completed == cfg.n_epochs
        if completed - last_checkpoint_epoch >= interval or is_last:
            record["checkpoint"] = True
            last_checkpoint_epoch = completed
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            tag = f"epoch_{completed:04d}"
            eqx.tree_serialise_leaves(str(ckpt_dir / f"{tag}.eqx"), model)
            _visualize_model(model, cfg, ckpt_dir / f"{tag}_pred.png", epoch=completed)

        metrics.append(record)

        if record.get("checkpoint"):
            (data_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
            plot_metrics(metrics, const_baseline, data_dir / "train_curve.png")

        pbar.set_postfix(
            train=f"{train_loss:.4f}",
            test=f"{test_loss:.4f}",
            lr=f"{current_lr:.2e}",
            s=f"{elapsed:.1f}s",
        )

    (data_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    plot_metrics(metrics, const_baseline, data_dir / "train_curve.png")
    print(f"Metrics → {data_dir / 'metrics.json'}")
    print(f"Train curve → {data_dir / 'train_curve.png'}")

    return model


def visualize(model: MandelbrotRNN, cfg: TrainConfig) -> None:
    data_dir = Path(cfg.data_dir)
    out = data_dir / "mandelbrot_prediction.png"
    _visualize_model(model, cfg, out)
    print(f"Saved {out}")


def main() -> None:
    """Entry point for god-mlp: baseline large-data experiment."""
    cfg = simple_config()
    model = train(cfg)
    visualize(model, cfg)


def main_grokk() -> None:
    """Entry point for god-mlp-grokk: primary small-data grokking experiment."""
    cfg = TrainConfig()
    model = train(cfg)
    visualize(model, cfg)

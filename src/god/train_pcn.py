"""Training script for a discriminative Predictive Coding Network (PCN) on the
Mandelbrot magnitude task, using the JPC library.

Architecture: feedforward MLP trained with PC inference (not BPTT).
  - Input:  Fourier-encoded c  (enc_dim = 2 + 4K features)
  - Hidden: (depth - 1) tanh layers of width `hidden_dim`
  - Output: 1 scalar — normalised magnitude  |z_T| / escape_radius

Weight tying
─────────────
JPC's MLP has independent weights per layer. To recover the same low-KC
inductive bias as the RNN (shared weights across all steps), hidden-to-hidden
layers (indices 1..depth-2, which are the only square width×width layers) are
projected back onto the weight-tied manifold after every parameter update by
averaging their weights. The input projection (enc_dim→width, layer 0) and
output head (width→1, layer depth-1) remain untied since they differ in shape.

Loss comparability
───────────────────
JPC's mse_loss = 0.5 * mean(sum((y - ŷ)², axis=1)).
Eval metrics here use plain MSE = mean((y - ŷ)²) to match train.py.
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
import jpc
import numpy as np
import optax
from jaxtyping import Array, Float
from tqdm import trange

from god.data import ESCAPE_RADIUS, make_dataset
from god.encoding import K_DEFAULT, enc_dim

DATA_DIR = Path("data/pcn")

_LR_FLOOR = 1e-9


@dataclass
class PCNConfig:
    # Model
    K: int = K_DEFAULT
    hidden_dim: int = 128
    depth: int = 10      # number of MLP layers (each layer = one hierarchy level)
    use_bias: bool = True
    act_fn: str = "tanh"

    # Data
    n_train: int = 10_000
    n_test: int = 2_000
    num_steps: int = 10  # used only for data generation (Mandelbrot iterations)

    # Training
    batch_size: int = 256
    n_epochs: int = 3000
    lr_peak: float = 1e-3
    warmup_frac: float = 0.05
    weight_decay: float = 0.01
    grad_clip: float = 1.0

    # PC inference
    max_t1: int = 20  # max integration time for PC inference ODE

    # Checkpointing
    checkpoint_min_interval: int = 10
    checkpoint_max_interval: int = 300

    seed: int = 0


@eqx.filter_jit
def _tie_hidden_layers(model: list) -> list:
    """Project to weight-tied manifold by averaging hidden layer parameters.

    Only layers 1..depth-2 (width×width) are tied; the input projection
    (enc_dim→width) and output head (width→1) are left free.
    """
    depth = len(model)
    if depth <= 3:
        return model  # 0 or 1 hidden layer — nothing to average
    hidden = list(range(1, depth - 1))
    W_avg = jnp.mean(jnp.stack([model[i].layers[1].weight for i in hidden]), axis=0)
    for i in hidden:
        model = eqx.tree_at(lambda m, _i=i: m[_i].layers[1].weight, model, W_avg)
    if model[hidden[0]].layers[1].bias is not None:
        b_avg = jnp.mean(jnp.stack([model[i].layers[1].bias for i in hidden]), axis=0)
        for i in hidden:
            model = eqx.tree_at(lambda m, _i=i: m[_i].layers[1].bias, model, b_avg)
    return model


def _checkpoint_interval(current_lr: float, cfg: PCNConfig) -> int:
    effective_lr = max(current_lr, _LR_FLOOR)
    raw = round(cfg.checkpoint_min_interval * cfg.lr_peak / effective_lr)
    return int(np.clip(raw, cfg.checkpoint_min_interval, cfg.checkpoint_max_interval))


def _pcn_predict(
    model: list,
    c_encs: Float[Array, "n d"],
) -> Float[Array, "n"]:
    """Feedforward prediction (no inference loop) — used for eval and visualisation."""
    acts = jpc.init_activities_with_ffwd(model=model, input=c_encs)
    return acts[-1].squeeze(-1)  # (n,)


def _mse(preds: Float[Array, "n"], targets: Float[Array, "n"]) -> float:
    return float(jnp.mean((preds - targets) ** 2))


def _visualize_pcn(
    model: list,
    cfg: PCNConfig,
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
    pred_mags_norm = _pcn_predict(model, c_encs)
    pred_mags = pred_mags_norm * ESCAPE_RADIUS

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
    fig.suptitle(f"PCN — Epoch {epoch}" if epoch is not None else "PCN — Final", fontsize=12)
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
    ax1.set_title("Mandelbrot PCN — grokking study (log–log)")
    ax1.grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=150)
    plt.close(fig)


def train(cfg: PCNConfig) -> list:
    key = jax.random.PRNGKey(cfg.seed)
    k_model, k_train, k_test = jax.random.split(key, 3)

    d = enc_dim(cfg.K)
    model = jpc.make_mlp(
        key=k_model,
        input_dim=d,
        width=cfg.hidden_dim,
        depth=cfg.depth,
        output_dim=1,
        act_fn=cfg.act_fn,
        use_bias=cfg.use_bias,
    )
    model = _tie_hidden_layers(model)  # start on the constraint manifold

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "config.json").write_text(json.dumps(dataclasses.asdict(cfg), indent=2))

    print("Generating datasets...")
    c_train, t_train = make_dataset(cfg.n_train, cfg.num_steps, k_train, train=True, K=cfg.K)
    c_test, t_test = make_dataset(cfg.n_test, cfg.num_steps, k_test, train=False, K=cfg.K)
    # JPC mse_loss expects (batch, output_dim) targets
    t_train_jpc = t_train[:, None]  # (n, 1)
    t_test_jpc = t_test[:, None]

    const_baseline = float(jnp.mean((t_train - jnp.mean(t_train)) ** 2))
    print(f"Constant-mean MSE baseline: {const_baseline:.4f}")

    steps_per_epoch = cfg.n_train // cfg.batch_size
    total_steps = steps_per_epoch * cfg.n_epochs
    warmup_steps = max(1, int(cfg.warmup_frac * total_steps))

    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=cfg.lr_peak,
        warmup_steps=warmup_steps,
        decay_steps=total_steps,
        end_value=0.0,
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(cfg.grad_clip),
        optax.adamw(learning_rate=schedule, weight_decay=cfg.weight_decay),
    )
    # JPC internally calls optim.update(params=(model, skip_model)), so the
    # optimizer must be initialized with the same (model, None) tuple structure.
    opt_state = optimizer.init(eqx.filter((model, None), eqx.is_array))

    @eqx.filter_jit
    def eval_mse(model: list, c_encs: jax.Array, targets: jax.Array) -> jax.Array:
        preds = _pcn_predict(model, c_encs)
        return jnp.mean((preds - targets) ** 2)

    rng = np.random.default_rng(cfg.seed)
    c_train_np = np.array(c_train)
    t_train_jpc_np = np.array(t_train_jpc)
    t_train_np = np.array(t_train)

    n_params = sum(x.size for x in jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array)))
    print(
        f"Model: enc_dim={d}, width={cfg.hidden_dim}, depth={cfg.depth}, "
        f"act={cfg.act_fn}, bias={cfg.use_bias}, params={n_params:,}"
    )
    print(f"Training: {cfg.n_train} samples, {steps_per_epoch} steps/epoch, {cfg.n_epochs} epochs")

    metrics: list[dict[str, Any]] = []
    ckpt_dir = DATA_DIR / "checkpoints"
    last_checkpoint_epoch = 0

    pbar = trange(cfg.n_epochs, desc="epochs")
    for epoch in pbar:
        t0 = time.perf_counter()
        idx = rng.permutation(cfg.n_train)
        epoch_loss = 0.0

        for i in range(steps_per_epoch):
            batch_idx = idx[i * cfg.batch_size : (i + 1) * cfg.batch_size]
            c_b = jnp.array(c_train_np[batch_idx])
            t_b = jnp.array(t_train_jpc_np[batch_idx])  # (batch, 1)
            result = jpc.make_pc_step(
                model=model,
                optim=optimizer,
                opt_state=opt_state,
                output=t_b,
                input=c_b,
                loss_id="mse",
                max_t1=cfg.max_t1,
                # weight_decay handled by optax.adamw; don't double-apply here
            )
            model = _tie_hidden_layers(result["model"])
            opt_state = result["opt_state"]
            # JPC loss = 0.5 * mean(sum(...)); convert to plain MSE for tracking
            epoch_loss += float(result["loss"]) * 2.0

        train_loss = epoch_loss / steps_per_epoch
        test_loss = float(eval_mse(model, c_test, t_test))
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
            _visualize_pcn(model, cfg, ckpt_dir / f"{tag}_pred.png", epoch=completed)

        metrics.append(record)

        if record.get("checkpoint"):
            (DATA_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
            plot_metrics(metrics, const_baseline, DATA_DIR / "train_curve.png")

        pbar.set_postfix(
            train=f"{train_loss:.4f}",
            test=f"{test_loss:.4f}",
            lr=f"{current_lr:.2e}",
            s=f"{elapsed:.1f}s",
        )

    (DATA_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    plot_metrics(metrics, const_baseline, DATA_DIR / "train_curve.png")
    print(f"Metrics → {DATA_DIR / 'metrics.json'}")
    print(f"Train curve → {DATA_DIR / 'train_curve.png'}")

    return model


def visualize(model: list, cfg: PCNConfig) -> None:
    out = DATA_DIR / "mandelbrot_prediction.png"
    _visualize_pcn(model, cfg, out)
    print(f"Saved {out}")


def main() -> None:
    cfg = PCNConfig()
    model = train(cfg)
    visualize(model, cfg)

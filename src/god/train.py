"""Training script for MandelbrotRNN.

Optimisation: AdamW + gradient clipping + single-cycle cosine LR with warmup.
Checkpoints saved to data/mandel/checkpoints/ every checkpoint_freq epochs.
"""

import dataclasses
import json
import time
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import numpy as np
from tqdm import trange

from god.encoding import K_DEFAULT, enc_dim, decode_magnitude
from god.model import MandelbrotRNN
from god.data import make_dataset, ESCAPE_RADIUS

DATA_DIR = Path("data/mandel")


@dataclass
class TrainConfig:
    # Model
    K: int = K_DEFAULT
    hidden_dim: int = 128
    depth: int = 2          # number of linear layers in the RNN cell
    num_steps: int = 10     # recurrent steps (= Mandelbrot iterations)
    linear_head: bool = False   # True = Linear(enc_dim→2); False = fixed h[0],h[1]

    # Data
    n_train: int = 10_000
    n_test: int = 2_000

    # Training
    batch_size: int = 256
    n_epochs: int = 30
    lr_peak: float = 1e-3
    warmup_frac: float = 0.05
    weight_decay: float = 0.01
    grad_clip: float = 1.0

    # Checkpointing
    checkpoint_freq: int = 5    # save every N epochs (0 to disable)

    # Reproducibility
    seed: int = 0


def _loss_fn(
    model: MandelbrotRNN,
    c_encs: jax.Array,
    targets: jax.Array,
) -> jax.Array:
    h_T = jax.vmap(model)(c_encs)                                           # (B, d)
    pred_mags = jax.vmap(model.predict_magnitude)(h_T)                      # (B,)
    return jnp.mean((pred_mags / ESCAPE_RADIUS - targets) ** 2)


def save_checkpoint(model: MandelbrotRNN, cfg: TrainConfig, epoch: int) -> None:
    ckpt_dir = DATA_DIR / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    eqx.tree_serialise_leaves(str(ckpt_dir / f"epoch_{epoch:04d}.eqx"), model)


def load_checkpoint(path: str | Path, cfg: TrainConfig) -> MandelbrotRNN:
    """Load a saved checkpoint. Requires a matching TrainConfig to reconstruct the template."""
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

    # Persist config alongside checkpoints
    if cfg.checkpoint_freq > 0:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        (DATA_DIR / "config.json").write_text(
            json.dumps(dataclasses.asdict(cfg), indent=2)
        )

    # Datasets
    print("Generating datasets...")
    c_train, t_train = make_dataset(cfg.n_train, cfg.num_steps, k_train, train=True, K=cfg.K)
    c_test, t_test = make_dataset(cfg.n_test, cfg.num_steps, k_test, train=False, K=cfg.K)
    const_baseline = float(jnp.mean((t_train - jnp.mean(t_train)) ** 2))
    print(f"Constant-mean MSE baseline: {const_baseline:.4f}")

    # Optimiser
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
        pbar.set_postfix(
            train=f"{train_loss:.4f}",
            test=f"{test_loss:.4f}",
            s=f"{elapsed:.1f}s",
        )

        if cfg.checkpoint_freq > 0 and (epoch + 1) % cfg.checkpoint_freq == 0:
            save_checkpoint(model, cfg, epoch + 1)

    return model


def visualize(model: MandelbrotRNN, cfg: TrainConfig) -> None:
    """Plot true vs. predicted Mandelbrot magnitude on a 2D grid."""
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

    _fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    kw = dict(origin="lower", extent=[-2, 2, -2, 2], vmin=0, vmax=ESCAPE_RADIUS)

    axes[0].imshow(true_grid, **kw)
    axes[0].set_title("True |z_T|")
    axes[1].imshow(pred_grid, **kw)
    axes[1].set_title("Predicted |z_T|")
    err = np.abs(true_grid - pred_grid)
    axes[2].imshow(err, origin="lower", extent=[-2, 2, -2, 2])
    axes[2].set_title("Absolute error")

    for ax in axes:
        ax.axhline(0, color="white", lw=0.5, ls="--")
        ax.set_xlabel("Re(c)")
        ax.set_ylabel("Im(c)")

    plt.tight_layout()
    out = DATA_DIR / "mandelbrot_prediction.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out), dpi=150)
    print(f"Saved {out}")
    plt.show()


def main() -> None:
    cfg = TrainConfig()
    model = train(cfg)
    visualize(model, cfg)

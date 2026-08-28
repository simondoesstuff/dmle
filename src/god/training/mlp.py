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

from god.datasets.mandelbrot import ESCAPE_RADIUS, K_DEFAULT, enc_dim, make_mandelbrot_dataset
from god.models.rnn import MandelbrotRNN
from god.training.common import checkpoint_interval, plot_metrics_standard, visualize_mandelbrot


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
    warmup_frac: float = 0.0
    lr_end: float = 1e-4
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
    def predict(c_encs: jax.Array) -> jax.Array:
        h_T = jax.vmap(model)(c_encs)
        return jax.vmap(model.predict_magnitude)(h_T)

    title = f"Epoch {epoch}" if epoch is not None else "Final"
    visualize_mandelbrot(predict, out_path, cfg.num_steps, cfg.K, title)


def load_checkpoint(path: str | Path, cfg: TrainConfig) -> MandelbrotRNN:
    d = enc_dim(cfg.K)
    template = MandelbrotRNN(
        d, cfg.hidden_dim, cfg.depth, cfg.num_steps,
        jax.random.PRNGKey(0), linear_head=cfg.linear_head,
    )
    return eqx.tree_deserialise_leaves(str(path), template)


def train(cfg: TrainConfig) -> MandelbrotRNN:
    key = jax.random.PRNGKey(cfg.seed)
    k_model, k_data = jax.random.split(key)

    d = enc_dim(cfg.K)
    model = MandelbrotRNN(
        d, cfg.hidden_dim, cfg.depth, cfg.num_steps, k_model,
        linear_head=cfg.linear_head,
    )

    data_dir = Path(cfg.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "config.json").write_text(json.dumps(dataclasses.asdict(cfg), indent=2))

    print("Generating datasets...")
    dataset = make_mandelbrot_dataset(cfg.n_train, cfg.n_test, cfg.num_steps, k_data, K=cfg.K)
    c_train, t_train = dataset.train_inputs, dataset.train_targets
    c_test, t_test = dataset.test_inputs, dataset.test_targets
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
        current_lr = float(jnp.asarray(schedule(completed * steps_per_epoch)))

        record: dict[str, Any] = {
            "epoch": completed,
            "train_loss": train_loss,
            "test_loss": test_loss,
            "lr": current_lr,
        }

        interval = checkpoint_interval(
            current_lr, cfg.lr_peak, cfg.checkpoint_min_interval, cfg.checkpoint_max_interval
        )
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
            plot_metrics_standard(
                metrics, const_baseline, data_dir / "train_curve.png",
                "Mandelbrot RNN — training (log–log)",
            )

        pbar.set_postfix(
            train=f"{train_loss:.4f}",
            test=f"{test_loss:.4f}",
            lr=f"{current_lr:.2e}",
            s=f"{elapsed:.1f}s",
        )

    (data_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    plot_metrics_standard(
        metrics, const_baseline, data_dir / "train_curve.png",
        "Mandelbrot RNN — training (log–log)",
    )
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

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

from god.datasets.mandelbrot import ESCAPE_RADIUS, K_DEFAULT, enc_dim, make_mandelbrot_dataset
from god.training.common import checkpoint_interval, plot_metrics_standard, visualize_mandelbrot


@dataclass
class PCNConfig:
    # Paths
    data_dir: str = "data/mandel/pcn"

    # Model
    K: int = K_DEFAULT
    hidden_dim: int = 128
    depth: int = 10
    use_bias: bool = True
    act_fn: str = "tanh"

    # Data
    n_train: int = 10_000
    n_test: int = 2_000
    num_steps: int = 10

    # Training
    batch_size: int = 256
    n_epochs: int = 3000
    lr_peak: float = 1e-3
    warmup_frac: float = 0.05
    weight_decay: float = 0.01
    grad_clip: float = 1.0

    # PC inference
    max_t1: int = 20

    # Checkpointing
    checkpoint_min_interval: int = 10
    checkpoint_max_interval: int = 300

    seed: int = 0


@eqx.filter_jit
def _tie_hidden_layers(model: list[Any]) -> list[Any]:
    """Project to weight-tied manifold by averaging hidden layer parameters."""
    depth = len(model)
    if depth <= 3:
        return model
    hidden = list(range(1, depth - 1))
    W_avg = jnp.mean(jnp.stack([model[i].layers[1].weight for i in hidden]), axis=0)
    for i in hidden:
        model = eqx.tree_at(lambda m, _i=i: m[_i].layers[1].weight, model, W_avg)
    if model[hidden[0]].layers[1].bias is not None:
        b_avg = jnp.mean(jnp.stack([model[i].layers[1].bias for i in hidden]), axis=0)
        for i in hidden:
            model = eqx.tree_at(lambda m, _i=i: m[_i].layers[1].bias, model, b_avg)
    return model


def _pcn_predict(
    model: list[Any],
    c_encs: Float[Array, "n d"],
) -> Float[Array, "n"]:
    """Feedforward prediction (no inference loop) — used for eval and visualisation."""
    acts = jpc.init_activities_with_ffwd(model=model, input=c_encs)
    return acts[-1].squeeze(-1)


def _mse(preds: Float[Array, "n"], targets: Float[Array, "n"]) -> float:
    return float(jnp.mean((preds - targets) ** 2))


def _visualize_pcn(
    model: list[Any],
    cfg: PCNConfig,
    out_path: Path,
    epoch: int | None = None,
) -> None:
    def predict(c_encs: jax.Array) -> jax.Array:
        return _pcn_predict(model, c_encs) * ESCAPE_RADIUS

    title = f"PCN — Epoch {epoch}" if epoch is not None else "PCN — Final"
    visualize_mandelbrot(predict, out_path, cfg.num_steps, cfg.K, title)


def train(cfg: PCNConfig) -> list[Any]:
    key = jax.random.PRNGKey(cfg.seed)
    k_model, k_data = jax.random.split(key)

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
    model = _tie_hidden_layers(model)

    data_dir = Path(cfg.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "config.json").write_text(json.dumps(dataclasses.asdict(cfg), indent=2))

    print("Generating datasets...")
    dataset = make_mandelbrot_dataset(cfg.n_train, cfg.n_test, cfg.num_steps, k_data, K=cfg.K)
    c_train, t_train = dataset.train_inputs, dataset.train_targets
    c_test, t_test = dataset.test_inputs, dataset.test_targets
    t_train_jpc = t_train[:, None]

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
    opt_state = optimizer.init(eqx.filter((model, None), eqx.is_array))

    @eqx.filter_jit
    def eval_mse(model: list[Any], c_encs: jax.Array, targets: jax.Array) -> jax.Array:
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
            t_b = jnp.array(t_train_jpc_np[batch_idx])
            result = jpc.make_pc_step(
                model=model,
                optim=optimizer,
                opt_state=opt_state,
                output=t_b,
                input=c_b,
                loss_id="mse",
                max_t1=cfg.max_t1,
            )
            model = _tie_hidden_layers(result["model"])
            opt_state = result["opt_state"]
            epoch_loss += float(result["loss"]) * 2.0

        train_loss = epoch_loss / steps_per_epoch
        test_loss = float(eval_mse(model, c_test, t_test))
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
            _visualize_pcn(model, cfg, ckpt_dir / f"{tag}_pred.png", epoch=completed)

        metrics.append(record)

        if record.get("checkpoint"):
            (data_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
            plot_metrics_standard(
                metrics, const_baseline, data_dir / "train_curve.png",
                "Mandelbrot PCN — grokking study (log–log)",
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
        "Mandelbrot PCN — grokking study (log–log)",
    )
    print(f"Metrics → {data_dir / 'metrics.json'}")
    print(f"Train curve → {data_dir / 'train_curve.png'}")

    return model


def visualize(model: list[Any], cfg: PCNConfig) -> None:
    out = Path(cfg.data_dir) / "mandelbrot_prediction.png"
    _visualize_pcn(model, cfg, out)
    print(f"Saved {out}")


def main() -> None:
    cfg = PCNConfig()
    model = train(cfg)
    visualize(model, cfg)

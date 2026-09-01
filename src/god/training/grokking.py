"""Grokking: modular addition (a + b) mod 97 with a tanh FFN.

Reproduces the delayed-generalization (double-descent) curve from Power et al.
(2022). Key ingredients: AdamW with strong weight decay (~1.0), ~30% train
split, and training long enough for the grok (~50k–200k epochs).

Entry points:
  god-grokking       — 50k epochs (captures memorization + onset of grok)
  god-grokking-long  — 200k epochs from scratch (full grok)
  god-grokking-cont  — continue from a saved checkpoint for more epochs
"""

import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from tqdm import trange

from god.datasets.modular import ModularConfig, Op, input_dim, make_modular_dataset
from god.models.ffn import TanhFFN


@dataclass
class TrainConfig:
    data_dir: str = "data/grokking"

    # Model
    hidden_dim: int = 200
    depth: int = 2

    # Data
    modulus: int = 97
    train_fraction: float = 0.3

    # Training — full-batch AdamW; weight_decay is the critical knob
    n_epochs: int = 50_000
    lr: float = 1e-3
    weight_decay: float = 1.0

    # Record accuracy/loss every log_interval epochs
    log_interval: int = 50

    # Reproducibility
    seed: int = 0


def _loss(model: TanhFFN, inputs: jax.Array, targets: jax.Array) -> jax.Array:
    logits = jax.vmap(model)(inputs)
    return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, targets))


def _accuracy(model: TanhFFN, inputs: jax.Array, targets: jax.Array) -> jax.Array:
    logits = jax.vmap(model)(inputs)
    return jnp.mean(jnp.argmax(logits, axis=-1) == targets)


def _plot(metrics: list[dict[str, Any]], out_path: Path) -> None:
    import matplotlib.pyplot as plt

    epochs = [m["epoch"] for m in metrics]
    train_acc = [m["train_acc"] for m in metrics]
    test_acc = [m["test_acc"] for m in metrics]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs, train_acc, label="train accuracy", lw=1.5, color="tab:blue")
    ax.plot(epochs, test_acc, label="test accuracy", lw=1.5, color="tab:orange")
    ax.set_xscale("log")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("epoch (log scale)")
    ax.set_ylabel("accuracy")
    ax.set_title("Grokking — (a+b) mod 97 — FFN")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=150)
    plt.close(fig)


def _run_loop(
    model: TanhFFN,
    optimizer: optax.GradientTransformation,
    opt_state: optax.OptState,
    dataset: Any,
    cfg: TrainConfig,
    n_epochs: int,
    start_epoch: int,
    existing_metrics: list[dict[str, Any]],
) -> tuple[TanhFFN, list[dict[str, Any]]]:
    """Inner training loop; appends to existing_metrics and returns updated model."""

    @eqx.filter_jit
    def step(
        model: TanhFFN,
        opt_state: optax.OptState,
        inputs: jax.Array,
        targets: jax.Array,
    ) -> tuple[TanhFFN, optax.OptState, jax.Array]:
        loss, grads = eqx.filter_value_and_grad(_loss)(model, inputs, targets)
        updates, new_state = optimizer.update(
            grads, opt_state, eqx.filter(model, eqx.is_array)
        )
        return eqx.apply_updates(model, updates), new_state, loss

    @eqx.filter_jit
    def eval_metrics(
        model: TanhFFN, inputs: jax.Array, targets: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        return _loss(model, inputs, targets), _accuracy(model, inputs, targets)

    metrics = list(existing_metrics)
    pbar = trange(n_epochs, desc="epochs")
    for i in pbar:
        epoch = start_epoch + i
        model, opt_state, _ = step(
            model, opt_state, dataset.train_inputs, dataset.train_targets
        )

        if (epoch + 1) % cfg.log_interval == 0 or (start_epoch == 0 and i == 0):
            tr_loss, tr_acc = eval_metrics(
                model, dataset.train_inputs, dataset.train_targets
            )
            te_loss, te_acc = eval_metrics(
                model, dataset.test_inputs, dataset.test_targets
            )
            record: dict[str, Any] = {
                "epoch": epoch + 1,
                "train_loss": float(tr_loss),
                "test_loss": float(te_loss),
                "train_acc": float(tr_acc),
                "test_acc": float(te_acc),
            }
            metrics.append(record)
            pbar.set_postfix(
                tr=f"{float(tr_acc):.3f}",
                te=f"{float(te_acc):.3f}",
            )

    return model, metrics


def _load_dataset_and_template(cfg: TrainConfig) -> tuple[Any, TanhFFN, int]:
    key = jax.random.PRNGKey(cfg.seed)
    k_model, k_data = jax.random.split(key)
    dataset_cfg = ModularConfig(
        modulus=cfg.modulus, ops=(Op.ADD,), train_fraction=cfg.train_fraction
    )
    dataset = make_modular_dataset(dataset_cfg, k_data)
    in_dim = input_dim(dataset_cfg)
    template = TanhFFN(in_dim, cfg.hidden_dim, cfg.depth, cfg.modulus, k_model)
    return dataset, template, in_dim


def train(cfg: TrainConfig) -> TanhFFN:
    data_dir = Path(cfg.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "config.json").write_text(json.dumps(dataclasses.asdict(cfg), indent=2))

    dataset, model, in_dim = _load_dataset_and_template(cfg)

    n_params = sum(x.size for x in jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array)))
    print(
        f"Model: in={in_dim}, hidden={cfg.hidden_dim}×{cfg.depth}, "
        f"out={cfg.modulus}, params={n_params:,}"
    )
    print(
        f"Data:  {dataset.train_inputs.shape[0]} train / "
        f"{dataset.test_inputs.shape[0]} test  ({cfg.train_fraction:.0%} split)"
    )

    optimizer = optax.adamw(learning_rate=cfg.lr, weight_decay=cfg.weight_decay)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    model, metrics = _run_loop(
        model, optimizer, opt_state, dataset, cfg,
        n_epochs=cfg.n_epochs, start_epoch=0, existing_metrics=[],
    )

    (data_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    eqx.tree_serialise_leaves(str(data_dir / "model.eqx"), model)
    curve_path = data_dir / "grokking_curve.png"
    _plot(metrics, curve_path)
    print(f"\nMetrics → {data_dir / 'metrics.json'}")
    print(f"Curve   → {curve_path}")
    print(f"Model   → {data_dir / 'model.eqx'}")

    return model


def continue_train(cfg: TrainConfig, additional_epochs: int) -> TanhFFN:
    """Load a saved checkpoint and continue training for more epochs.

    The optimizer restarts cold (moments reset), which causes a brief loss
    spike but does not affect long-run convergence.
    """
    data_dir = Path(cfg.data_dir)
    model_path = data_dir / "model.eqx"
    metrics_path = data_dir / "metrics.json"

    if not model_path.exists():
        raise FileNotFoundError(
            f"No checkpoint at {model_path}. Run train() first."
        )

    dataset, template, in_dim = _load_dataset_and_template(cfg)
    model = eqx.tree_deserialise_leaves(str(model_path), template)
    existing_metrics: list[dict[str, Any]] = json.loads(metrics_path.read_text())
    start_epoch = existing_metrics[-1]["epoch"]

    print(f"Resuming from epoch {start_epoch}, running {additional_epochs} more")

    optimizer = optax.adamw(learning_rate=cfg.lr, weight_decay=cfg.weight_decay)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    model, metrics = _run_loop(
        model, optimizer, opt_state, dataset, cfg,
        n_epochs=additional_epochs, start_epoch=start_epoch,
        existing_metrics=existing_metrics,
    )

    (data_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    eqx.tree_serialise_leaves(str(model_path), model)
    curve_path = data_dir / "grokking_curve.png"
    _plot(metrics, curve_path)
    print(f"\nMetrics → {data_dir / 'metrics.json'}")
    print(f"Curve   → {curve_path}")

    return model


def main() -> None:
    train(TrainConfig())


def main_long() -> None:
    train(TrainConfig(n_epochs=200_000))


def main_cont() -> None:
    continue_train(TrainConfig(), additional_epochs=150_000)

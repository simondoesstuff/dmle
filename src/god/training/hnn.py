"""Hypernetwork (HNN) training for Mandelbrot magnitude prediction.

Stimulus x ~ Uniform[-1, 1]^n_stimulus is sampled fresh each step.
A meta-batch of n_mc_train noise vectors is drawn per step; all share the
same (c_enc, target) batch so different target networks are compared fairly.

Loss per step:
    MSE averaged over n_mc_train networks + lambda_target_decay * mean(||params||²)

Diversity metrics are logged at every checkpoint to detect stimulus collapse.
Resumption: `train()` auto-detects the latest checkpoint in data_dir/checkpoints/
and continues from there if one exists.
See docs/hnn.md for full architecture description.
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

from god.datasets.mandelbrot import ESCAPE_RADIUS, K_DEFAULT, make_mandelbrot_dataset
from god.models.hnn import HyperNetwork, make_hypernetwork, n_trainable_params, target_network_diversity
from god.training.common import visualize_mandelbrot


@dataclass
class HNNConfig:
    data_dir: str = "data/mandel/hnn"

    # Encoding
    K: int = K_DEFAULT

    # Target RNN cell — depth=1: one hidden tanh layer + linear output (TanhFFN convention)
    n_stimulus: int = 16
    rnn_hidden_dim: int = 16
    rnn_depth: int = 1

    # Stimulus FFN
    stim_ffn_hidden: int = 64
    stim_ffn_depth: int = 1
    init_scale: float = 0.1

    # Data
    n_train: int = 10_000
    n_test: int = 2_000
    num_steps: int = 10  # Mandelbrot iterations = RNN recurrences

    # Training
    n_steps: int = 150_000
    batch_size: int = 64
    lr: float = 3e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    n_mc_train: int = 8
    n_mc_eval: int = 32

    # L2 regularisation on generated target net parameters
    lambda_target_decay: float = 1e-3

    # Cosine annealing LR with warmup
    warmup_frac: float = 0.3
    lr_end_frac: float = 0.01

    checkpoint_interval: int = 500
    seed: int = 0


# ── checkpoint helpers ────────────────────────────────────────────────────────

def _ckpt_tag(step: int) -> str:
    return f"step_{step:06d}"


def _make_model_template(cfg: HNNConfig, key: jax.Array | None = None) -> HyperNetwork:
    if key is None:
        key = jax.random.split(jax.random.PRNGKey(cfg.seed), 5)[0]
    return make_hypernetwork(
        key,
        K=cfg.K,
        n_stimulus=cfg.n_stimulus,
        rnn_hidden_dim=cfg.rnn_hidden_dim,
        rnn_depth=cfg.rnn_depth,
        rnn_num_steps=cfg.num_steps,
        stim_ffn_hidden=cfg.stim_ffn_hidden,
        stim_ffn_depth=cfg.stim_ffn_depth,
        init_scale=cfg.init_scale,
    )


def _find_latest_checkpoint(ckpt_dir: Path) -> int | None:
    """Return the step number of the most recent checkpoint, or None."""
    metas = list(ckpt_dir.glob("step_*_meta.json"))
    if metas:
        return max(int(p.stem.replace("_meta", "").split("_")[1]) for p in metas)
    eqxs = [p for p in ckpt_dir.glob("step_*.eqx") if "_opt" not in p.name]
    if eqxs:
        return max(int(p.stem.split("_")[1]) for p in eqxs)
    return None


def _save_checkpoint(
    ckpt_dir: Path,
    completed: int,
    model: HyperNetwork,
    opt_state: optax.OptState,
    rng: np.random.Generator,
) -> None:
    tag = _ckpt_tag(completed)
    eqx.tree_serialise_leaves(str(ckpt_dir / f"{tag}.eqx"), model)
    eqx.tree_serialise_leaves(str(ckpt_dir / f"{tag}_opt.eqx"), opt_state)
    meta = {
        "step": completed,
        "numpy_rng_state": rng.bit_generator.state,
    }
    (ckpt_dir / f"{tag}_meta.json").write_text(json.dumps(meta))


def _load_checkpoint(
    ckpt_dir: Path,
    step: int,
    model_template: HyperNetwork,
    opt_state_template: optax.OptState,
) -> tuple[HyperNetwork, optax.OptState, np.random.Generator]:
    tag = _ckpt_tag(step)
    model = eqx.tree_deserialise_leaves(str(ckpt_dir / f"{tag}.eqx"), model_template)

    opt_path = ckpt_dir / f"{tag}_opt.eqx"
    if opt_path.exists():
        opt_state = eqx.tree_deserialise_leaves(str(opt_path), opt_state_template)
    else:
        opt_state = opt_state_template

    rng = np.random.default_rng()
    meta_path = ckpt_dir / f"{tag}_meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        rng.bit_generator.state = meta["numpy_rng_state"]
    else:
        rng = np.random.default_rng(step)

    return model, opt_state, rng


# ── loss / eval helpers ───────────────────────────────────────────────────────

def _task_loss(
    model: HyperNetwork,
    c_encs: jax.Array,
    targets: jax.Array,
    x_samples: jax.Array,
    lambda_target_decay: float = 0.0,
) -> jax.Array:
    def loss_for_x(x: jax.Array) -> jax.Array:
        preds = jax.vmap(model, in_axes=(0, None))(c_encs, x)
        return jnp.mean((preds - targets) ** 2)

    mse = jnp.mean(jax.vmap(loss_for_x)(x_samples))
    if lambda_target_decay > 0.0:
        l2 = jnp.mean(
            jax.vmap(lambda x: jnp.mean(model.target_params(x) ** 2))(x_samples)
        )
        return mse + lambda_target_decay * l2
    return mse


def _mc_predict(
    model: HyperNetwork,
    c_encs: jax.Array,
    x_samples: jax.Array,
) -> jax.Array:
    """E_x[f(c_enc, x)] averaged over pre-drawn noise samples (n_mc, n_stimulus)."""
    per_x = jax.vmap(
        lambda x: jax.vmap(model, in_axes=(0, None))(c_encs, x)
    )(x_samples)
    return jnp.mean(per_x, axis=0)


# ── visualisation / plotting ──────────────────────────────────────────────────

def _visualize_hnn(
    model: HyperNetwork,
    cfg: HNNConfig,
    out_path: Path,
    eval_x_samples: jax.Array,
    step: int | None = None,
) -> None:
    def predict(c_encs: jax.Array) -> jax.Array:
        return _mc_predict(model, c_encs, eval_x_samples) * ESCAPE_RADIUS

    n_mc = len(eval_x_samples)
    title = f"HNN — {'Step ' + str(step) if step else 'Final'} (MC n={n_mc})"
    visualize_mandelbrot(predict, out_path, cfg.num_steps, cfg.K, title)


def plot_metrics(metrics: list[dict[str, Any]], baseline: float, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    steps     = [m["step"] for m in metrics]
    train     = [m["train_loss"] for m in metrics]
    test      = [m["test_loss"] for m in metrics]
    param_std = [m["param_std"] for m in metrics]
    pred_std  = [m["pred_std"] for m in metrics]

    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    axes[0].plot(steps, train, label="train MSE", lw=1.5, color="tab:blue")
    axes[0].plot(steps, test,  label="test MSE",  lw=1.5, color="tab:orange")
    axes[0].axhline(baseline, color="gray", ls="--", lw=1, label=f"baseline ({baseline:.3f})")
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("MSE (log)")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3, which="both")
    axes[0].set_title("HNN — loss")
    axes[1].plot(steps, param_std, lw=1.5, color="tab:purple")
    axes[1].set_ylabel("param std across x")
    axes[1].set_title("Target network parameter diversity")
    axes[1].grid(True, alpha=0.3)
    axes[2].plot(steps, pred_std, lw=1.5, color="tab:red")
    axes[2].set_ylabel("prediction std across x")
    axes[2].set_xlabel("step")
    axes[2].set_title("Target network functional diversity")
    axes[2].grid(True, alpha=0.3)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=150)
    plt.close(fig)


# ── main training loop ────────────────────────────────────────────────────────

def train(cfg: HNNConfig) -> HyperNetwork:
    key = jax.random.PRNGKey(cfg.seed)
    k_model, k_data, k_noise, k_eval = jax.random.split(key, 4)

    data_dir = Path(cfg.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = data_dir / "checkpoints"

    model = _make_model_template(cfg, k_model)

    print("Generating datasets...")
    dataset = make_mandelbrot_dataset(cfg.n_train, cfg.n_test, cfg.num_steps, k_data, K=cfg.K)
    c_train, t_train = dataset.train_inputs, dataset.train_targets
    c_test,  t_test  = dataset.test_inputs,  dataset.test_targets
    const_baseline = float(jnp.mean((t_train - jnp.mean(t_train)) ** 2))

    eval_x_samples = jax.random.uniform(
        k_eval, (cfg.n_mc_eval, cfg.n_stimulus), minval=-1.0, maxval=1.0
    )

    total_steps  = cfg.n_steps
    warmup_steps = int(cfg.warmup_frac * total_steps)
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=cfg.lr * cfg.lr_end_frac,
        peak_value=cfg.lr,
        warmup_steps=warmup_steps,
        decay_steps=total_steps,
        end_value=cfg.lr * cfg.lr_end_frac,
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(cfg.grad_clip),
        optax.adamw(learning_rate=schedule, weight_decay=cfg.weight_decay),
    )
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    rng = np.random.default_rng(cfg.seed)

    # ── Resume detection ──────────────────────────────────────────────────────
    resume_step = _find_latest_checkpoint(ckpt_dir)
    if resume_step is not None and resume_step >= cfg.n_steps:
        print(f"Already complete at step {resume_step}; nothing to do.")
        return _load_checkpoint(ckpt_dir, resume_step, model, opt_state)[0]

    if resume_step is not None:
        print(f"Resuming from step {resume_step} → {cfg.n_steps} ...")
        model, opt_state, rng = _load_checkpoint(ckpt_dir, resume_step, model, opt_state)
        metrics: list[dict[str, Any]] = [
            m for m in json.loads((data_dir / "metrics.json").read_text())
            if m["step"] <= resume_step
        ]
        start_step = resume_step
    else:
        print(f"Starting fresh → {cfg.n_steps} steps")
        (data_dir / "config.json").write_text(json.dumps(dataclasses.asdict(cfg), indent=2))
        metrics = []
        start_step = 0

    print(f"Constant-mean MSE baseline: {const_baseline:.4f}")
    print(f"HNN trainable params: {n_trainable_params(model):,}")

    # ── JIT-compiled step ─────────────────────────────────────────────────────
    lambda_target_decay = cfg.lambda_target_decay

    @eqx.filter_jit
    def do_step(
        model: HyperNetwork,
        opt_state: optax.OptState,
        c_b: jax.Array,
        t_b: jax.Array,
        x_samples: jax.Array,
    ) -> tuple[HyperNetwork, optax.OptState, jax.Array]:
        loss, grads = eqx.filter_value_and_grad(
            lambda m: _task_loss(m, c_b, t_b, x_samples, lambda_target_decay)
        )(model)
        updates, new_state = optimizer.update(grads, opt_state, eqx.filter(model, eqx.is_array))
        return eqx.apply_updates(model, updates), new_state, loss

    @eqx.filter_jit
    def eval_loss(model: HyperNetwork, c_encs: jax.Array, targets: jax.Array) -> jax.Array:
        preds = _mc_predict(model, c_encs, eval_x_samples)
        return jnp.mean((preds - targets) ** 2)

    c_train_np = np.array(c_train)
    t_train_np = np.array(t_train)

    # ── Training loop ─────────────────────────────────────────────────────────
    pbar = trange(start_step, cfg.n_steps, desc="steps", initial=start_step, total=cfg.n_steps)
    for s in pbar:
        t0 = time.perf_counter()

        k_noise, subkey = jax.random.split(k_noise)
        x_samples = jax.random.uniform(
            subkey, (cfg.n_mc_train, cfg.n_stimulus), minval=-1.0, maxval=1.0
        )
        batch_idx = rng.choice(cfg.n_train, size=cfg.batch_size, replace=False)
        c_b = jnp.array(c_train_np[batch_idx])
        t_b = jnp.array(t_train_np[batch_idx])

        model, opt_state, loss = do_step(model, opt_state, c_b, t_b, x_samples)

        completed = s + 1
        if (completed % cfg.checkpoint_interval == 0) or (completed == cfg.n_steps):
            test_loss  = float(eval_loss(model, c_test,  t_test))
            train_loss = float(eval_loss(model, c_train, t_train))
            diversity  = target_network_diversity(model, c_b, eval_x_samples)
            elapsed    = time.perf_counter() - t0

            metrics.append({
                "step": completed,
                "train_loss": train_loss,
                "test_loss": test_loss,
                **diversity,
            })

            ckpt_dir.mkdir(parents=True, exist_ok=True)
            _save_checkpoint(ckpt_dir, completed, model, opt_state, rng)
            tag = _ckpt_tag(completed)
            _visualize_hnn(model, cfg, ckpt_dir / f"{tag}_pred.png", eval_x_samples, step=completed)

            (data_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
            plot_metrics(metrics, const_baseline, data_dir / "train_curve.png")

            pbar.set_postfix(
                train=f"{train_loss:.4f}",
                test=f"{test_loss:.4f}",
                p_std=f"{diversity['param_std']:.4f}",
                f_std=f"{diversity['pred_std']:.4f}",
                s=f"{elapsed:.1f}s",
            )

    (data_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    plot_metrics(metrics, const_baseline, data_dir / "train_curve.png")
    return model


def load_checkpoint(path: str | Path, cfg: HNNConfig) -> HyperNetwork:
    """Load a model checkpoint by explicit path."""
    template = _make_model_template(cfg)
    return eqx.tree_deserialise_leaves(str(path), template)


def main() -> None:
    cfg = HNNConfig()
    train(cfg)

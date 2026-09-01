"""Hypernetwork (HNN) training for Mandelbrot magnitude prediction.

Stimulus x ~ Uniform[-1, 1]^n_stimulus is sampled fresh each step.
A meta-batch of n_mc_train noise vectors is drawn per step; all share the
same (c_enc, target) batch so different target networks are compared fairly.

Loss curriculum:
    Phase 1 (steps 0 .. softmax_switch_frac*n_steps): soft-min over per-x MSEs.
        Gradient concentrates on the best-performing target network, leaving the
        rest of the stimulus space free → diversity preserved.
    Phase 2 (remaining steps): soft-max over per-x MSEs.
        Gradient concentrates on the worst-performing target network, forcing
        the stimulus FFN to produce good predictions for all x → collapse onto
        best solution.

    soft_aggregate(l, T, alpha) = (T/alpha) * log(mean(exp(alpha * l / T)))
        alpha=-1, T→0  →  min(l)   [softmin]
        alpha=+1, T→0  →  max(l)   [softmax]
        any alpha, T→∞ →  mean(l)

Temperature is adaptive: T_eff = scale × std(per_x_losses), so selection pressure
stays meaningful as losses shrink during training.  The scale hyperparameter
(softmin_temp_scale / softmax_temp_scale) is dimensionless and loss-scale-invariant.

Diversity metrics (param_std, pred_std, eff_n, eff_temp) are logged at every checkpoint.
eff_n = 1 / sum(w_i²) measures how concentrated the soft-min/max selection is;
    eff_n ≈ n_mc  → uniform (scale too high, degenerates to mean)
    eff_n ≈ 1     → hard selection (scale very low)
    sweet spot:   eff_n ≈ 2–3

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

from god.datasets.mandelbrot import ESCAPE_RADIUS, K_DEFAULT, encode, make_mandelbrot_dataset, mandelbrot_mag
from god.models.hnn import HyperNetwork, make_hypernetwork, n_trainable_params, target_network_diversity


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

    # Soft min/max curriculum
    # Temperature is adaptive: T_eff = scale × std(per_x_losses), so it tracks
    # the actual spread of losses across MC samples regardless of loss magnitude.
    softmin_temp_scale: float = 1.0   # multiplier of loss-std during softmin phase (alpha=-1)
    softmax_temp_scale: float = 0.3   # multiplier of loss-std during softmax phase (alpha=+1)
    softmax_switch_frac: float = 0.9  # fraction of total steps before switching to softmax
    softmax_lr_floor: float = 1e-4    # minimum LR during softmax phase (avoids LR starvation)

    checkpoint_interval: int = 500
    seed: int = 0


# ── soft aggregate helpers ────────────────────────────────────────────────────

def _soft_aggregate(
    losses: jax.Array,
    temperature: float | jax.Array,
    alpha: float | jax.Array,
) -> jax.Array:
    """Differentiable soft-min (alpha=-1) or soft-max (alpha=+1) over losses.

    Formula: (T/alpha) * log(mean(exp(alpha * l / T)))
    Uses the log-sum-exp trick for numerical stability.
    """
    scaled = alpha * losses / temperature
    log_mean_exp = jax.scipy.special.logsumexp(scaled) - jnp.log(losses.shape[0])
    return (temperature / alpha) * log_mean_exp


def _soft_weights(
    losses: jax.Array,
    temperature: float | jax.Array,
    alpha: float | jax.Array,
) -> jax.Array:
    """Softmax weights used by the soft-aggregate, useful for computing eff_n."""
    return jax.nn.softmax(alpha * losses / temperature)


def _adaptive_temp(losses: jax.Array, scale: float | jax.Array) -> jax.Array:
    """T_eff = scale × std(losses), floored to avoid div-by-zero at collapse."""
    return jnp.maximum(scale * jnp.std(losses), jnp.array(1e-8))


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
    alpha: float | jax.Array = -1.0,
    temp_scale: float | jax.Array = 1.0,
) -> jax.Array:
    def loss_for_x(x: jax.Array) -> jax.Array:
        preds = jax.vmap(model, in_axes=(0, None))(c_encs, x)
        return jnp.mean((preds - targets) ** 2)

    per_x_losses = jax.vmap(loss_for_x)(x_samples)  # (n_mc,)
    T_eff = _adaptive_temp(per_x_losses, temp_scale)
    task = _soft_aggregate(per_x_losses, T_eff, alpha)

    if lambda_target_decay > 0.0:
        l2 = jnp.mean(
            jax.vmap(lambda x: jnp.mean(model.target_params(x) ** 2))(x_samples)
        )
        return task + lambda_target_decay * l2
    return task


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
    """4-panel Mandelbrot plot: target | best bulb | worst bulb | average."""
    import matplotlib.pyplot as plt

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

    # (n_mc, n_points) predictions in Mandelbrot scale
    per_x_preds = jax.vmap(
        lambda x: jax.vmap(model, in_axes=(0, None))(c_encs, x) * ESCAPE_RADIUS
    )(eval_x_samples)

    per_x_mse = jnp.mean((per_x_preds - true_mags[None, :]) ** 2, axis=1)
    best_idx = int(jnp.argmin(per_x_mse))
    worst_idx = int(jnp.argmax(per_x_mse))

    true_grid  = np.array(true_mags).reshape(res, res)
    best_grid  = np.array(per_x_preds[best_idx]).reshape(res, res)
    worst_grid = np.array(per_x_preds[worst_idx]).reshape(res, res)
    avg_grid   = np.array(jnp.mean(per_x_preds, axis=0)).reshape(res, res)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    kw = dict(origin="lower", extent=[-2, 2, -2, 2], vmin=0, vmax=ESCAPE_RADIUS, cmap="inferno")

    axes[0].imshow(true_grid, **kw)
    axes[0].set_title("Target")
    axes[1].imshow(best_grid, **kw)
    axes[1].set_title(f"Best (MSE={float(per_x_mse[best_idx]):.4f})")
    axes[2].imshow(worst_grid, **kw)
    axes[2].set_title(f"Worst (MSE={float(per_x_mse[worst_idx]):.4f})")
    axes[3].imshow(avg_grid, **kw)
    axes[3].set_title("Average")

    for ax in axes:
        ax.axhline(0, color="white", lw=0.5, ls="--")
        ax.set_xlabel("Re(c)")
        ax.set_ylabel("Im(c)")

    n_mc = len(eval_x_samples)
    fig.suptitle(
        f"HNN — {'Step ' + str(step) if step else 'Final'} (MC n={n_mc})", fontsize=12
    )
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=150)
    plt.close(fig)


def plot_metrics(metrics: list[dict[str, Any]], baseline: float, out_path: Path, switch_step: int | None = None) -> None:
    import matplotlib.pyplot as plt

    steps     = [m["step"] for m in metrics]
    train     = [m["train_loss"] for m in metrics]
    test      = [m["test_loss"] for m in metrics]
    param_std = [m["param_std"] for m in metrics]
    pred_std  = [m["pred_std"] for m in metrics]
    eff_n     = [m.get("eff_n") for m in metrics]
    has_eff_n = any(v is not None for v in eff_n)

    n_rows = 4 if has_eff_n else 3
    fig, axes = plt.subplots(n_rows, 1, figsize=(12, 4 * n_rows))

    def _vline(ax: Any) -> None:
        if switch_step is not None:
            ax.axvline(switch_step, color="tab:green", ls=":", lw=1.2, label="softmax switch")

    axes[0].plot(steps, train, label="train MSE", lw=1.5, color="tab:blue")
    axes[0].plot(steps, test,  label="test MSE",  lw=1.5, color="tab:orange")
    axes[0].axhline(baseline, color="gray", ls="--", lw=1, label=f"baseline ({baseline:.3f})")
    _vline(axes[0])
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("MSE (log)")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3, which="both")
    axes[0].set_title("HNN — loss")

    axes[1].plot(steps, param_std, lw=1.5, color="tab:purple")
    _vline(axes[1])
    axes[1].set_ylabel("param std across x")
    axes[1].set_title("Target network parameter diversity")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(steps, pred_std, lw=1.5, color="tab:red")
    _vline(axes[2])
    axes[2].set_ylabel("prediction std across x")
    axes[2].set_xlabel("step" if not has_eff_n else "")
    axes[2].set_title("Target network functional diversity")
    axes[2].grid(True, alpha=0.3)

    if has_eff_n:
        eff_n_vals = [v for v in eff_n if v is not None]
        eff_n_steps = [s for s, v in zip(steps, eff_n) if v is not None]
        axes[3].plot(eff_n_steps, eff_n_vals, lw=1.5, color="tab:brown")
        _vline(axes[3])
        axes[3].set_ylabel("effective-N")
        axes[3].set_xlabel("step")
        axes[3].set_title("Soft-aggregate selection breadth (1=hard, n_mc=uniform)")
        axes[3].grid(True, alpha=0.3)

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
    switch_step  = int(cfg.softmax_switch_frac * total_steps)
    warmup_steps = int(cfg.warmup_frac * total_steps)

    base_schedule = optax.warmup_cosine_decay_schedule(
        init_value=cfg.lr * cfg.lr_end_frac,
        peak_value=cfg.lr,
        warmup_steps=warmup_steps,
        decay_steps=total_steps,
        end_value=cfg.lr * cfg.lr_end_frac,
    )

    if cfg.softmax_lr_floor > 0.0:
        floor = cfg.softmax_lr_floor

        def schedule(step: jax.Array) -> jax.Array:
            lr = base_schedule(step)
            return jnp.where(step >= switch_step, jnp.maximum(lr, floor), lr)
    else:
        schedule = base_schedule

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
    print(f"Soft-min phase: steps 0–{switch_step}, soft-max phase: {switch_step}–{total_steps}")

    # ── JIT-compiled step ─────────────────────────────────────────────────────
    lambda_target_decay = cfg.lambda_target_decay

    @eqx.filter_jit
    def do_step(
        model: HyperNetwork,
        opt_state: optax.OptState,
        c_b: jax.Array,
        t_b: jax.Array,
        x_samples: jax.Array,
        alpha: jax.Array,
        temp_scale: jax.Array,
    ) -> tuple[HyperNetwork, optax.OptState, jax.Array]:
        loss, grads = eqx.filter_value_and_grad(
            lambda m: _task_loss(m, c_b, t_b, x_samples, lambda_target_decay, alpha, temp_scale)
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

        in_softmax   = (s + 1) > switch_step
        alpha_val    = 1.0 if in_softmax else -1.0
        scale_val    = cfg.softmax_temp_scale if in_softmax else cfg.softmin_temp_scale

        k_noise, subkey = jax.random.split(k_noise)
        x_samples = jax.random.uniform(
            subkey, (cfg.n_mc_train, cfg.n_stimulus), minval=-1.0, maxval=1.0
        )
        batch_idx = rng.choice(cfg.n_train, size=cfg.batch_size, replace=False)
        c_b = jnp.array(c_train_np[batch_idx])
        t_b = jnp.array(t_train_np[batch_idx])

        model, opt_state, loss = do_step(
            model, opt_state, c_b, t_b, x_samples,
            jnp.array(alpha_val), jnp.array(scale_val),
        )

        completed = s + 1
        if (completed % cfg.checkpoint_interval == 0) or (completed == cfg.n_steps):
            test_loss  = float(eval_loss(model, c_test,  t_test))
            train_loss = float(eval_loss(model, c_train, t_train))
            diversity  = target_network_diversity(model, c_b, eval_x_samples)
            elapsed    = time.perf_counter() - t0

            # Effective-N and effective temperature for the current phase
            per_x_losses_ckpt = jax.vmap(
                lambda x: jnp.mean((jax.vmap(model, in_axes=(0, None))(c_b, x) - t_b) ** 2)
            )(eval_x_samples)
            T_eff = float(_adaptive_temp(per_x_losses_ckpt, scale_val))
            weights_ckpt = _soft_weights(per_x_losses_ckpt, T_eff, alpha_val)
            eff_n = float(1.0 / jnp.sum(weights_ckpt ** 2))

            metrics.append({
                "step": completed,
                "train_loss": train_loss,
                "test_loss": test_loss,
                "phase": "softmax" if in_softmax else "softmin",
                "eff_n": eff_n,
                "eff_temp": T_eff,
                **diversity,
            })

            ckpt_dir.mkdir(parents=True, exist_ok=True)
            _save_checkpoint(ckpt_dir, completed, model, opt_state, rng)
            tag = _ckpt_tag(completed)
            _visualize_hnn(model, cfg, ckpt_dir / f"{tag}_pred.png", eval_x_samples, step=completed)

            (data_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
            plot_metrics(metrics, const_baseline, data_dir / "train_curve.png", switch_step=switch_step)

            pbar.set_postfix(
                phase="smax" if in_softmax else "smin",
                train=f"{train_loss:.4f}",
                test=f"{test_loss:.4f}",
                p_std=f"{diversity['param_std']:.4f}",
                eff_n=f"{eff_n:.1f}",
                T=f"{T_eff:.2e}",
                s=f"{elapsed:.1f}s",
            )

    (data_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    plot_metrics(metrics, const_baseline, data_dir / "train_curve.png", switch_step=switch_step)
    return model


def load_checkpoint(path: str | Path, cfg: HNNConfig) -> HyperNetwork:
    """Load a model checkpoint by explicit path."""
    template = _make_model_template(cfg)
    return eqx.tree_deserialise_leaves(str(path), template)


def main() -> None:
    cfg = HNNConfig()
    train(cfg)

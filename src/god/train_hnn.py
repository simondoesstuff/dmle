"""Hypernetwork (HNN) training for Mandelbrot magnitude prediction.

Oscillatory training schedule:
  - N_gaussian steps: update only GaussianEncoder (μ, log σ²)
    Loss = task MSE + lambda_kl * KL(N(μ, σ²) ‖ N(0, I))
  - 1 step: update stimulus_to_coord_params
    Loss = task MSE + lambda_target_decay * mean(||generated_params||²)

At each step a fresh noise vector x ~ Uniform[-1, 1]^n_stimulus is sampled
and passed through the encoder (reparameterisation trick).
See docs/hnn.md for the full architecture description.
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
from god.encoding import K_DEFAULT
from god.hnn import HyperNetwork, make_hypernetwork, n_trainable_params


@dataclass
class HNNConfig:
    data_dir: str = "data/mandel/hnn"

    # Encoding
    K: int = K_DEFAULT

    # Architecture
    n_stimulus: int = 32
    node_vec_dim: int = 16
    coord_net_hidden: int = 32
    target_hidden_dim: int = 32
    n_target_hidden_layers: int = 2
    temperature: float = 1.0

    # Data
    n_train: int = 10_000
    n_test: int = 2_000
    num_steps: int = 10

    # Training
    n_steps: int = 40_000
    batch_size: int = 64
    lr: float = 3e-4
    lr_gauss: float = 3e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    lambda_kl: float = 0.1       # KL(N(μ,σ²) ‖ N(0,I)) weight (Gaussian phase)
    n_gaussian: int = 3          # gaussian-only steps per cycle (then 1 rest step)
    n_mc_eval: int = 32          # MC samples for eval/visualise (E_x[f(c_enc, x)])
    n_mc_train: int = 8          # MC samples per training step (same batch for all)

    # Cosine annealing LR (both optimisers)
    warmup_frac: float = 0.3
    lr_end_frac: float = 0.01

    # Architecture init
    init_scale: float = 0.1

    # Stimulus FFN (between encoder and coord net weights)
    stim_ffn_hidden: int = 128
    stim_ffn_depth: int = 2

    # RNN target network (matches standalone RNN defaults)
    target_is_rnn: bool = True
    rnn_hidden_dim: int = 128
    rnn_depth: int = 2
    rnn_num_steps: int = 10

    # L2 regularisation on generated target net parameters (rest phase only)
    lambda_target_decay: float = 1e-3

    checkpoint_interval: int = 500
    seed: int = 0


def _task_loss(
    model: HyperNetwork,
    c_encs: jax.Array,
    targets: jax.Array,
    x_samples: jax.Array,  # (n_mc, n_stimulus) — same batch shared across all networks
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
    # vmap over x_samples; for each x average predictions over all c_encs
    per_x = jax.vmap(
        lambda x: jax.vmap(model, in_axes=(0, None))(c_encs, x)
    )(x_samples)  # (n_mc, n_points)
    return jnp.mean(per_x, axis=0)  # (n_points,)


def _visualize_hnn(
    model: HyperNetwork,
    cfg: HNNConfig,
    out_path: Path,
    eval_x_samples: jax.Array,
    step: int | None = None,
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
    pred_mags = _mc_predict(model, c_encs, eval_x_samples) * ESCAPE_RADIUS

    true_grid = np.array(true_mags).reshape(res, res)
    pred_grid = np.array(pred_mags).reshape(res, res)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    kw = dict(
        origin="lower",
        extent=[-2, 2, -2, 2],
        vmin=0,
        vmax=ESCAPE_RADIUS,
        cmap="inferno",
    )

    axes[0].imshow(true_grid, **kw)
    axes[0].set_title("True |z_T|")
    axes[1].imshow(pred_grid, **kw)
    axes[1].set_title(f"Predicted |z_T| (MC n={len(eval_x_samples)})")

    err = np.abs(true_grid - pred_grid)
    axes[2].imshow(err, origin="lower", extent=[-2, 2, -2, 2], cmap="hot")
    axes[2].set_title("Absolute error")

    for ax in axes:
        ax.axhline(0, color="white", lw=0.5, ls="--")
        ax.set_xlabel("Re(c)")
        ax.set_ylabel("Im(c)")

    title = f"Step {step}" if step is not None else "Final"
    fig.suptitle(f"HNN — {title}", fontsize=12)
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

    steps = [m["step"] for m in metrics]
    train = [m["train_loss"] for m in metrics]
    test = [m["test_loss"] for m in metrics]
    kl = [m["kl_loss"] for m in metrics]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    ax1.plot(steps, train, label="train MSE", lw=1.5, color="tab:blue")
    ax1.plot(steps, test, label="test MSE", lw=1.5, color="tab:orange")
    ax1.axhline(
        baseline, color="gray", ls="--", lw=1, label=f"const baseline ({baseline:.3f})"
    )
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("step (log)")
    ax1.set_ylabel("MSE (log)")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(True, alpha=0.3, which="both")
    ax1.set_title("HNN — training loss")

    ax2.plot(steps, kl, label="KL loss", lw=1.5, color="tab:purple")
    ax2.set_xlabel("step")
    ax2.set_ylabel("KL(N(μ,σ²) ‖ N(0,I))")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=150)
    plt.close(fig)


def train(cfg: HNNConfig) -> HyperNetwork:
    key = jax.random.PRNGKey(cfg.seed)
    k_model, k_train, k_test, k_noise, k_eval = jax.random.split(key, 5)

    model = make_hypernetwork(
        k_model,
        K=cfg.K,
        n_stimulus=cfg.n_stimulus,
        node_vec_dim=cfg.node_vec_dim,
        coord_net_hidden=cfg.coord_net_hidden,
        target_hidden_dim=cfg.target_hidden_dim,
        n_target_hidden_layers=cfg.n_target_hidden_layers,
        target_is_rnn=cfg.target_is_rnn,
        rnn_hidden_dim=cfg.rnn_hidden_dim,
        rnn_depth=cfg.rnn_depth,
        rnn_num_steps=cfg.rnn_num_steps,
        temperature=cfg.temperature,
        init_scale=cfg.init_scale,
        stim_ffn_hidden=cfg.stim_ffn_hidden,
        stim_ffn_depth=cfg.stim_ffn_depth,
    )

    data_dir = Path(cfg.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "config.json").write_text(json.dumps(dataclasses.asdict(cfg), indent=2))

    print("Generating datasets...")
    c_train, t_train = make_dataset(
        cfg.n_train, cfg.num_steps, k_train, train=True, K=cfg.K
    )
    c_test, t_test = make_dataset(
        cfg.n_test, cfg.num_steps, k_test, train=False, K=cfg.K
    )
    const_baseline = float(jnp.mean((t_train - jnp.mean(t_train)) ** 2))
    print(f"Constant-mean MSE baseline: {const_baseline:.4f}")
    print(f"HNN trainable params: {n_trainable_params(model):,}")

    # Fixed noise samples used for all eval calls — consistent metric, no train/eval mismatch
    eval_x_samples = jax.random.uniform(
        k_eval, (cfg.n_mc_eval, cfg.n_stimulus), minval=-1.0, maxval=1.0
    )

    total_steps = cfg.n_steps
    warmup_steps = int(cfg.warmup_frac * total_steps)

    def _cosine_schedule(peak_lr: float) -> optax.Schedule:
        return optax.warmup_cosine_decay_schedule(
            init_value=peak_lr * cfg.lr_end_frac,
            peak_value=peak_lr,
            warmup_steps=warmup_steps,
            decay_steps=total_steps,
            end_value=peak_lr * cfg.lr_end_frac,
        )

    # Gaussian phase: Adam (KL already regularises; no extra weight decay needed)
    opt_gauss = optax.chain(
        optax.clip_by_global_norm(cfg.grad_clip),
        optax.adam(learning_rate=_cosine_schedule(cfg.lr_gauss)),
    )

    # Rest phase: AdamW with weight decay on stim_ffn
    opt_rest = optax.chain(
        optax.clip_by_global_norm(cfg.grad_clip),
        optax.adamw(
            learning_rate=_cosine_schedule(cfg.lr), weight_decay=cfg.weight_decay
        ),
    )

    opt_state_gauss = opt_gauss.init(eqx.filter(model.gaussian_encoder, eqx.is_array))
    opt_state_rest = opt_rest.init(
        eqx.filter(model.stimulus_to_coord_params, eqx.is_array)
    )

    lambda_kl = cfg.lambda_kl
    lambda_target_decay = cfg.lambda_target_decay

    @eqx.filter_jit
    def step_gaussian(
        model: HyperNetwork,
        opt_state: optax.OptState,
        c_b: jax.Array,
        t_b: jax.Array,
        x_samples: jax.Array,  # (n_mc_train, n_stimulus)
    ) -> tuple[HyperNetwork, optax.OptState, jax.Array]:
        def loss_fn(gauss_enc):
            m = eqx.tree_at(lambda m: m.gaussian_encoder, model, gauss_enc)
            return _task_loss(m, c_b, t_b, x_samples) + lambda_kl * gauss_enc.kl_loss()

        gauss_enc = model.gaussian_encoder
        loss, grads = eqx.filter_value_and_grad(loss_fn)(gauss_enc)
        updates, new_state = opt_gauss.update(
            grads, opt_state, eqx.filter(gauss_enc, eqx.is_array)
        )
        new_gauss = eqx.apply_updates(gauss_enc, updates)
        new_model = eqx.tree_at(lambda m: m.gaussian_encoder, model, new_gauss)
        return new_model, new_state, loss

    @eqx.filter_jit
    def step_rest(
        model: HyperNetwork,
        opt_state: optax.OptState,
        c_b: jax.Array,
        t_b: jax.Array,
        x_samples: jax.Array,  # (n_mc_train, n_stimulus)
    ) -> tuple[HyperNetwork, optax.OptState, jax.Array]:
        def loss_fn(stim):
            m = eqx.tree_at(lambda m: m.stimulus_to_coord_params, model, stim)
            return _task_loss(m, c_b, t_b, x_samples, lambda_target_decay)

        stim = model.stimulus_to_coord_params
        loss, grads = eqx.filter_value_and_grad(loss_fn)(stim)
        updates, new_state = opt_rest.update(
            grads, opt_state, eqx.filter(stim, eqx.is_array)
        )
        new_stim = eqx.apply_updates(stim, updates)
        new_model = eqx.tree_at(lambda m: m.stimulus_to_coord_params, model, new_stim)
        return new_model, new_state, loss

    @eqx.filter_jit
    def eval_loss(
        model: HyperNetwork,
        c_encs: jax.Array,
        targets: jax.Array,
        x_samples: jax.Array,
    ) -> jax.Array:
        preds = _mc_predict(model, c_encs, x_samples)
        return jnp.mean((preds - targets) ** 2)

    rng = np.random.default_rng(cfg.seed)
    c_train_np = np.array(c_train)
    t_train_np = np.array(t_train)

    metrics: list[dict[str, Any]] = []
    ckpt_dir = data_dir / "checkpoints"
    cycle_len = cfg.n_gaussian + 1
    noise_key = k_noise

    pbar = trange(cfg.n_steps, desc="steps")
    for step in pbar:
        t0 = time.perf_counter()

        noise_key, subkey = jax.random.split(noise_key)
        x_samples = jax.random.uniform(
            subkey, (cfg.n_mc_train, cfg.n_stimulus), minval=-1.0, maxval=1.0
        )

        batch_idx = rng.choice(cfg.n_train, size=cfg.batch_size, replace=False)
        c_b = jnp.array(c_train_np[batch_idx])
        t_b = jnp.array(t_train_np[batch_idx])

        phase = step % cycle_len
        if phase < cfg.n_gaussian:
            model, opt_state_gauss, loss = step_gaussian(
                model, opt_state_gauss, c_b, t_b, x_samples
            )
        else:
            model, opt_state_rest, loss = step_rest(model, opt_state_rest, c_b, t_b, x_samples)

        completed = step + 1
        is_checkpoint = (completed % cfg.checkpoint_interval == 0) or (
            completed == cfg.n_steps
        )

        if is_checkpoint:
            test_loss = float(eval_loss(model, c_test, t_test, eval_x_samples))
            train_loss = float(eval_loss(model, c_train, t_train, eval_x_samples))
            kl = float(model.gaussian_encoder.kl_loss())
            elapsed = time.perf_counter() - t0

            record: dict[str, Any] = {
                "step": completed,
                "train_loss": train_loss,
                "test_loss": test_loss,
                "kl_loss": kl,
                "phase": "gauss" if phase < cfg.n_gaussian else "rest",
            }
            metrics.append(record)

            ckpt_dir.mkdir(parents=True, exist_ok=True)
            tag = f"step_{completed:05d}"
            eqx.tree_serialise_leaves(str(ckpt_dir / f"{tag}.eqx"), model)
            _visualize_hnn(model, cfg, ckpt_dir / f"{tag}_pred.png", eval_x_samples, step=completed)

            (data_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
            plot_metrics(metrics, const_baseline, data_dir / "train_curve.png")

            pbar.set_postfix(
                train=f"{train_loss:.4f}",
                test=f"{test_loss:.4f}",
                KL=f"{kl:.4f}",
                s=f"{elapsed:.1f}s",
            )

    (data_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    plot_metrics(metrics, const_baseline, data_dir / "train_curve.png")
    print(f"Metrics  → {data_dir / 'metrics.json'}")
    print(f"Curve    → {data_dir / 'train_curve.png'}")

    return model


def load_checkpoint(path: str | Path, cfg: HNNConfig) -> HyperNetwork:
    key = jax.random.PRNGKey(cfg.seed)
    template = make_hypernetwork(
        key,
        K=cfg.K,
        n_stimulus=cfg.n_stimulus,
        node_vec_dim=cfg.node_vec_dim,
        coord_net_hidden=cfg.coord_net_hidden,
        target_hidden_dim=cfg.target_hidden_dim,
        n_target_hidden_layers=cfg.n_target_hidden_layers,
        target_is_rnn=cfg.target_is_rnn,
        rnn_hidden_dim=cfg.rnn_hidden_dim,
        rnn_depth=cfg.rnn_depth,
        rnn_num_steps=cfg.rnn_num_steps,
        temperature=cfg.temperature,
        stim_ffn_hidden=cfg.stim_ffn_hidden,
        stim_ffn_depth=cfg.stim_ffn_depth,
    )
    return eqx.tree_deserialise_leaves(str(path), template)


def main() -> None:
    cfg = HNNConfig()
    train(cfg)

"""Train tanh FFNs to learn the single-step map z -> z^2 + c.

Sweeps hidden widths [1, 2, 3, 4, 6, 8] with 1 hidden layer to find the
minimum network that achieves high fit.

After the sweep, iterates the learned cells T times over a c-grid and
compares to the reference Mandelbrot orbit — closing the loop with the RNN
question (a well-fit step cell should reproduce multi-step dynamics).
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
import matplotlib.pyplot as plt
import numpy as np
import optax
from jaxtyping import PRNGKeyArray
from tqdm import trange

from god.datasets.mandelbrot import ESCAPE_RADIUS, mandelbrot_mag
from god.datasets.step import make_step_dataset
from god.models.ffn import TanhFFN


@dataclass
class TrainConfig:
    data_dir: str = "data/mandel/ffn_direct"
    widths: tuple[int, ...] = (1, 2, 3, 4, 6, 8)
    depth: int = 1
    n_train: int = 20_000
    n_test: int = 4_000
    batch_size: int = 512
    n_epochs: int = 2_000
    lr_peak: float = 3e-3
    warmup_frac: float = 0.05
    lr_end: float = 1e-5
    weight_decay: float = 1e-3
    grad_clip: float = 5.0
    # T-step iteration check after training
    iterate_steps: int = 50
    seed: int = 0


def _mse(pred: jax.Array, target: jax.Array) -> jax.Array:
    return jnp.mean((pred - target) ** 2)


def _r2(pred: jax.Array, target: jax.Array) -> float:
    ss_res = float(jnp.sum((pred - target) ** 2))
    ss_tot = float(jnp.sum((target - jnp.mean(target, axis=0)) ** 2))
    return 1.0 - ss_res / (ss_tot + 1e-12)


def _train_one(
    width: int,
    cfg: TrainConfig,
    dataset: Any,
    key: PRNGKeyArray,
) -> tuple[TanhFFN, list[dict[str, Any]]]:
    model = TanhFFN(4, width, cfg.depth, 2, key)

    steps_per_epoch = max(1, cfg.n_train // cfg.batch_size)
    total_steps = steps_per_epoch * cfg.n_epochs
    warmup_steps = int(cfg.warmup_frac * total_steps)

    schedule: optax.Schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=cfg.lr_peak,
        warmup_steps=warmup_steps,
        decay_steps=total_steps,
        end_value=cfg.lr_end,
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(cfg.grad_clip),
        optax.adamw(learning_rate=schedule, weight_decay=cfg.weight_decay),
    )
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    def step_fn(
        model: TanhFFN,
        opt_state: optax.OptState,
        x: jax.Array,
        y: jax.Array,
    ) -> tuple[TanhFFN, optax.OptState, jax.Array]:
        def loss_fn(m: TanhFFN) -> jax.Array:
            return _mse(jax.vmap(m)(x), y)
        loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
        updates, new_state = optimizer.update(grads, opt_state, eqx.filter(model, eqx.is_array))
        return eqx.apply_updates(model, updates), new_state, loss

    @eqx.filter_jit
    def eval_fn(model: TanhFFN, x: jax.Array, y: jax.Array) -> jax.Array:
        return _mse(jax.vmap(model)(x), y)

    x_train = np.array(dataset.train_inputs)
    y_train = np.array(dataset.train_targets)
    x_test = dataset.test_inputs
    y_test = dataset.test_targets
    rng = np.random.default_rng(cfg.seed)

    metrics: list[dict[str, Any]] = []
    pbar = trange(cfg.n_epochs, desc=f"width={width}", leave=False)
    for epoch in pbar:
        idx = rng.permutation(cfg.n_train)
        epoch_loss = 0.0
        for i in range(steps_per_epoch):
            b = idx[i * cfg.batch_size : (i + 1) * cfg.batch_size]
            xb = jnp.array(x_train[b])
            yb = jnp.array(y_train[b])
            model, opt_state, loss = step_fn(model, opt_state, xb, yb)
            epoch_loss += float(loss)
        train_mse = epoch_loss / steps_per_epoch
        test_mse = float(eval_fn(model, x_test, y_test))
        metrics.append({"epoch": epoch + 1, "train_mse": train_mse, "test_mse": test_mse})
        pbar.set_postfix(train=f"{train_mse:.4f}", test=f"{test_mse:.4f}")

    return model, metrics


def _make_iterate_fn(model: TanhFFN, T: int, escape_radius: float = ESCAPE_RADIUS):
    """Return a vmappable fn (c_re, c_im) -> |z_T| that iterates model T times.

    T must be closed over so jax.lax.scan sees a concrete int, not a traced value.
    """
    def iterate(c_re: jax.Array, c_im: jax.Array) -> jax.Array:
        def body(z: jax.Array, _: None) -> tuple[jax.Array, None]:
            x = jnp.concatenate([z, jnp.stack([c_re, c_im])])
            z_new = model(x)
            mag = jnp.sqrt(z_new[0] ** 2 + z_new[1] ** 2)
            scale = jnp.where(mag > escape_radius, escape_radius / (mag + 1e-12), 1.0)
            return z_new * scale, None

        z_T, _ = jax.lax.scan(body, jnp.zeros(2), None, length=T)
        return jnp.sqrt(z_T[0] ** 2 + z_T[1] ** 2)

    return iterate


def _plot_sweep(
    sweep_results: list[dict[str, Any]],
    out_path: Path,
) -> None:
    widths = [r["width"] for r in sweep_results]
    test_mses = [r["test_mse"] for r in sweep_results]
    r2s = [r["r2"] for r in sweep_results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.semilogy(widths, test_mses, "o-", lw=1.5)
    ax1.set_xlabel("hidden width")
    ax1.set_ylabel("test MSE (log)")
    ax1.set_title("Test MSE vs. width")
    ax1.set_xticks(widths)
    ax1.grid(True, alpha=0.3)

    ax2.plot(widths, r2s, "s-", lw=1.5, color="tab:orange")
    ax2.axhline(1.0, color="gray", ls="--", lw=0.8)
    ax2.set_xlabel("hidden width")
    ax2.set_ylabel("R²")
    ax2.set_title("R² vs. width (higher = better)")
    ax2.set_xticks(widths)
    ax2.set_ylim(bottom=min(0.0, min(r2s) - 0.05))
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150)
    plt.close(fig)


def _plot_iteration_check(
    model: TanhFFN,
    width: int,
    T: int,
    out_path: Path,
) -> None:
    res = 200
    re_vals = np.linspace(-2.0, 0.6, res)
    im_vals = np.linspace(-1.0, 1.0, res)
    RE, IM = np.meshgrid(re_vals, im_vals)
    c_r = jnp.array(RE.ravel())
    c_i = jnp.array(IM.ravel())

    true_mags = jax.vmap(mandelbrot_mag, in_axes=(0, 0, None, None))(
        c_r, c_i, T, ESCAPE_RADIUS
    )
    iterate_fn = jax.jit(jax.vmap(_make_iterate_fn(model, T)))
    pred_mags = iterate_fn(c_r, c_i)

    true_grid = np.array(true_mags).reshape(res, res)
    pred_grid = np.array(pred_mags).reshape(res, res)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    kw = dict(origin="lower", extent=[-2, 0.6, -1, 1], vmin=0, vmax=ESCAPE_RADIUS, cmap="inferno")
    axes[0].imshow(true_grid, **kw)
    axes[0].set_title(f"True |z_{T}|")
    axes[1].imshow(pred_grid, **kw)
    axes[1].set_title(f"Learned cell iterated {T}×  (width={width})")
    err = np.abs(true_grid - pred_grid)
    axes[2].imshow(err, origin="lower", extent=[-2, 0.6, -1, 1], cmap="hot")
    axes[2].set_title("Absolute error")
    for ax in axes:
        ax.set_xlabel("Re(c)")
        ax.set_ylabel("Im(c)")
    plt.suptitle(f"TanhFFN width={width}, depth=1, iterated {T} steps", fontsize=11)
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150)
    plt.close(fig)


def train(cfg: TrainConfig) -> None:
    key = jax.random.PRNGKey(cfg.seed)
    k_data, *k_models = jax.random.split(key, len(cfg.widths) + 1)

    data_dir = Path(cfg.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "config.json").write_text(json.dumps(dataclasses.asdict(cfg), indent=2))

    print("Generating step dataset...")
    dataset = make_step_dataset(cfg.n_train, cfg.n_test, k_data)
    x_test = dataset.test_inputs
    y_test = dataset.test_targets

    sweep_results: list[dict[str, Any]] = []
    trained_models: dict[int, TanhFFN] = {}

    for width, k_model in zip(cfg.widths, k_models, strict=True):
        n_params = sum(
            x.size for x in jax.tree_util.tree_leaves(
                eqx.filter(TanhFFN(4, width, cfg.depth, 2, k_model), eqx.is_array)
            )
        )
        print(f"\n--- width={width}  params={n_params} ---")
        t0 = time.perf_counter()
        model, metrics = _train_one(width, cfg, dataset, k_model)
        elapsed = time.perf_counter() - t0

        pred = jax.vmap(model)(x_test)
        final_mse = float(jnp.mean((pred - y_test) ** 2))
        r2 = _r2(pred, y_test)

        result: dict[str, Any] = {
            "width": width,
            "n_params": n_params,
            "test_mse": final_mse,
            "r2": r2,
            "elapsed_s": round(elapsed, 1),
        }
        sweep_results.append(result)
        trained_models[width] = model
        (data_dir / f"metrics_w{width}.json").write_text(json.dumps(metrics, indent=2))
        print(f"  test MSE={final_mse:.6f}  R²={r2:.4f}  ({elapsed:.0f}s)")

    (data_dir / "sweep.json").write_text(json.dumps(sweep_results, indent=2))
    _plot_sweep(sweep_results, data_dir / "width_sweep.png")
    print(f"\nSweep → {data_dir / 'width_sweep.png'}")

    for r in sweep_results:
        print(f"  width={r['width']:2d}  params={r['n_params']:4d}  "
              f"MSE={r['test_mse']:.6f}  R²={r['r2']:.4f}")

    # Iterate the minimum-viable cell (first width that achieves R²>0.999)
    viable: list[dict[str, Any]] = [r for r in sweep_results if r["r2"] > 0.999]
    if not viable:
        viable = [max(sweep_results, key=lambda r: r["r2"])]
    min_viable = min(viable, key=lambda r: r["n_params"])
    min_model = trained_models[min_viable["width"]]

    print(f"\nMinimum viable width={min_viable['width']} — plotting iteration check...")
    _plot_iteration_check(min_model, min_viable["width"], cfg.iterate_steps, data_dir / "iteration_check.png")
    print(f"Iteration check → {data_dir / 'iteration_check.png'}")

    # Also plot the widest cell for comparison
    max_width = max(cfg.widths)
    if max_width != min_viable["width"]:
        max_model = trained_models[max_width]
        _plot_iteration_check(max_model, max_width, cfg.iterate_steps, data_dir / f"iteration_check_w{max_width}.png")
        print(f"Width={max_width} iteration check → {data_dir / f'iteration_check_w{max_width}.png'}")


def main() -> None:
    cfg = TrainConfig()
    train(cfg)


if __name__ == "__main__":
    main()

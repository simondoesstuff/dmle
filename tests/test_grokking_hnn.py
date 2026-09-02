"""Tests for ModularHNN and HNN grokking training."""

import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import pytest

from god.datasets.modular import ModularConfig, Op, input_dim, make_modular_dataset
from god.models.modular_hnn import ModularHNN, make_modular_hypernetwork, n_trainable_params
from god.training.grokking_hnn import GrokkingHNNConfig, _loss_and_metrics, _setup, train

KEY = jax.random.PRNGKey(42)
P = 7  # small modulus for tests


def _make_small_model(key=KEY, in_dim=2 * P + 1, out_dim=P):
    return make_modular_hypernetwork(
        key,
        in_dim=in_dim,
        out_dim=out_dim,
        target_hidden=8,
        target_depth=1,
        n_stimulus=4,
        stim_ffn_hidden=4,
        stim_ffn_depth=1,
    )


def _x(n: int = 4, n_stimulus: int = 4) -> jax.Array:
    return jax.random.uniform(KEY, (n, n_stimulus), minval=-1.0, maxval=1.0)


# ── model construction ────────────────────────────────────────────────────────

def test_make_modular_hypernetwork():
    model = _make_small_model()
    assert isinstance(model, ModularHNN)
    assert model.n_target_layers == 2


def test_param_layout():
    model = _make_small_model()
    # target: (2*7+1=15) → 8 → 7
    # W0: (8, 15), b0: (8,), W1: (7, 8), b1: (7,)
    in_dim = 2 * P + 1
    expected = [
        ("W0", (8, in_dim)),
        ("b0", (8,)),
        ("W1", (P, 8)),
        ("b1", (P,)),
    ]
    assert model.param_layout == expected


def test_n_trainable_params_positive():
    model = _make_small_model()
    assert n_trainable_params(model) > 0


def test_target_params_shape():
    model = _make_small_model()
    x = _x(1)[0]
    flat = model.target_params(x)
    n_expected = sum(
        int(jnp.prod(jnp.array(s))) for _, s in model.param_layout
    )
    assert flat.shape == (n_expected,)


# ── forward pass ──────────────────────────────────────────────────────────────

def test_output_shape():
    model = _make_small_model()
    inputs = jnp.zeros(2 * P + 1)
    x = _x(1)[0]
    out = model(inputs, x)
    assert out.shape == (P,)


def test_output_is_finite():
    model = _make_small_model()
    inputs = jax.random.normal(KEY, (2 * P + 1,))
    x = _x(1)[0]
    assert jnp.all(jnp.isfinite(model(inputs, x)))


def test_batched_forward():
    model = _make_small_model()
    B = 16
    inputs = jax.random.normal(KEY, (B, 2 * P + 1))
    x = _x(1)[0]
    logits = jax.vmap(model, in_axes=(0, None))(inputs, x)
    assert logits.shape == (B, P)


def test_different_x_produce_different_logits():
    model = _make_small_model()
    inputs = jax.random.normal(KEY, (2 * P + 1,))
    xs = _x(8)
    outs = jax.vmap(lambda x: model(inputs, x))(xs)
    assert float(jnp.std(outs)) > 1e-6


# ── loss and metrics ──────────────────────────────────────────────────────────

def test_loss_is_finite():
    cfg = ModularConfig(modulus=P, ops=(Op.ADD,), train_fraction=0.5)
    ds = make_modular_dataset(cfg, KEY)
    model = _make_small_model()
    x_samples = _x(4)
    loss, acc, ce = _loss_and_metrics(model, ds.train_inputs, ds.train_targets, x_samples, 0.0)
    assert jnp.isfinite(loss)
    assert 0.0 <= float(acc) <= 1.0


def test_loss_decreases_with_training():
    cfg = ModularConfig(modulus=P, ops=(Op.ADD,), train_fraction=0.5)
    ds = make_modular_dataset(cfg, KEY)
    model = _make_small_model()
    x_samples = _x(4)

    loss_before = float(
        _loss_and_metrics(model, ds.train_inputs, ds.train_targets, x_samples, 0.0)[0]
    )

    opt = optax.adamw(learning_rate=1e-3, weight_decay=0.01)
    opt_state = opt.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    def step(m, s, xs):
        loss, g = eqx.filter_value_and_grad(
            lambda m: _loss_and_metrics(m, ds.train_inputs, ds.train_targets, xs, 0.0)[0]
        )(m)
        updates, s2 = opt.update(g, s, eqx.filter(m, eqx.is_array))
        return eqx.apply_updates(m, updates), s2, loss

    for _ in range(100):
        model, opt_state, _ = step(model, opt_state, x_samples)

    loss_after = float(
        _loss_and_metrics(model, ds.train_inputs, ds.train_targets, x_samples, 0.0)[0]
    )
    assert loss_after < loss_before


# ── gradient flow ─────────────────────────────────────────────────────────────

def test_gradients_reach_stimulus_ffn():
    cfg = ModularConfig(modulus=P, ops=(Op.ADD,), train_fraction=0.5)
    ds = make_modular_dataset(cfg, KEY)
    model = _make_small_model()
    x_samples = _x(4)
    w_before = model.stimulus_ffn.layers[0].weight

    grads = eqx.filter_grad(
        lambda m: _loss_and_metrics(m, ds.train_inputs, ds.train_targets, x_samples, 0.0)[0]
    )(model)
    assert jnp.any(grads.stimulus_ffn.layers[0].weight != 0.0)


# ── smoke: train() completes and saves outputs ────────────────────────────────

def test_train_smoke(tmp_path):
    cfg = GrokkingHNNConfig(
        data_dir=str(tmp_path / "grokking_hnn"),
        modulus=11,
        target_hidden=8,
        target_depth=1,
        n_stimulus=4,
        stim_ffn_hidden=4,
        stim_ffn_depth=1,
        train_fraction=0.5,
        n_epochs=20,
        log_interval=10,
        n_mc=2,
        hexplot_interval=0,
        seed=0,
    )
    model = train(cfg)
    assert isinstance(model, ModularHNN)
    assert (tmp_path / "grokking_hnn" / "metrics.json").exists()
    assert (tmp_path / "grokking_hnn" / "grokking_hnn_curve.png").exists()
    assert (tmp_path / "grokking_hnn" / "model.eqx").exists()

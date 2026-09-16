"""Tests for ModularHNN and HNN grokking training."""

import json

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


# ── ablation knobs: output_bias / shrink_target / probe metrics ───────────────

def _small_ablation_cfg(tmp_path, **overrides) -> GrokkingHNNConfig:
    defaults = dict(
        data_dir=str(tmp_path / "grokking_hnn"),
        modulus=11,
        target_hidden=8,
        target_depth=1,
        n_stimulus=4,
        stim_ffn_hidden=4,
        stim_ffn_depth=1,
        train_fraction=0.5,
        lambda_complexity=0.0,
        n_epochs=20,
        log_interval=10,
        n_mc=2,
        hexplot_interval=0,
        seed=0,
    )
    defaults.update(overrides)
    return GrokkingHNNConfig(**defaults)


def test_output_bias_false_has_no_bias():
    model = make_modular_hypernetwork(
        KEY, in_dim=2 * P + 1, out_dim=P, target_hidden=8, target_depth=1,
        n_stimulus=4, stim_ffn_hidden=4, stim_ffn_depth=1, output_bias=False,
    )
    assert model.stimulus_ffn.layers[-1].bias is None


def test_output_bias_false_trains(tmp_path):
    cfg = _small_ablation_cfg(tmp_path, output_bias=False)
    model = train(cfg)
    assert isinstance(model, ModularHNN)
    assert model.stimulus_ffn.layers[-1].bias is None


def test_shrink_target_requires_compatible_output_bias():
    with pytest.raises(ValueError):
        GrokkingHNNConfig(output_bias=False, shrink_target="bias", shrink_wd=1.0)
    with pytest.raises(ValueError):
        GrokkingHNNConfig(shrink_target="bias", shrink_wd=0.0)  # wd=0 is a no-op
    with pytest.raises(ValueError):
        GrokkingHNNConfig(shrink_target="not-a-target")


def test_shrink_bias_reduces_bias_norm(tmp_path):
    """Bias shrinkage should pull ‖b_last‖ below the no-shrinkage baseline."""
    base = _small_ablation_cfg(tmp_path / "off", shrink_target="none", n_epochs=50, log_interval=25)
    shrunk = _small_ablation_cfg(tmp_path / "on", shrink_target="bias", shrink_wd=50.0, n_epochs=50, log_interval=25)

    model_off = train(base)
    model_on = train(shrunk)

    bias_off = model_off.stimulus_ffn.layers[-1].bias
    bias_on = model_on.stimulus_ffn.layers[-1].bias
    assert bias_off is not None and bias_on is not None
    assert float(jnp.linalg.norm(bias_on)) < float(jnp.linalg.norm(bias_off))


def test_shrink_weight_reduces_weight_norm(tmp_path):
    """Weight shrinkage (the output_bias=False carrier) should pull ‖W_last‖ below baseline."""
    base = _small_ablation_cfg(
        tmp_path / "off", output_bias=False, shrink_target="none", n_epochs=50, log_interval=25,
    )
    shrunk = _small_ablation_cfg(
        tmp_path / "on", output_bias=False, shrink_target="weight", shrink_wd=50.0,
        n_epochs=50, log_interval=25,
    )

    model_off = train(base)
    model_on = train(shrunk)

    assert float(jnp.linalg.norm(model_on.stimulus_ffn.layers[-1].weight)) < float(
        jnp.linalg.norm(model_off.stimulus_ffn.layers[-1].weight)
    )


def test_probe_metrics_logged(tmp_path):
    cfg = _small_ablation_cfg(tmp_path, probe_n=8)
    train(cfg)
    metrics = json.loads((tmp_path / "grokking_hnn" / "metrics.json").read_text())
    assert "probe_param_norm" in metrics[-1]
    assert "probe_bias_norm" in metrics[-1]
    assert "probe_residual_norm" in metrics[-1]
    assert "probe_param_std" in metrics[-1]


def test_probe_metrics_identical_across_fixed_x(tmp_path):
    """probe_* metrics use a probe set independent of fixed_x/seed, so param_std
    there should be nonzero even when fixed_x=True (unlike the fixed_x-biased
    `param_std` field, which collapses to 0 by construction)."""
    cfg = _small_ablation_cfg(tmp_path, fixed_x=True, n_mc=1, probe_n=8)
    train(cfg)
    metrics = json.loads((tmp_path / "grokking_hnn" / "metrics.json").read_text())
    assert metrics[-1]["param_std"] == 0.0
    assert metrics[-1]["probe_param_std"] > 0.0


def test_checkpoint_round_trip_with_output_bias_false(tmp_path):
    cfg = _small_ablation_cfg(
        tmp_path, output_bias=False, n_epochs=20, checkpoint_interval=10,
    )
    train(cfg)
    data_dir = tmp_path / "grokking_hnn"
    ckpts = sorted(data_dir.glob("checkpoint_*.eqx"))
    # epoch 1 (first-iter checkpoint), epoch 10, epoch 20
    assert len(ckpts) == 3

    saved_cfg = json.loads((data_dir / "config.json").read_text())
    assert saved_cfg["output_bias"] is False

    _, template = _setup(cfg)
    restored = eqx.tree_deserialise_leaves(str(ckpts[-1]), template)
    assert restored.stimulus_ffn.layers[-1].bias is None

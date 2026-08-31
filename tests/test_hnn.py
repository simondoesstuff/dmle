"""Tests for the simplified HyperNetwork (direct parameter generation)."""

import jax
import jax.numpy as jnp
import equinox as eqx
import optax

from god.datasets.mandelbrot import enc_dim, encode, make_mandelbrot_dataset, K_DEFAULT
from god.models.hnn import (
    HyperNetwork,
    _unflatten_target_params,
    make_hypernetwork,
    n_trainable_params,
    target_network_diversity,
)

KEY = jax.random.PRNGKey(0)
D = enc_dim(K_DEFAULT)  # 18
N_STIMULUS = 16


def _x_samples(n: int = 8, key=KEY) -> jax.Array:
    return jax.random.uniform(key, (n, N_STIMULUS), minval=-1.0, maxval=1.0)


# ── model construction ────────────────────────────────────────────────────────


def test_make_hypernetwork_default():
    model = make_hypernetwork(KEY)
    assert isinstance(model, HyperNetwork)
    assert model.rnn_num_steps == 10
    assert model.n_cell_layers == 2  # rnn_depth=1 → 1 hidden + 1 linear


def test_param_layout_matches_target_dims():
    model = make_hypernetwork(KEY, rnn_hidden_dim=16, rnn_depth=1)
    # Cell dims: [2*18=36, 16, 18]
    # W0: (16, 36), b0: (16,), W1: (18, 16), b1: (18,)
    expected = [("W0", (16, 36)), ("b0", (16,)), ("W1", (18, 16)), ("b1", (18,))]
    assert model.param_layout == expected


def test_stimulus_ffn_output_matches_n_target_params():
    model = make_hypernetwork(KEY, rnn_hidden_dim=16, rnn_depth=1)
    n_params = sum(
        jax.numpy.prod(jax.numpy.array(shape)) for _, shape in model.param_layout
    )
    x = _x_samples(1)[0]
    flat = model.target_params(x)
    assert flat.shape == (int(n_params),)


# ── HyperNetwork forward ─────────────────────────────────────────────────────


def test_hypernetwork_output_shape():
    model = make_hypernetwork(KEY)
    c_enc = encode(jnp.array(0.3), jnp.array(-0.4))
    x = _x_samples(1)[0]
    out = model(c_enc, x)
    assert out.shape == ()


def test_hypernetwork_output_is_non_negative():
    """Output is sqrt(...) so must be ≥ 0."""
    model = make_hypernetwork(KEY)
    c_encs = jax.random.normal(KEY, (32, D))
    x = _x_samples(1)[0]
    outs = jax.vmap(model, in_axes=(0, None))(c_encs, x)
    assert jnp.all(outs >= 0.0)


def test_hypernetwork_output_is_finite():
    model = make_hypernetwork(KEY)
    c_encs = jax.random.normal(KEY, (32, D))
    x = _x_samples(1)[0]
    outs = jax.vmap(model, in_axes=(0, None))(c_encs, x)
    assert jnp.all(jnp.isfinite(outs))


def test_hypernetwork_batched_via_vmap():
    model = make_hypernetwork(KEY)
    B = 8
    c_encs = jax.random.normal(KEY, (B, D))
    x = _x_samples(1)[0]
    outs = jax.vmap(model, in_axes=(0, None))(c_encs, x)
    assert outs.shape == (B,)


def test_different_x_produce_different_outputs():
    """Different noise inputs must produce different predictions at init."""
    model = make_hypernetwork(KEY)
    c_enc = encode(jnp.array(0.3), jnp.array(-0.4))
    xs = _x_samples(16)
    preds = jax.vmap(lambda x: model(c_enc, x))(xs)
    assert float(jnp.std(preds)) > 1e-6, "all x produce identical output — no stimulus sensitivity"


# ── gradient flow ─────────────────────────────────────────────────────────────


def test_gradients_reach_stimulus_ffn():
    model = make_hypernetwork(KEY)
    c_enc = encode(jnp.array(0.3), jnp.array(-0.4))
    x = _x_samples(1)[0]
    target = jnp.array(0.5)

    grads = eqx.filter_grad(lambda m: (m(c_enc, x) - target) ** 2)(model)
    assert jnp.any(grads.stimulus_ffn.layers[0].weight != 0.0)


# ── diversity metric ──────────────────────────────────────────────────────────


def test_diversity_metrics_at_init():
    model = make_hypernetwork(KEY)
    c_encs = jax.random.normal(KEY, (16, D))
    xs = _x_samples(32)
    div = target_network_diversity(model, c_encs, xs)

    assert div["param_std"] > 0.0, "no parameter diversity at init"
    assert div["pred_std"] > 0.0, "no functional diversity at init"
    assert div["pred_range"] > 0.0


# ── n_trainable_params ────────────────────────────────────────────────────────


def test_n_trainable_params():
    model = make_hypernetwork(KEY)
    n = n_trainable_params(model)
    assert n > 0
    # All params are trainable (no frozen embeddings)
    total = sum(
        x.size for x in jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array))
    )
    assert n == total


# ── integration ───────────────────────────────────────────────────────────────


def test_loss_is_finite_at_init():
    from god.training.hnn import _task_loss

    model = make_hypernetwork(KEY)
    k1, k2 = jax.random.split(KEY)
    ds = make_mandelbrot_dataset(64, 8, 5, k1)
    x_samples = _x_samples(8, k2)
    loss = _task_loss(model, ds.train_inputs, ds.train_targets, x_samples)
    assert jnp.isfinite(loss)


def test_step_updates_stimulus_ffn():
    """A training step must change stimulus_ffn weights."""
    from god.training.hnn import _task_loss, HNNConfig

    cfg = HNNConfig(n_train=32, num_steps=5, seed=1)
    key = jax.random.PRNGKey(cfg.seed)
    k_model, k_data, k_noise = jax.random.split(key, 3)

    model = make_hypernetwork(k_model)
    ds = make_mandelbrot_dataset(cfg.n_train, 8, cfg.num_steps, k_data)
    c_train, t_train = ds.train_inputs, ds.train_targets
    w_before = model.stimulus_ffn.layers[0].weight.copy()

    opt = optax.adamw(learning_rate=1e-3)
    opt_state = opt.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    def do_step(model, opt_state, c_b, t_b, xs):
        loss, grads = eqx.filter_value_and_grad(
            lambda m: _task_loss(m, c_b, t_b, xs)
        )(model)
        updates, new_state = opt.update(grads, opt_state, eqx.filter(model, eqx.is_array))
        return eqx.apply_updates(model, updates), new_state

    xs = _x_samples(cfg.n_mc_train, k_noise)
    model, _ = do_step(model, opt_state, c_train[:8], t_train[:8], xs)
    assert not jnp.all(model.stimulus_ffn.layers[0].weight == w_before)

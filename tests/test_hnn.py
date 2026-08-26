"""Tests for the HyperNetwork architecture."""

import jax
import jax.numpy as jnp
import equinox as eqx

from god.encoding import enc_dim, encode, K_DEFAULT
from god.hnn import (
    HyperNetwork,
    _all_param_values,
    _build_param_embeddings,
    _coord_net_forward,
    _stimulus_to_coord_weights,
    _unflatten_target_params,
    make_hypernetwork,
    n_trainable_params,
    target_network_diversity,
)

KEY = jax.random.PRNGKey(0)
D = enc_dim(K_DEFAULT)  # 18
N_STIMULUS = 32


def _x_samples(n: int = 8, key=KEY) -> jax.Array:
    return jax.random.uniform(key, (n, N_STIMULUS), minval=-1.0, maxval=1.0)


# ── param embeddings ─────────────────────────────────────────────────────────


def test_param_embeddings_shape():
    model = make_hypernetwork(KEY)
    # Default: RNN target, enc_dim=18, rnn_hidden_dim=128, rnn_depth=2 (headless)
    # Cell dims: [36, 128, 18]
    cell_dims = [36, 128, 18]
    n_params = sum(
        (out_sz * in_sz + out_sz)
        for in_sz, out_sz in zip(cell_dims, cell_dims[1:])
    )
    embed_dim = 34  # 2 × (node_vec_dim=16 + 1)
    assert model.param_embeddings.shape == (n_params, embed_dim)


def test_param_embeddings_are_frozen():
    """stop_gradient must zero out the grad through param_embeddings."""
    model = make_hypernetwork(KEY)
    c_enc = encode(jnp.array(0.3), jnp.array(-0.4))
    x = _x_samples(1)[0]

    grads = eqx.filter_grad(lambda m: m(c_enc, x))(model)
    assert jnp.all(grads.param_embeddings == 0.0)


# ── HyperNetwork forward ─────────────────────────────────────────────────────


def test_hypernetwork_output_shape():
    model = make_hypernetwork(KEY)
    c_enc = encode(jnp.array(0.3), jnp.array(-0.4))
    x = _x_samples(1)[0]
    out = model(c_enc, x)
    assert out.shape == ()


def test_hypernetwork_batched_via_vmap():
    model = make_hypernetwork(KEY)
    B = 8
    c_encs = jax.random.normal(KEY, (B, D))
    x = _x_samples(1)[0]
    outs = jax.vmap(model, in_axes=(0, None))(c_encs, x)
    assert outs.shape == (B,)


def test_hypernetwork_output_is_finite():
    model = make_hypernetwork(KEY)
    c_encs = jax.random.normal(KEY, (32, D))
    x = _x_samples(1)[0]
    outs = jax.vmap(model, in_axes=(0, None))(c_encs, x)
    assert jnp.all(jnp.isfinite(outs))


def test_different_x_produce_different_outputs():
    """Different noise inputs must produce different predictions at init."""
    model = make_hypernetwork(KEY)
    c_enc = encode(jnp.array(0.3), jnp.array(-0.4))
    xs = _x_samples(16)
    preds = jax.vmap(lambda x: model(c_enc, x))(xs)
    assert float(jnp.std(preds)) > 1e-6, "all x produce identical output — no stimulus sensitivity"


# ── gradient flow ─────────────────────────────────────────────────────────────


def test_gradients_reach_stim_ffn():
    """Full-model grad must reach stimulus_to_coord_params."""
    model = make_hypernetwork(KEY)
    c_enc = encode(jnp.array(0.3), jnp.array(-0.4))
    x = _x_samples(1)[0]
    target = jnp.array(0.5)

    grads = eqx.filter_grad(lambda m: (m(c_enc, x) - target) ** 2)(model)
    assert jnp.any(grads.stimulus_to_coord_params.layers[0].weight != 0.0)


# ── diversity metric ──────────────────────────────────────────────────────────


def test_diversity_metrics_at_init():
    """At init the model should show non-zero parameter and functional diversity."""
    model = make_hypernetwork(KEY)
    c_encs = jax.random.normal(KEY, (16, D))
    xs = _x_samples(32)
    div = target_network_diversity(model, c_encs, xs)

    assert div["param_std"] > 0.0, "no parameter diversity at init"
    assert div["pred_std"] > 0.0, "no functional diversity at init"
    assert div["pred_range"] > 0.0


# ── integration ───────────────────────────────────────────────────────────────


def test_loss_is_finite_at_init():
    from god.data import make_dataset
    from god.train_hnn import _task_loss

    model = make_hypernetwork(KEY)
    k1, k2 = jax.random.split(KEY)
    c_encs, targets = make_dataset(64, 5, k1, train=True)
    x_samples = _x_samples(8, k2)
    loss = _task_loss(model, c_encs, targets, x_samples)
    assert jnp.isfinite(loss)


def test_n_trainable_params():
    model = make_hypernetwork(KEY)
    n = n_trainable_params(model)
    assert n > 0
    total = sum(
        x.size for x in jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array))
    )
    assert n < total  # frozen param_embeddings excluded


def test_step_updates_stim_ffn():
    """A training step must change stim_ffn weights."""
    import optax
    from god.train_hnn import _task_loss, HNNConfig
    from god.data import make_dataset

    cfg = HNNConfig(n_train=32, num_steps=5, seed=1)
    key = jax.random.PRNGKey(cfg.seed)
    k_model, k_train, k_noise = jax.random.split(key, 3)

    model = make_hypernetwork(k_model)
    c_train, t_train = make_dataset(cfg.n_train, cfg.num_steps, k_train, train=True)
    w_before = model.stimulus_to_coord_params.layers[0].weight.copy()

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
    assert not jnp.all(model.stimulus_to_coord_params.layers[0].weight == w_before)

"""Tests for the HyperNetwork architecture."""

import jax
import jax.numpy as jnp
import equinox as eqx
import pytest

from god.encoding import enc_dim, encode, K_DEFAULT
from god.hnn import (
    GaussianEncoder,
    HyperNetwork,
    _all_param_values,
    _build_param_embeddings,
    _coord_net_forward,
    _stimulus_to_coord_weights,
    _unflatten_target_params,
    make_hypernetwork,
    n_trainable_params,
)

KEY = jax.random.PRNGKey(0)
D = enc_dim(K_DEFAULT)  # 18
N_STIMULUS = 32


def _x_noise(n: int = N_STIMULUS, key=KEY) -> jax.Array:
    return jax.random.uniform(key, (n,), minval=-1.0, maxval=1.0)


# ── GaussianEncoder ──────────────────────────────────────────────────────────


def test_gaussian_encoder_output_shape():
    enc = GaussianEncoder(n_stimulus=N_STIMULUS)
    x = _x_noise()
    s = enc(x, 1.0)
    assert s.shape == (N_STIMULUS,)


def test_gaussian_encoder_mean_prediction():
    """x=0 → stimulus = μ regardless of temperature."""
    enc = GaussianEncoder(n_stimulus=N_STIMULUS)
    x_zeros = jnp.zeros(N_STIMULUS)
    s = enc(x_zeros, 1.0)
    assert jnp.allclose(s, enc.mu)


def test_gaussian_encoder_temperature_scales_noise():
    """Higher temperature → larger deviation from μ for the same x."""
    enc = GaussianEncoder(n_stimulus=4)
    # Give non-zero log_var and non-zero mu to make effect visible
    enc = eqx.tree_at(lambda e: e.log_var, enc, jnp.ones(4))
    x = jnp.ones(4) * 0.5
    s_cold = enc(x, 0.1)
    s_hot = enc(x, 5.0)
    # |s_hot - mu| > |s_cold - mu| everywhere
    assert jnp.all(jnp.abs(s_hot - enc.mu) > jnp.abs(s_cold - enc.mu))


def test_kl_loss_non_negative():
    """KL(N(μ, σ²) ‖ N(0, I)) ≥ 0 always."""
    enc = GaussianEncoder(n_stimulus=8)
    assert float(enc.kl_loss()) >= 0.0


def test_kl_loss_zero_at_prior():
    """KL = 0 iff μ=0, σ=1 (i.e. log_var=0). Init matches prior."""
    enc = GaussianEncoder(n_stimulus=8)
    assert jnp.allclose(enc.kl_loss(), jnp.zeros(()), atol=1e-6)


def test_kl_loss_increases_away_from_prior():
    enc = GaussianEncoder(n_stimulus=8)
    kl_before = float(enc.kl_loss())
    enc_shifted = eqx.tree_at(lambda e: e.mu, enc, enc.mu + 2.0)
    assert float(enc_shifted.kl_loss()) > kl_before


def test_kl_loss_gradient_pushes_to_prior():
    """Gradient of KL w.r.t. μ must have same sign as μ (pulls toward 0)."""
    enc = GaussianEncoder(n_stimulus=4)
    enc = eqx.tree_at(lambda e: e.mu, enc, jnp.array([1.0, -1.0, 2.0, -0.5]))
    grads = eqx.filter_grad(lambda e: e.kl_loss())(enc)
    # ∂KL/∂μ = μ — same sign as μ
    assert jnp.all(jnp.sign(grads.mu) == jnp.sign(enc.mu))


# ── param embeddings ─────────────────────────────────────────────────────────


def test_param_embeddings_shape():
    model = make_hypernetwork(KEY)
    # Default: RNN target, enc_dim=18, rnn_hidden_dim=128, rnn_depth=2 (headless)
    # Cell dims: [2*18=36, 128, 18]
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
    x = _x_noise()

    grads = eqx.filter_grad(lambda m: m(c_enc, x))(model)
    assert jnp.all(grads.param_embeddings == 0.0)


# ── HyperNetwork forward ─────────────────────────────────────────────────────


def test_hypernetwork_output_shape():
    model = make_hypernetwork(KEY)
    c_enc = encode(jnp.array(0.3), jnp.array(-0.4))
    x = _x_noise()
    out = model(c_enc, x)
    assert out.shape == ()


def test_hypernetwork_batched_via_vmap():
    model = make_hypernetwork(KEY)
    B = 8
    c_encs = jax.random.normal(KEY, (B, D))
    x = _x_noise()
    outs = jax.vmap(model, in_axes=(0, None))(c_encs, x)
    assert outs.shape == (B,)


def test_hypernetwork_output_is_finite():
    model = make_hypernetwork(KEY)
    c_encs = jax.random.normal(KEY, (32, D))
    x = _x_noise()
    outs = jax.vmap(model, in_axes=(0, None))(c_encs, x)
    assert jnp.all(jnp.isfinite(outs))


# ── gradient flow ─────────────────────────────────────────────────────────────


def test_gradients_reach_encoder():
    """Full-model grad must reach mu and log_var in gaussian_encoder."""
    model = make_hypernetwork(KEY)
    c_enc = encode(jnp.array(0.3), jnp.array(-0.4))
    x = _x_noise()
    target = jnp.array(0.5)

    grads = eqx.filter_grad(lambda m: (m(c_enc, x) - target) ** 2)(model)
    assert jnp.any(grads.gaussian_encoder.mu != 0.0)
    assert jnp.any(grads.gaussian_encoder.log_var != 0.0)
    assert jnp.any(grads.stimulus_to_coord_params.layers[0].weight != 0.0)


def test_gaussian_phase_only_updates_encoder():
    """After a gaussian-phase step, stim_params must be bit-identical."""
    import optax
    from god.hnn import make_hypernetwork
    from god.data import make_dataset
    from god.train_hnn import _task_loss, HNNConfig

    cfg = HNNConfig(n_train=32, num_steps=5, seed=1)
    key = jax.random.PRNGKey(cfg.seed)
    k_model, k_train, k_noise, _ = jax.random.split(key, 4)

    model = make_hypernetwork(k_model)
    c_train, t_train = make_dataset(cfg.n_train, cfg.num_steps, k_train, train=True)

    stim_before = model.stimulus_to_coord_params.layers[0].weight.copy()
    mu_before = model.gaussian_encoder.mu.copy()

    opt_gauss = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(learning_rate=1e-4),
    )
    opt_state_gauss = opt_gauss.init(eqx.filter(model.gaussian_encoder, eqx.is_array))
    lambda_kl = cfg.lambda_kl

    @eqx.filter_jit
    def step_gaussian(model, opt_state, c_b, t_b, x_samples):
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

    c_b, t_b = c_train[:8], t_train[:8]
    x_samples = jax.random.uniform(k_noise, (cfg.n_mc_train, cfg.n_stimulus), minval=-1.0, maxval=1.0)
    model, _, _ = step_gaussian(model, opt_state_gauss, c_b, t_b, x_samples)

    assert jnp.all(
        model.stimulus_to_coord_params.layers[0].weight == stim_before
    ), "stim_params changed during gaussian phase"
    # Encoder must have changed (mu or log_var)
    changed = not jnp.all(model.gaussian_encoder.mu == mu_before)
    assert changed, "gaussian_encoder did not update during gaussian phase"


def test_rest_phase_does_not_change_encoder():
    """After a rest-phase step, gaussian_encoder must be bit-identical."""
    import optax
    from god.hnn import make_hypernetwork
    from god.data import make_dataset
    from god.train_hnn import _task_loss, HNNConfig

    cfg = HNNConfig(n_train=32, num_steps=5, seed=2)
    key = jax.random.PRNGKey(cfg.seed)
    k_model, k_train, k_noise, _ = jax.random.split(key, 4)

    model = make_hypernetwork(k_model)
    c_train, t_train = make_dataset(cfg.n_train, cfg.num_steps, k_train, train=True)

    mu_before = model.gaussian_encoder.mu.copy()
    log_var_before = model.gaussian_encoder.log_var.copy()
    embeds_before = model.param_embeddings.copy()

    opt_rest = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=1e-4, weight_decay=0.1),
    )
    opt_state_rest = opt_rest.init(
        eqx.filter(model.stimulus_to_coord_params, eqx.is_array)
    )

    @eqx.filter_jit
    def step_rest(model, opt_state, c_b, t_b, x_samples):
        def loss_fn(stim):
            m = eqx.tree_at(lambda m: m.stimulus_to_coord_params, model, stim)
            return _task_loss(m, c_b, t_b, x_samples)

        stim = model.stimulus_to_coord_params
        loss, grads = eqx.filter_value_and_grad(loss_fn)(stim)
        updates, new_state = opt_rest.update(
            grads, opt_state, eqx.filter(stim, eqx.is_array)
        )
        new_stim = eqx.apply_updates(stim, updates)
        new_model = eqx.tree_at(lambda m: m.stimulus_to_coord_params, model, new_stim)
        return new_model, new_state, loss

    c_b, t_b = c_train[:8], t_train[:8]
    x_samples = jax.random.uniform(k_noise, (cfg.n_mc_train, cfg.n_stimulus), minval=-1.0, maxval=1.0)
    model, _, _ = step_rest(model, opt_state_rest, c_b, t_b, x_samples)

    assert jnp.all(model.gaussian_encoder.mu == mu_before)
    assert jnp.all(model.gaussian_encoder.log_var == log_var_before)
    assert jnp.all(model.param_embeddings == embeds_before), "param_embeddings drifted via weight decay"


# ── integration ───────────────────────────────────────────────────────────────


def test_loss_is_finite_at_init():
    from god.data import make_dataset
    from god.train_hnn import _task_loss

    model = make_hypernetwork(KEY)
    k1, k2 = jax.random.split(KEY)
    c_encs, targets = make_dataset(64, 5, k1, train=True)
    x_samples = jax.random.uniform(k2, (8, N_STIMULUS), minval=-1.0, maxval=1.0)
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

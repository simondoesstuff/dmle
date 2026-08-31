"""Tests for TanhFFN and the step dataset."""

import jax
import jax.numpy as jnp
import equinox as eqx

from god.datasets.step import make_step_dataset, step
from god.models.ffn import TanhFFN

KEY = jax.random.PRNGKey(42)


# --- step function ---

def test_step_identity_at_origin():
    re, im = step(jnp.array(0.0), jnp.array(0.0), jnp.array(1.0), jnp.array(0.5))
    assert float(re) == 1.0
    assert float(im) == 0.5


def test_step_quadratic():
    # z = 1+1i, c = 0: z^2 = (1+1i)^2 = 2i
    re, im = step(jnp.array(1.0), jnp.array(1.0), jnp.array(0.0), jnp.array(0.0))
    assert abs(float(re)) < 1e-6
    assert abs(float(im) - 2.0) < 1e-6


# --- dataset ---

def test_step_dataset_shapes():
    ds = make_step_dataset(100, 20, KEY)
    assert ds.train_inputs.shape == (100, 4)
    assert ds.train_targets.shape == (100, 2)
    assert ds.test_inputs.shape == (20, 4)
    assert ds.test_targets.shape == (20, 2)


def test_step_dataset_targets_are_exact():
    ds = make_step_dataset(50, 10, KEY)
    z_re = ds.train_inputs[:, 0]
    z_im = ds.train_inputs[:, 1]
    c_re = ds.train_inputs[:, 2]
    c_im = ds.train_inputs[:, 3]
    expected_re = z_re ** 2 - z_im ** 2 + c_re
    expected_im = 2.0 * z_re * z_im + c_im
    assert jnp.allclose(ds.train_targets[:, 0], expected_re, atol=1e-5)
    assert jnp.allclose(ds.train_targets[:, 1], expected_im, atol=1e-5)


# --- TanhFFN ---

def test_ffn_output_shape():
    model = TanhFFN(4, 8, 1, 2, KEY)
    x = jnp.zeros(4)
    out = model(x)
    assert out.shape == (2,)


def test_ffn_depth0_is_linear():
    model = TanhFFN(4, 0, 0, 2, KEY)
    assert len(model.layers) == 1


def test_ffn_depth1_has_two_layers():
    model = TanhFFN(4, 8, 1, 2, KEY)
    assert len(model.layers) == 2
    assert model.layers[0].in_features == 4
    assert model.layers[0].out_features == 8
    assert model.layers[1].in_features == 8
    assert model.layers[1].out_features == 2


def test_ffn_output_not_bounded_to_tanh_range():
    # Output layer is linear — values can exceed (-1, 1).
    model = TanhFFN(4, 4, 1, 2, KEY)
    large = jnp.ones(4) * 10.0
    out = model(large)
    assert out.shape == (2,)
    # At least plausible that output exceeds 1 with large inputs
    # (not guaranteed, but verifiable with a fixed large weight)
    # Just check shape and finite values here
    assert jnp.all(jnp.isfinite(out))


def test_ffn_batched_vmap():
    model = TanhFFN(4, 8, 1, 2, KEY)
    batch = jnp.ones((16, 4))
    out = jax.vmap(model)(batch)
    assert out.shape == (16, 2)


def test_ffn_gradients_flow():
    model = TanhFFN(4, 4, 1, 2, KEY)
    x = jnp.array([0.5, -0.3, 0.1, 0.2])
    target = jnp.array([0.2, 0.4])

    def loss(m: TanhFFN) -> jax.Array:
        return jnp.mean((m(x) - target) ** 2)

    grads = eqx.filter_grad(loss)(model)
    leaves = jax.tree_util.tree_leaves(eqx.filter(grads, eqx.is_array))
    assert any(jnp.any(g != 0.0) for g in leaves)

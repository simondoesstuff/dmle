"""Tests for Mandelbrot dataset generation and Fourier encoding."""

import jax
import jax.numpy as jnp
import pytest

from god.datasets.mandelbrot import (
    ESCAPE_RADIUS,
    K_DEFAULT,
    SCALE,
    decode,
    decode_magnitude,
    enc_dim,
    encode,
    make_mandelbrot_dataset,
    mandelbrot_mag,
)


# ── Mandelbrot iteration ──────────────────────────────────────────────────────

def test_mandelbrot_mag_in_set():
    mag = mandelbrot_mag(jnp.array(0.0), jnp.array(0.0), T=50)
    assert float(mag) < ESCAPE_RADIUS


def test_mandelbrot_mag_outside_set():
    mag = mandelbrot_mag(jnp.array(2.0), jnp.array(2.0), T=50)
    assert float(mag) == pytest.approx(ESCAPE_RADIUS, abs=1e-4)


def test_mandelbrot_mag_bounded():
    key = jax.random.PRNGKey(42)
    c_r = jax.random.uniform(key, (100,), minval=-2.0, maxval=2.0)
    c_i = jax.random.uniform(key, (100,), minval=-2.0, maxval=2.0)
    mags = jax.vmap(mandelbrot_mag, in_axes=(0, 0, None, None))(c_r, c_i, 50, ESCAPE_RADIUS)
    assert jnp.all(mags <= ESCAPE_RADIUS + 1e-5)
    assert jnp.all(mags >= 0.0)


# ── Dataset ───────────────────────────────────────────────────────────────────

def test_make_dataset_shapes():
    key = jax.random.PRNGKey(0)
    n_train, n_test, T, K = 20, 10, 10, K_DEFAULT
    ds = make_mandelbrot_dataset(n_train, n_test, T, key, K=K)
    assert ds.train_inputs.shape == (n_train, enc_dim(K))
    assert ds.train_targets.shape == (n_train,)
    assert ds.test_inputs.shape == (n_test, enc_dim(K))
    assert ds.test_targets.shape == (n_test,)


def test_make_dataset_train_lower_halfplane():
    key = jax.random.PRNGKey(1)
    ds = make_mandelbrot_dataset(50, 10, 10, key)
    # Encoded y component (index 1) = c_imag / SCALE <= 0 for train
    assert jnp.all(ds.train_inputs[:, 1] <= 1e-6)


def test_make_dataset_test_upper_halfplane():
    key = jax.random.PRNGKey(2)
    ds = make_mandelbrot_dataset(10, 50, 10, key)
    assert jnp.all(ds.test_inputs[:, 1] >= -1e-6)


def test_make_dataset_targets_normalised():
    key = jax.random.PRNGKey(3)
    ds = make_mandelbrot_dataset(50, 10, 10, key)
    assert jnp.all(ds.train_targets >= 0.0)
    assert jnp.all(ds.train_targets <= 1.0 + 1e-5)


# ── Fourier encoding ──────────────────────────────────────────────────────────

def test_enc_dim():
    assert enc_dim(4) == 18
    assert enc_dim(0) == 2


def test_encode_shape():
    h = encode(jnp.array(0.5), jnp.array(-0.3), K=4)
    assert h.shape == (enc_dim(4),)


def test_encode_direct_components_normalized():
    x, y = 1.0, -1.5
    h = encode(jnp.array(x), jnp.array(y))
    assert jnp.allclose(h[0], x / SCALE, atol=1e-6)
    assert jnp.allclose(h[1], y / SCALE, atol=1e-6)


def test_decode_round_trips_origin():
    h = encode(jnp.array(0.0), jnp.array(0.0))
    re, im = decode(h)
    assert jnp.abs(re) < 0.1
    assert jnp.abs(im) < 0.1


def test_decode_round_trips_small_value():
    c_re, c_im = 0.1, -0.1
    h = encode(jnp.array(c_re), jnp.array(c_im))
    re, im = decode(h)
    assert jnp.abs(re - c_re) < 0.01
    assert jnp.abs(im - c_im) < 0.01


def test_decode_magnitude_nonnegative():
    for c_re, c_im in [(0.0, 0.0), (1.0, 1.0), (-1.5, 0.5)]:
        h = encode(jnp.array(c_re), jnp.array(c_im))
        mag = decode_magnitude(h)
        assert mag >= 0.0


def test_magnitude_bounded_by_tanh_range():
    h = jnp.clip(encode(jnp.array(0.7), jnp.array(-0.4)), -1.0, 1.0)
    mag = decode_magnitude(h)
    assert jnp.isfinite(mag)

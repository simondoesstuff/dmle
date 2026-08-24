import jax
import jax.numpy as jnp
import pytest
from god.data import mandelbrot_mag, make_dataset, ESCAPE_RADIUS
from god.encoding import enc_dim, K_DEFAULT


def test_mandelbrot_mag_in_set():
    # c=0 is in the Mandelbrot set; |z_T| should stay small
    mag = mandelbrot_mag(jnp.array(0.0), jnp.array(0.0), T=50)
    assert float(mag) < ESCAPE_RADIUS


def test_mandelbrot_mag_outside_set():
    # c=2+2i diverges immediately; magnitude should reach escape radius
    mag = mandelbrot_mag(jnp.array(2.0), jnp.array(2.0), T=50)
    assert float(mag) == pytest.approx(ESCAPE_RADIUS, abs=1e-4)


def test_mandelbrot_mag_bounded():
    key = jax.random.PRNGKey(42)
    c_r = jax.random.uniform(key, (100,), minval=-2.0, maxval=2.0)
    c_i = jax.random.uniform(key, (100,), minval=-2.0, maxval=2.0)
    mags = jax.vmap(mandelbrot_mag, in_axes=(0, 0, None, None))(c_r, c_i, 50, ESCAPE_RADIUS)
    assert jnp.all(mags <= ESCAPE_RADIUS + 1e-5)
    assert jnp.all(mags >= 0.0)


def test_make_dataset_shapes():
    key = jax.random.PRNGKey(0)
    n, T, K = 20, 10, K_DEFAULT
    c_encs, targets = make_dataset(n, T, key, train=True, K=K)
    assert c_encs.shape == (n, enc_dim(K))
    assert targets.shape == (n,)


def test_make_dataset_train_lower_halfplane():
    key = jax.random.PRNGKey(1)
    # The encoded y component (index 1) should be <= 0 for train
    c_encs, _ = make_dataset(50, 10, key, train=True)
    # h[1] = c_imag / SCALE <= 0
    assert jnp.all(c_encs[:, 1] <= 1e-6)


def test_make_dataset_test_upper_halfplane():
    key = jax.random.PRNGKey(2)
    c_encs, _ = make_dataset(50, 10, key, train=False)
    assert jnp.all(c_encs[:, 1] >= -1e-6)


def test_make_dataset_targets_normalised():
    key = jax.random.PRNGKey(3)
    _, targets = make_dataset(50, 10, key, train=True)
    assert jnp.all(targets >= 0.0)
    assert jnp.all(targets <= 1.0 + 1e-5)

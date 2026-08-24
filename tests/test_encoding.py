import jax.numpy as jnp
import pytest
from god.encoding import encode, decode, decode_magnitude, enc_dim, SCALE, K_DEFAULT


def test_enc_dim():
    assert enc_dim(4) == 18
    assert enc_dim(0) == 2


def test_encode_shape():
    h = encode(jnp.array(0.5), jnp.array(-0.3), K=4)
    assert h.shape == (enc_dim(4),)


def test_encode_direct_components_normalized():
    # First two elements should be the normalised coordinates
    x, y = 1.0, -1.5
    h = encode(jnp.array(x), jnp.array(y))
    assert jnp.allclose(h[0], x / SCALE, atol=1e-6)
    assert jnp.allclose(h[1], y / SCALE, atol=1e-6)


def test_decode_round_trips_origin():
    # At (0, 0) every frequency gives 0 phase; decode should return ~(0, 0)
    h = encode(jnp.array(0.0), jnp.array(0.0))
    re, im = decode(h)
    assert jnp.abs(re) < 0.1
    assert jnp.abs(im) < 0.1


def test_decode_round_trips_small_value():
    # K=4 bands are unambiguous for |x_norm| < 1/2^K = 0.0625, i.e. |c| < 0.125.
    # Use c well within that range so all bands agree and the pool is accurate.
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
    # If h is produced by tanh outputs (values in [-1,1]), decoded mag should be finite
    h = jnp.clip(encode(jnp.array(0.7), jnp.array(-0.4)), -1.0, 1.0)
    mag = decode_magnitude(h)
    assert jnp.isfinite(mag)

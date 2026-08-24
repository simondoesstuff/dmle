"""Fourier encoding/decoding for complex numbers.

Maps z = x + yi to a high-dimensional representation that is smooth and
gradient-friendly for BPTT through chaotic dynamics.

Encoding layout (dim = 2 + 4K):
  [x_n, y_n, sin(π x_n), cos(π x_n), sin(π y_n), cos(π y_n),
         sin(2π x_n), cos(2π x_n), sin(2π y_n), cos(2π y_n), ...]
where x_n = x / SCALE, y_n = y / SCALE normalizes to [-1, 1].
"""

import jax.numpy as jnp
import einx
from jaxtyping import Array, Float

SCALE = 2.0  # Mandelbrot sampling range ±SCALE
_EPS = 1e-8

K_DEFAULT = 4


def enc_dim(K: int = K_DEFAULT) -> int:
    return 2 + 4 * K


def encode(
    c_real: Float[Array, ""],
    c_imag: Float[Array, ""],
    K: int = K_DEFAULT,
) -> Float[Array, "d"]:
    x = c_real / SCALE
    y = c_imag / SCALE
    freqs = 2.0 ** jnp.arange(K)  # [1, 2, 4, ..., 2^(K-1)]
    phases_x = einx.multiply("k, -> k", freqs * jnp.pi, x)
    phases_y = einx.multiply("k, -> k", freqs * jnp.pi, y)
    # (K, 4): [sin_x, cos_x, sin_y, cos_y] per frequency
    fourier = jnp.stack(
        [jnp.sin(phases_x), jnp.cos(phases_x), jnp.sin(phases_y), jnp.cos(phases_y)],
        axis=1,
    ).reshape(-1)
    return jnp.concatenate([jnp.array([x, y]), fourier])


def decode(
    h: Float[Array, "d"],
    K: int = K_DEFAULT,
) -> tuple[Float[Array, ""], Float[Array, ""]]:
    """Pool direct + K Fourier estimates to recover (re, im) in Mandelbrot scale."""
    x = h[0]
    y = h[1]
    fourier = h[2:].reshape(K, 4)  # (K, [sin_x, cos_x, sin_y, cos_y])
    freqs = 2.0 ** jnp.arange(K)

    # atan2(sin θ, cos θ) / (2^k π) ≈ x_norm when x_norm ∈ [-1/2^k, 1/2^k]
    x_ests = jnp.arctan2(fourier[:, 0] + _EPS, fourier[:, 1] + _EPS) / (freqs * jnp.pi)
    y_ests = jnp.arctan2(fourier[:, 2] + _EPS, fourier[:, 3] + _EPS) / (freqs * jnp.pi)

    # Pool: include direct (x, y) component and all K Fourier estimates
    x_pool = jnp.concatenate([jnp.array([x]), x_ests])
    y_pool = jnp.concatenate([jnp.array([y]), y_ests])
    x_out = einx.mean("k ->", x_pool) * SCALE
    y_out = einx.mean("k ->", y_pool) * SCALE
    return x_out, y_out


def decode_magnitude(
    h: Float[Array, "d"],
    K: int = K_DEFAULT,
) -> Float[Array, ""]:
    """Magnitude from h for use in the loss.

    Uses only the direct (x, y) head — h[0] and h[1] — because the atan2-based
    Fourier estimates have near-zero gradient when sin/cos features start small,
    which blocks learning. The full `decode` (with pooling) is available for
    analysis after training converges.
    """
    del K  # unused; signature kept for API consistency
    re = h[0] * SCALE
    im = h[1] * SCALE
    return jnp.sqrt(re**2 + im**2 + _EPS**2)

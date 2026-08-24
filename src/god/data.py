"""Mandelbrot dataset generation.

Reference iteration: z_{t+1} = z_t^2 + c, z_0 = c (matching h_0 = x = encode(c)).
Magnitude is clamped to escape_radius at each step to prevent overflow.

Train split: Im(c) <= 0  (lower half-plane)
Test split:  Im(c) >= 0  (upper half-plane)
"""

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from god.encoding import SCALE, encode

ESCAPE_RADIUS = 2.0


def mandelbrot_mag(
    c_real: Float[Array, ""],
    c_imag: Float[Array, ""],
    T: int,
    escape_radius: float = ESCAPE_RADIUS,
) -> Float[Array, ""]:
    """Iterate z = z^2 + c for T steps starting at z_0 = c; return |z_T| clamped."""

    def step(z: tuple[Array, Array], _: None) -> tuple[tuple[Array, Array], None]:
        re, im = z
        re_new = re * re - im * im + c_real
        im_new = 2.0 * re * im + c_imag
        mag = jnp.sqrt(re_new**2 + im_new**2)
        scale = jnp.where(mag > escape_radius, escape_radius / (mag + 1e-12), 1.0)
        return (re_new * scale, im_new * scale), None

    (re_T, im_T), _ = jax.lax.scan(step, (c_real, c_imag), None, length=T)
    return jnp.sqrt(re_T**2 + im_T**2)


_mandelbrot_mag_vmap = jax.jit(
    jax.vmap(mandelbrot_mag, in_axes=(0, 0, None, None)),
    static_argnums=(2, 3),
)


def make_dataset(
    n: int,
    T: int,
    key: PRNGKeyArray,
    *,
    train: bool = True,
    K: int = 4,
    escape_radius: float = ESCAPE_RADIUS,
) -> tuple[Float[Array, "n d"], Float[Array, "n"]]:
    """Sample c uniformly from the Mandelbrot view and encode it.

    Returns:
        c_encs: (n, enc_dim) Fourier-encoded complex inputs
        targets: (n,) target magnitude in [0, 1] (normalised by escape_radius)
    """
    # Sample complex coordinates uniformly in [-SCALE, SCALE]^2
    k1, k2 = jax.random.split(key)
    c_real = jax.random.uniform(k1, (n,), minval=-SCALE, maxval=SCALE)
    c_imag = jax.random.uniform(k2, (n,), minval=-SCALE, maxval=SCALE)

    # Enforce half-plane split
    if train:
        c_imag = -jnp.abs(c_imag)  # Im(c) <= 0
    else:
        c_imag = jnp.abs(c_imag)   # Im(c) >= 0

    # Encode each c
    c_encs = jax.vmap(encode, in_axes=(0, 0, None))(c_real, c_imag, K)

    # Compute reference magnitudes, normalised to [0, 1]
    mags = _mandelbrot_mag_vmap(c_real, c_imag, T, escape_radius)
    targets = mags / escape_radius

    return c_encs, targets

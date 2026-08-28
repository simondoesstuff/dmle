"""Mandelbrot dataset: Fourier encoding and dataset generation.

Fourier encoding maps z = x + yi to a high-dimensional representation that is
smooth and gradient-friendly for BPTT through chaotic dynamics.

Encoding layout (dim = 2 + 4K):
  [x_n, y_n, sin(π x_n), cos(π x_n), sin(π y_n), cos(π y_n),
         sin(2π x_n), cos(2π x_n), sin(2π y_n), cos(2π y_n), ...]
where x_n = x / SCALE, y_n = y / SCALE normalizes to [-1, 1].

Train split: Im(c) <= 0  (lower half-plane)
Test split:  Im(c) >= 0  (upper half-plane)
"""

import jax
import jax.numpy as jnp
import einx
from jaxtyping import Array, Float, PRNGKeyArray

from god.datasets.base import Dataset

SCALE = 2.0
ESCAPE_RADIUS = 2.0
EPS = 1e-8
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
    freqs = 2.0 ** jnp.arange(K)
    phases_x = einx.multiply("k, -> k", freqs * jnp.pi, x)
    phases_y = einx.multiply("k, -> k", freqs * jnp.pi, y)
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
    fourier = h[2:].reshape(K, 4)
    freqs = 2.0 ** jnp.arange(K)
    x_ests = jnp.arctan2(fourier[:, 0] + EPS, fourier[:, 1] + EPS) / (freqs * jnp.pi)
    y_ests = jnp.arctan2(fourier[:, 2] + EPS, fourier[:, 3] + EPS) / (freqs * jnp.pi)
    x_pool = jnp.concatenate([jnp.array([x]), x_ests])
    y_pool = jnp.concatenate([jnp.array([y]), y_ests])
    x_out = einx.mean("k ->", x_pool) * SCALE
    y_out = einx.mean("k ->", y_pool) * SCALE
    return x_out, y_out


def decode_magnitude(
    h: Float[Array, "d"],
    K: int = K_DEFAULT,
) -> Float[Array, ""]:
    """Magnitude from h using only the direct (x, y) head.

    Avoids the near-zero-gradient problem of atan2-based Fourier estimates
    during early training. Full decode (with pooling) is available for analysis.
    """
    del K
    re = h[0] * SCALE
    im = h[1] * SCALE
    return jnp.sqrt(re**2 + im**2 + EPS**2)


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


def _make_split(
    n: int,
    T: int,
    key: PRNGKeyArray,
    *,
    train: bool,
    K: int,
    escape_radius: float,
) -> tuple[Float[Array, "n d"], Float[Array, "n"]]:
    k1, k2 = jax.random.split(key)
    c_real = jax.random.uniform(k1, (n,), minval=-SCALE, maxval=SCALE)
    c_imag = jax.random.uniform(k2, (n,), minval=-SCALE, maxval=SCALE)
    if train:
        c_imag = -jnp.abs(c_imag)
    else:
        c_imag = jnp.abs(c_imag)
    c_encs = jax.vmap(encode, in_axes=(0, 0, None))(c_real, c_imag, K)
    mags = _mandelbrot_mag_vmap(c_real, c_imag, T, escape_radius)
    targets = mags / escape_radius
    return c_encs, targets


def make_mandelbrot_dataset(
    n_train: int,
    n_test: int,
    T: int,
    key: PRNGKeyArray,
    *,
    K: int = K_DEFAULT,
    escape_radius: float = ESCAPE_RADIUS,
) -> Dataset:
    """Sample c uniformly and encode it for both train and test splits.

    Train: Im(c) <= 0 (lower half-plane); Test: Im(c) >= 0 (upper half-plane).
    Targets are magnitudes normalised to [0, 1] by escape_radius.
    """
    k_train, k_test = jax.random.split(key)
    train_inputs, train_targets = _make_split(n_train, T, k_train, train=True, K=K, escape_radius=escape_radius)
    test_inputs, test_targets = _make_split(n_test, T, k_test, train=False, K=K, escape_radius=escape_radius)
    return Dataset(
        train_inputs=train_inputs,
        train_targets=train_targets,
        test_inputs=test_inputs,
        test_targets=test_targets,
    )

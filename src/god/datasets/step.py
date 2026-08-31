"""Single-step Mandelbrot dataset: learn z -> z^2 + c.

Domain: z uniformly from [-z_range, z_range]^2, c uniformly from the
"slightly larger than main Mandelbrot bulb" rectangle, sampled independently.

Targets are the exact one-step iterate (re(z^2+c), im(z^2+c)).
"""

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from god.datasets.base import Dataset

# Covers main cardioid + period-2 bulb, scaled ~1.3x
C_REAL_MIN, C_REAL_MAX = -1.5, 0.5
C_IMAG_MIN, C_IMAG_MAX = -0.85, 0.85
Z_RANGE = 1.5


def step(
    z_re: Float[Array, ""],
    z_im: Float[Array, ""],
    c_re: Float[Array, ""],
    c_im: Float[Array, ""],
) -> tuple[Float[Array, ""], Float[Array, ""]]:
    """One Mandelbrot iterate: z -> z^2 + c (exact)."""
    return z_re * z_re - z_im * z_im + c_re, 2.0 * z_re * z_im + c_im


def make_step_dataset(
    n_train: int,
    n_test: int,
    key: PRNGKeyArray,
    *,
    z_range: float = Z_RANGE,
) -> Dataset:
    """Sample (z, c) pairs independently; targets are exact z^2+c.

    Train/test split is a random 80/20 over the same domain (no deliberate
    extrapolation — we're studying approximation, not generalisation).
    """
    k1, k2, k3, k4 = jax.random.split(key, 4)
    n_total = n_train + n_test

    z_re = jax.random.uniform(k1, (n_total,), minval=-z_range, maxval=z_range)
    z_im = jax.random.uniform(k2, (n_total,), minval=-z_range, maxval=z_range)
    c_re = jax.random.uniform(k3, (n_total,), minval=C_REAL_MIN, maxval=C_REAL_MAX)
    c_im = jax.random.uniform(k4, (n_total,), minval=C_IMAG_MIN, maxval=C_IMAG_MAX)

    out_re, out_im = jax.vmap(step)(z_re, z_im, c_re, c_im)

    inputs = jnp.stack([z_re, z_im, c_re, c_im], axis=1)   # (n, 4)
    targets = jnp.stack([out_re, out_im], axis=1)            # (n, 2)

    return Dataset(
        train_inputs=inputs[:n_train],
        train_targets=targets[:n_train],
        test_inputs=inputs[n_train:],
        test_targets=targets[n_train:],
    )

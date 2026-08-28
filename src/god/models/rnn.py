"""Recurrent Mandelbrot approximator.

RNNCell: depth-parameterised tanh FFN.
  depth=1 → Linear(2d→d) → tanh  (hidden_dim unused)
  depth=N → [2d→h→…→h→d], all tanh

MandelbrotRNN: applies the cell num_steps times with fixed input c, then
reads out a predicted magnitude via one of two heads:
  linear_head=False  →  h_T[0]*SCALE, h_T[1]*SCALE  (fixed-index projection)
  linear_head=True   →  Linear(enc_dim→2) applied to h_T  (learned projection)
"""

import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array, Float, PRNGKeyArray

from god.datasets.mandelbrot import SCALE, EPS


class RNNCell(eqx.Module):
    """(h_t, x) -> h_{t+1} via an N-layer tanh FFN."""

    layers: list[eqx.nn.Linear]

    def __init__(
        self,
        enc_dim: int,
        hidden_dim: int,
        depth: int,
        key: PRNGKeyArray,
    ) -> None:
        super().__init__()
        keys = jax.random.split(key, depth)
        if depth == 1:
            dims = [2 * enc_dim, enc_dim]
        else:
            dims = [2 * enc_dim] + [hidden_dim] * (depth - 1) + [enc_dim]
        self.layers = [
            eqx.nn.Linear(dims[i], dims[i + 1], key=keys[i])
            for i in range(depth)
        ]

    def __call__(
        self,
        h_t: Float[Array, "d"],
        x: Float[Array, "d"],
    ) -> Float[Array, "d"]:
        h = jnp.concatenate([h_t, x])
        for layer in self.layers:
            h = jnp.tanh(layer(h))
        return h


class MandelbrotRNN(eqx.Module):
    """Runs RNNCell for num_steps steps with h_0 = x = encode(c).

    Output head options:
      head=None   → use h_T[0], h_T[1] directly (fixed-index projection)
      head=Linear → learned Linear(enc_dim→2) maps all of h_T to (re, im)
    """

    cell: RNNCell
    head: eqx.nn.Linear | None
    num_steps: int = eqx.field(static=True)

    def __init__(
        self,
        enc_dim: int,
        hidden_dim: int,
        depth: int,
        num_steps: int,
        key: PRNGKeyArray,
        *,
        linear_head: bool = False,
    ) -> None:
        super().__init__()
        k_cell, k_head = jax.random.split(key)
        self.cell = RNNCell(enc_dim, hidden_dim, depth, k_cell)
        self.head = eqx.nn.Linear(enc_dim, 2, key=k_head) if linear_head else None
        self.num_steps = num_steps

    def __call__(self, c_enc: Float[Array, "d"]) -> Float[Array, "d"]:
        def step(h: Array, _: None) -> tuple[Array, None]:
            return self.cell(h, c_enc), None

        h_T, _ = jax.lax.scan(step, c_enc, None, length=self.num_steps)
        return h_T

    def predict_magnitude(self, h_T: Float[Array, "d"]) -> Float[Array, ""]:
        """Map h_T to a predicted Mandelbrot magnitude (Mandelbrot scale)."""
        if self.head is not None:
            out = self.head(h_T)
            re, im = out[0] * SCALE, out[1] * SCALE
        else:
            re, im = h_T[0] * SCALE, h_T[1] * SCALE
        return jnp.sqrt(re**2 + im**2 + EPS**2)

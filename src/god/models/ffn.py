"""Tanh FFN for learning the single-step Mandelbrot map z -> z^2 + c.

Architecture: [in_dim] -tanh-> [hidden]*depth -linear-> [out_dim]
The output layer is always linear so the network isn't artificially capped.
"""

import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array, Float, PRNGKeyArray


class TanhFFN(eqx.Module):
    """Depth-hidden-layer tanh FFN with linear readout.

    depth=0: single linear map (no hidden layers)
    depth>=1: depth tanh layers, then one linear layer
    """

    layers: list[eqx.nn.Linear]
    depth: int = eqx.field(static=True)

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        depth: int,
        out_dim: int,
        key: PRNGKeyArray,
    ) -> None:
        super().__init__()
        self.depth = depth
        dims = [in_dim] + [hidden_dim] * depth + [out_dim]
        keys = jax.random.split(key, len(dims) - 1)
        self.layers = [
            eqx.nn.Linear(dims[i], dims[i + 1], key=keys[i])
            for i in range(len(dims) - 1)
        ]

    def __call__(self, x: Float[Array, "in_dim"]) -> Float[Array, "out_dim"]:
        for layer in self.layers[:-1]:
            x = jnp.tanh(layer(x))
        return self.layers[-1](x)

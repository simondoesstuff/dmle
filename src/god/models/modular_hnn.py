"""HyperNetwork for modular addition classification.

Architecture:
    x (noise ∈ [-1, 1]^n_stimulus)
       │
       ▼
    stimulus_ffn  (learnable MLP)
       │
       ▼
    flat target FFN parameters  (n_target_params,)
       │
       ▼
    target FFN: inputs → [hidden]*depth → logits (out_dim classes)
       (tanh hidden layers + linear output, matching TanhFFN convention)
       │
       ▼
    class logits  (out_dim,)

Different x values generate different target classifiers. Training over many x
samples forces the stimulus FFN to produce classifiers that generalise regardless
of x — weight decay then collapses them toward a single compact solution.
"""

import math

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray


def _unflatten(
    flat: Float[Array, "n_params"],
    layout: list[tuple[str, tuple[int, ...]]],
) -> dict[str, Float[Array, "..."]]:
    result: dict[str, Float[Array, "..."]] = {}
    offset = 0
    for name, shape in layout:
        n = math.prod(shape)
        result[name] = flat[offset : offset + n].reshape(shape)
        offset += n
    return result


class ModularHNN(eqx.Module):
    stimulus_ffn: eqx.nn.MLP
    param_layout: list[tuple[str, tuple[int, ...]]] = eqx.field(static=True)
    n_target_layers: int = eqx.field(static=True)

    def target_params(
        self, x_noise: Float[Array, "n_stimulus"]
    ) -> Float[Array, "n_params"]:
        return self.stimulus_ffn(x_noise)

    def __call__(
        self,
        inputs: Float[Array, "in_dim"],
        x_noise: Float[Array, "n_stimulus"],
    ) -> Float[Array, "out_dim"]:
        params = _unflatten(self.target_params(x_noise), self.param_layout)
        act = inputs
        for l in range(self.n_target_layers - 1):
            act = jnp.tanh(params[f"W{l}"] @ act + params[f"b{l}"])
        l = self.n_target_layers - 1
        return params[f"W{l}"] @ act + params[f"b{l}"]


def make_modular_hypernetwork(
    key: PRNGKeyArray,
    *,
    in_dim: int = 195,
    out_dim: int = 97,
    target_hidden: int = 32,
    target_depth: int = 1,
    n_stimulus: int = 8,
    stim_ffn_hidden: int = 8,
    stim_ffn_depth: int = 1,
    init_scale: float = 0.1,
) -> ModularHNN:
    """Build a ModularHNN.

    Args:
        target_hidden: hidden width of the generated target FFN.
        target_depth: number of hidden tanh layers in the target FFN.
        n_stimulus: dimension of the noise vector x.
        stim_ffn_hidden: hidden width of the stimulus MLP.
        init_scale: scale factor for the stimulus FFN output layer (keeps
            generated target weights small at init to avoid gradient blowup).
    """
    dims = [in_dim] + [target_hidden] * target_depth + [out_dim]
    param_layout: list[tuple[str, tuple[int, ...]]] = []
    for l, (i, o) in enumerate(zip(dims, dims[1:])):
        param_layout.append((f"W{l}", (o, i)))
        param_layout.append((f"b{l}", (o,)))

    n_target_params = sum(math.prod(s) for _, s in param_layout)
    n_target_layers = len(dims) - 1

    stimulus_ffn = eqx.nn.MLP(
        in_size=n_stimulus,
        out_size=n_target_params,
        width_size=stim_ffn_hidden,
        depth=stim_ffn_depth,
        activation=jax.nn.tanh,
        key=key,
    )
    assert stimulus_ffn.layers[-1].bias is not None
    stimulus_ffn = eqx.tree_at(
        lambda m: (m.layers[-1].weight, m.layers[-1].bias),
        stimulus_ffn,
        (
            stimulus_ffn.layers[-1].weight * init_scale,
            stimulus_ffn.layers[-1].bias * init_scale,
        ),
    )

    return ModularHNN(
        stimulus_ffn=stimulus_ffn,
        param_layout=param_layout,
        n_target_layers=n_target_layers,
    )


def n_trainable_params(model: ModularHNN) -> int:
    return sum(
        x.size for x in jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array))
    )

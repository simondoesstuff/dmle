"""Hypernetwork (HNN) for Mandelbrot magnitude prediction.

Architecture:
    x (noise ∈ [-1, 1]^n_stimulus)
       │
       ▼
    stimulus_ffn  (learnable MLP)
       │
       ▼
    flat target RNN parameters  (n_target_params,)
       │
       ▼
    target RNN cell — depth hidden tanh layers + linear output
    h_0 = c_enc;  h_{t+1} = cell(h_t, c_enc)  [rnn_num_steps times]
       │
       ▼
    sqrt(h_T[0]² + h_T[1]²)  →  predicted |z_T| / ESCAPE_RADIUS

The stimulus is the raw noise vector x ~ U[-1,1]^n_stimulus, passed directly
into the stimulus FFN. Different x values generate different target networks.
"""

import math

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from god.datasets.mandelbrot import EPS, K_DEFAULT, enc_dim


def _unflatten_target_params(
    flat_params: Float[Array, "n_params"],
    param_layout: list[tuple[str, tuple[int, ...]]],
) -> dict[str, Float[Array, "..."]]:
    result: dict[str, Float[Array, "..."]] = {}
    offset = 0
    for name, shape in param_layout:
        n = math.prod(shape)
        result[name] = flat_params[offset : offset + n].reshape(shape)
        offset += n
    return result


def _run_rnn_cell(
    params: dict[str, Float[Array, "..."]],
    x: Float[Array, "enc_dim"],
    num_steps: int,
    n_cell_layers: int,
) -> Float[Array, "enc_dim"]:
    """Run the generated RNN cell for num_steps steps; return h_T.

    Applies tanh to layers 0..n_cell_layers-2 and linear to the last layer,
    matching TanhFFN convention (depth hidden tanh layers + linear output).
    """
    def cell_step(h: Float[Array, "enc_dim"], _: None) -> tuple[Float[Array, "enc_dim"], None]:
        act = jnp.concatenate([h, x])
        for l in range(n_cell_layers - 1):
            act = jnp.tanh(params[f"W{l}"] @ act + params[f"b{l}"])
        l = n_cell_layers - 1
        act = params[f"W{l}"] @ act + params[f"b{l}"]
        return act, None

    h_T, _ = jax.lax.scan(cell_step, x, None, length=num_steps)
    return h_T


class HyperNetwork(eqx.Module):
    stimulus_ffn: eqx.nn.MLP

    param_layout: list[tuple[str, tuple[int, ...]]] = eqx.field(static=True)
    n_cell_layers: int = eqx.field(static=True)
    rnn_num_steps: int = eqx.field(static=True)

    def target_params(
        self,
        x_noise: Float[Array, "n_stimulus"],
    ) -> Float[Array, "n_params"]:
        """Generate the flat target parameter vector from stimulus x_noise."""
        return self.stimulus_ffn(x_noise)

    def __call__(
        self,
        c_enc: Float[Array, "enc_dim"],
        x_noise: Float[Array, "n_stimulus"],
    ) -> Float[Array, ""]:
        flat_params = self.target_params(x_noise)
        params = _unflatten_target_params(flat_params, self.param_layout)
        h_T = _run_rnn_cell(params, c_enc, self.rnn_num_steps, self.n_cell_layers)
        return jnp.sqrt(h_T[0] ** 2 + h_T[1] ** 2 + EPS ** 2)


def make_hypernetwork(
    key: PRNGKeyArray,
    *,
    K: int = K_DEFAULT,
    n_stimulus: int = 16,
    rnn_hidden_dim: int = 16,
    rnn_depth: int = 1,
    rnn_num_steps: int = 10,
    stim_ffn_hidden: int = 64,
    stim_ffn_depth: int = 1,
    init_scale: float = 0.1,
) -> HyperNetwork:
    """Build a HyperNetwork with the given architecture.

    Args:
        rnn_depth: number of hidden tanh layers in the target cell (depth=1 → one
            hidden tanh layer then one linear output layer, matching TanhFFN convention).
    """
    d = enc_dim(K)
    n_cell_layers = rnn_depth + 1  # hidden tanh layers + 1 linear output

    cell_dims = [2 * d] + [rnn_hidden_dim] * rnn_depth + [d]
    param_layout: list[tuple[str, tuple[int, ...]]] = []
    for l, (in_sz, out_sz) in enumerate(zip(cell_dims, cell_dims[1:])):
        param_layout.append((f"W{l}", (out_sz, in_sz)))
        param_layout.append((f"b{l}", (out_sz,)))

    n_target_params = sum(math.prod(shape) for _, shape in param_layout)

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
        (stimulus_ffn.layers[-1].weight * init_scale, stimulus_ffn.layers[-1].bias * init_scale),
    )

    return HyperNetwork(
        stimulus_ffn=stimulus_ffn,
        param_layout=param_layout,
        n_cell_layers=n_cell_layers,
        rnn_num_steps=rnn_num_steps,
    )


def n_trainable_params(model: HyperNetwork) -> int:
    return sum(
        x.size for x in jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array))
    )


def target_network_diversity(
    model: HyperNetwork,
    c_encs: Float[Array, "n_points enc_dim"],
    x_samples: Float[Array, "n_mc n_stimulus"],
) -> dict[str, float]:
    """Measure functional diversity of sampled target networks."""
    all_params = jax.vmap(model.target_params)(x_samples)
    param_std = float(jnp.mean(jnp.std(all_params, axis=0)))

    per_x_preds = jax.vmap(
        lambda x: jax.vmap(model, in_axes=(0, None))(c_encs, x)
    )(x_samples)
    pred_std = float(jnp.mean(jnp.std(per_x_preds, axis=0)))
    pred_range = float(jnp.mean(jnp.max(per_x_preds, axis=0) - jnp.min(per_x_preds, axis=0)))

    return {"param_std": param_std, "pred_std": pred_std, "pred_range": pred_range}

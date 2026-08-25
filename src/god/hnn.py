"""Hypernetwork (HNN) for Mandelbrot magnitude prediction.

Architecture:
    x (noise ∈ [-1, 1]^n_stimulus) → GaussianEncoder → stimulus
    stimulus → stimulus_to_coord_params (FFN) → coord net weights
    param_embeddings (frozen) ──(coord net)──→ flat target net params
    target net params → target RNN → scalar magnitude prediction

GaussianEncoder uses the reparameterisation trick with learnable μ, log σ²:
    stimulus = x · σ · temperature + μ
Regularised by KL(N(μ, diag(σ²)) ‖ N(0, I)), bounding entropy above and below.

Oscillatory training:
  - Gaussian phase: update GaussianEncoder (μ, log σ²) with KL regularisation
  - Rest phase: update stimulus_to_coord_params
"""

import math
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from god.encoding import K_DEFAULT, enc_dim


class CoordWeights(NamedTuple):
    W1: Float[Array, "coord_hidden embed_dim"]
    b1: Float[Array, "coord_hidden"]
    W2: Float[Array, "1 coord_hidden"]
    b2: Float[Array, "1"]


class GaussianEncoder(eqx.Module):
    """Reparameterised Gaussian encoder: stimulus = x · σ · temperature + μ.

    x ∈ [-1, 1]^n_stimulus is provided externally each forward pass.
    Learnable μ and log σ² (= log_var) parameterise the distribution;
    exp(log_var) = σ² is always positive by construction.
    """

    mu: Float[Array, "n_stimulus"]
    log_var: Float[Array, "n_stimulus"]  # log(σ²); σ = exp(0.5 · log_var)

    def __init__(self, n_stimulus: int) -> None:
        # Initialise at the prior N(0, I) — KL = 0 at start
        self.mu = jnp.zeros(n_stimulus)
        self.log_var = jnp.zeros(n_stimulus)

    def __call__(
        self,
        x: Float[Array, "n_stimulus"],
        temperature: float,
    ) -> Float[Array, "n_stimulus"]:
        sigma = jnp.exp(0.5 * self.log_var)
        return x * sigma * temperature + self.mu

    def kl_loss(self) -> Float[Array, ""]:
        """KL(N(μ, diag(σ²)) ‖ N(0, I)) = 0.5 · Σ(μ² + σ² − 1 − log σ²).

        Always ≥ 0; zero iff μ = 0 and σ = 1 everywhere.
        Prevents both bandwidth collapse (σ→0) and explosion (σ→∞).
        """
        return 0.5 * jnp.sum(self.mu ** 2 + jnp.exp(self.log_var) - 1.0 - self.log_var)


def _build_param_embeddings(
    param_layout: list[tuple[str, tuple[int, ...]]],
    target_layer_sizes: list[int],
    node_vec_dim: int,
    key: PRNGKeyArray,
) -> Float[Array, "n_params embed_dim"]:
    """Precompute frozen parameter embeddings for a target net.

    For each node (layer l, position j): node_embed = cat(random_vec, [sin(l)]).
    For weight W_l[i, j]: embed = cat(node_embed(l, j), node_embed(l+1, i)).
    For bias  b_l[i]:     embed = cat(node_embed(l+1, i), zeros(node_embed_dim)).

    Called once at construction; result is a frozen array (stop_gradient in forward).
    """
    node_embed_dim = node_vec_dim + 1

    total_nodes = sum(target_layer_sizes)
    node_vecs = jax.random.normal(key, (total_nodes, node_vec_dim))

    layer_offsets = [0]
    for s in target_layer_sizes[:-1]:
        layer_offsets.append(layer_offsets[-1] + s)

    def node_embed(l: int, j: int) -> Float[Array, "node_embed_dim"]:
        flat_idx = layer_offsets[l] + j
        return jnp.concatenate([node_vecs[flat_idx], jnp.array([jnp.sin(float(l))])])

    zero_node = jnp.zeros(node_embed_dim)

    all_embeds: list[Float[Array, "embed_dim"]] = []
    for name, shape in param_layout:
        l = int(name[1])
        if name.startswith("W"):
            out_size, in_size = shape
            for i in range(out_size):
                for j in range(in_size):
                    all_embeds.append(
                        jnp.concatenate([node_embed(l, j), node_embed(l + 1, i)])
                    )
        else:  # bias
            (out_size,) = shape
            for i in range(out_size):
                all_embeds.append(jnp.concatenate([node_embed(l + 1, i), zero_node]))

    return jnp.stack(all_embeds)


def _coord_net_forward(
    param_embed: Float[Array, "embed_dim"],
    cw: CoordWeights,
) -> Float[Array, ""]:
    x = jnp.tanh(cw.W1 @ param_embed + cw.b1)
    return (cw.W2 @ x + cw.b2).squeeze()


def _all_param_values(
    param_embeddings: Float[Array, "n_params embed_dim"],
    cw: CoordWeights,
) -> Float[Array, "n_params"]:
    return jax.vmap(_coord_net_forward, in_axes=(0, None))(param_embeddings, cw)


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


def _apply_target_net(
    params: dict[str, Float[Array, "..."]],
    x: Float[Array, "enc_dim"],
    n_layers: int,
) -> Float[Array, ""]:
    h = x
    for l in range(n_layers - 1):
        h = jnp.tanh(params[f"W{l}"] @ h + params[f"b{l}"])
    l = n_layers - 1
    return (params[f"W{l}"] @ h + params[f"b{l}"]).squeeze()


def _run_rnn_cell(
    params: dict[str, Float[Array, "..."]],
    x: Float[Array, "enc_dim"],
    num_steps: int,
    n_cell_layers: int,
) -> Float[Array, "enc_dim"]:
    """Run the generated RNN cell for num_steps steps; return full h_T.

    h_T[0] is the predicted mean (headless readout, same as standalone RNN convention).
    """
    def cell_step(h: Float[Array, "enc_dim"], _: None) -> tuple[Float[Array, "enc_dim"], None]:
        act = jnp.concatenate([h, x])
        for l in range(n_cell_layers):
            act = jnp.tanh(params[f"W{l}"] @ act + params[f"b{l}"])
        return act, None

    h_T, _ = jax.lax.scan(cell_step, x, None, length=num_steps)
    return h_T


def _stimulus_to_coord_weights(
    ffn: eqx.Module,
    stimulus: Float[Array, "n_stimulus"],
    embed_dim: int,
    coord_hidden: int,
) -> CoordWeights:
    flat = ffn(stimulus)
    i = 0
    W1 = flat[i : i + coord_hidden * embed_dim].reshape(coord_hidden, embed_dim)
    i += coord_hidden * embed_dim
    b1 = flat[i : i + coord_hidden]
    i += coord_hidden
    W2 = flat[i : i + coord_hidden].reshape(1, coord_hidden)
    i += coord_hidden
    b2 = flat[i : i + 1]
    return CoordWeights(W1=W1, b1=b1, W2=W2, b2=b2)


class HyperNetwork(eqx.Module):
    # Learnable parameters (pytree leaves)
    gaussian_encoder: GaussianEncoder
    stimulus_to_coord_params: eqx.nn.MLP
    param_embeddings: Float[Array, "n_params embed_dim"]  # frozen via stop_gradient

    # Static metadata (not pytree leaves)
    coord_net_hidden: int = eqx.field(static=True)
    coord_net_embed_dim: int = eqx.field(static=True)
    param_layout: list = eqx.field(static=True)
    n_target_layers: int = eqx.field(static=True)  # for FFN target; 0 if RNN
    target_is_rnn: bool = eqx.field(static=True)
    n_cell_layers: int = eqx.field(static=True)    # RNN cell depth; 0 if FFN
    rnn_num_steps: int = eqx.field(static=True)    # RNN recurrence steps; 0 if FFN
    temperature: float = eqx.field(static=True)

    def target_params(
        self,
        x_noise: Float[Array, "n_stimulus"],
    ) -> Float[Array, "n_params"]:
        """Generate the flat target network parameter vector."""
        stimulus = self.gaussian_encoder(x_noise, self.temperature)
        cw = _stimulus_to_coord_weights(
            self.stimulus_to_coord_params,
            stimulus,
            self.coord_net_embed_dim,
            self.coord_net_hidden,
        )
        frozen_embeds = jax.lax.stop_gradient(self.param_embeddings)
        return _all_param_values(frozen_embeds, cw)

    def __call__(
        self,
        c_enc: Float[Array, "enc_dim"],
        x_noise: Float[Array, "n_stimulus"],
    ) -> Float[Array, ""]:
        stimulus = self.gaussian_encoder(x_noise, self.temperature)
        cw = _stimulus_to_coord_weights(
            self.stimulus_to_coord_params,
            stimulus,
            self.coord_net_embed_dim,
            self.coord_net_hidden,
        )
        frozen_embeds = jax.lax.stop_gradient(self.param_embeddings)
        flat_params = _all_param_values(frozen_embeds, cw)
        params = _unflatten_target_params(flat_params, self.param_layout)
        if self.target_is_rnn:
            return _run_rnn_cell(params, c_enc, self.rnn_num_steps, self.n_cell_layers)[0]
        else:
            return _apply_target_net(params, c_enc, self.n_target_layers)


def make_hypernetwork(
    key: PRNGKeyArray,
    *,
    K: int = K_DEFAULT,
    n_stimulus: int = 32,
    node_vec_dim: int = 16,
    coord_net_hidden: int = 32,
    # FFN target params (used when target_is_rnn=False)
    target_hidden_dim: int = 32,
    n_target_hidden_layers: int = 2,
    # RNN target params (used when target_is_rnn=True)
    target_is_rnn: bool = True,
    rnn_hidden_dim: int = 128,
    rnn_depth: int = 2,
    rnn_num_steps: int = 10,
    # Shared
    temperature: float = 1.0,
    init_scale: float = 0.1,
    stim_ffn_hidden: int = 128,
    stim_ffn_depth: int = 2,
) -> HyperNetwork:
    d = enc_dim(K)

    if target_is_rnn:
        # Cell: [2*d] + [rnn_hidden_dim]*(rnn_depth-1) + [d]; headless (returns h_T[0])
        cell_dims = [2 * d] + [rnn_hidden_dim] * (rnn_depth - 1) + [d]
        param_layout: list[tuple[str, tuple[int, ...]]] = []
        for l, (in_sz, out_sz) in enumerate(zip(cell_dims, cell_dims[1:])):
            param_layout.append((f"W{l}", (out_sz, in_sz)))
            param_layout.append((f"b{l}", (out_sz,)))
        target_layer_sizes = cell_dims
        n_target_layers_val = 0
        n_cell_layers_val = rnn_depth
        rnn_num_steps_val = rnn_num_steps
    else:
        target_layers = [d] + [target_hidden_dim] * n_target_hidden_layers + [1]
        n_target_layers_val = len(target_layers) - 1
        param_layout = []
        for l, (in_sz, out_sz) in enumerate(zip(target_layers, target_layers[1:])):
            param_layout.append((f"W{l}", (out_sz, in_sz)))
            param_layout.append((f"b{l}", (out_sz,)))
        target_layer_sizes = target_layers
        n_cell_layers_val = 0
        rnn_num_steps_val = 0

    node_embed_dim = node_vec_dim + 1
    embed_dim = 2 * node_embed_dim
    n_coord_params = coord_net_hidden * embed_dim + coord_net_hidden + coord_net_hidden + 1

    k1, k2 = jax.random.split(key, 2)

    stim_ffn = eqx.nn.MLP(
        in_size=n_stimulus,
        out_size=n_coord_params,
        width_size=stim_ffn_hidden,
        depth=stim_ffn_depth,
        activation=jax.nn.tanh,
        key=k1,
    )
    # Scale output layer to keep coord net weights small at init
    stim_ffn = eqx.tree_at(
        lambda m: (m.layers[-1].weight, m.layers[-1].bias),
        stim_ffn,
        (stim_ffn.layers[-1].weight * init_scale, stim_ffn.layers[-1].bias * init_scale),
    )

    return HyperNetwork(
        gaussian_encoder=GaussianEncoder(n_stimulus),
        stimulus_to_coord_params=stim_ffn,
        param_embeddings=_build_param_embeddings(
            param_layout=param_layout,
            target_layer_sizes=target_layer_sizes,
            node_vec_dim=node_vec_dim,
            key=k2,
        ),
        coord_net_hidden=coord_net_hidden,
        coord_net_embed_dim=embed_dim,
        param_layout=param_layout,
        n_target_layers=n_target_layers_val,
        target_is_rnn=target_is_rnn,
        n_cell_layers=n_cell_layers_val,
        rnn_num_steps=rnn_num_steps_val,
        temperature=temperature,
    )


def n_trainable_params(model: HyperNetwork) -> int:
    """Count trainable parameters (excludes frozen param_embeddings)."""
    rest = (model.gaussian_encoder, model.stimulus_to_coord_params)
    return sum(x.size for x in jax.tree_util.tree_leaves(eqx.filter(rest, eqx.is_array)))

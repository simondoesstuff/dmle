import jax
import jax.numpy as jnp
import equinox as eqx
from god.datasets.mandelbrot import enc_dim, encode, K_DEFAULT
from god.models.rnn import RNNCell, MandelbrotRNN


ENC_DIM = enc_dim(K_DEFAULT)
HIDDEN = 32
DEPTH = 1  # 1 hidden tanh layer + linear output
NUM_STEPS = 5
KEY = jax.random.PRNGKey(0)


def test_rnn_cell_output_shape():
    cell = RNNCell(ENC_DIM, HIDDEN, DEPTH, KEY)
    h = jnp.zeros(ENC_DIM)
    x = jnp.zeros(ENC_DIM)
    out = cell(h, x)
    assert out.shape == (ENC_DIM,)


def test_rnn_cell_depth0_single_linear():
    """depth=0: single linear layer, no hidden."""
    cell = RNNCell(ENC_DIM, hidden_dim=999, depth=0, key=KEY)
    assert len(cell.layers) == 1
    assert cell.layers[0].in_features == 2 * ENC_DIM
    assert cell.layers[0].out_features == ENC_DIM


def test_rnn_cell_depth1_one_hidden():
    """depth=1: one tanh hidden layer then linear output = 2 layers total."""
    cell = RNNCell(ENC_DIM, HIDDEN, depth=1, key=KEY)
    assert len(cell.layers) == 2
    assert cell.layers[0].in_features == 2 * ENC_DIM
    assert cell.layers[0].out_features == HIDDEN
    assert cell.layers[1].in_features == HIDDEN
    assert cell.layers[1].out_features == ENC_DIM


def test_rnn_cell_depth2():
    """depth=2: two tanh hidden layers + linear output = 3 layers total."""
    cell = RNNCell(ENC_DIM, HIDDEN, depth=2, key=KEY)
    assert len(cell.layers) == 3
    assert cell.layers[0].in_features == 2 * ENC_DIM
    assert cell.layers[1].in_features == HIDDEN
    assert cell.layers[2].out_features == ENC_DIM


def test_rnn_cell_output_is_finite():
    cell = RNNCell(ENC_DIM, HIDDEN, DEPTH, KEY)
    h = jnp.ones(ENC_DIM) * 0.5
    x = jnp.ones(ENC_DIM) * -0.3
    out = cell(h, x)
    assert jnp.all(jnp.isfinite(out))


def test_mandelbrot_rnn_output_shape():
    model = MandelbrotRNN(ENC_DIM, HIDDEN, DEPTH, NUM_STEPS, KEY)
    c_enc = encode(jnp.array(0.3), jnp.array(-0.4))
    h_T = model(c_enc)
    assert h_T.shape == (ENC_DIM,)


def test_mandelbrot_rnn_batched_via_vmap():
    model = MandelbrotRNN(ENC_DIM, HIDDEN, DEPTH, NUM_STEPS, KEY)
    B = 8
    keys = jax.random.split(KEY, B)
    c_encs = jax.vmap(lambda k: encode(
        jax.random.uniform(k, minval=-2.0, maxval=2.0),
        jax.random.uniform(k, minval=-2.0, maxval=0.0),
    ))(keys)
    h_Ts = jax.vmap(model)(c_encs)
    assert h_Ts.shape == (B, ENC_DIM)


def test_predict_magnitude_direct_head():
    model = MandelbrotRNN(ENC_DIM, HIDDEN, DEPTH, NUM_STEPS, KEY)
    assert model.head is None
    h_T = model(encode(jnp.array(0.3), jnp.array(-0.4)))
    mag = model.predict_magnitude(h_T)
    assert mag.shape == ()
    assert float(mag) >= 0.0


def test_predict_magnitude_linear_head():
    model = MandelbrotRNN(ENC_DIM, HIDDEN, DEPTH, NUM_STEPS, KEY, linear_head=True)
    assert model.head is not None
    h_T = model(encode(jnp.array(0.3), jnp.array(-0.4)))
    mag = model.predict_magnitude(h_T)
    assert mag.shape == ()
    assert float(mag) >= 0.0


def test_gradients_flow():
    model = MandelbrotRNN(ENC_DIM, HIDDEN, DEPTH, NUM_STEPS, KEY)
    c_enc = encode(jnp.array(0.5), jnp.array(-0.5))
    target = jnp.array(0.4)

    def loss(m):
        h_T = m(c_enc)
        return (m.predict_magnitude(h_T) / 2.0 - target) ** 2

    grads = eqx.filter_grad(loss)(model)
    leaf_grads = jax.tree_util.tree_leaves(eqx.filter(grads, eqx.is_array))
    assert any(jnp.any(g != 0.0) for g in leaf_grads)


def test_gradients_flow_linear_head():
    model = MandelbrotRNN(ENC_DIM, HIDDEN, DEPTH, NUM_STEPS, KEY, linear_head=True)
    c_enc = encode(jnp.array(0.5), jnp.array(-0.5))
    target = jnp.array(0.4)

    def loss(m):
        h_T = m(c_enc)
        return (m.predict_magnitude(h_T) / 2.0 - target) ** 2

    grads = eqx.filter_grad(loss)(model)
    leaf_grads = jax.tree_util.tree_leaves(eqx.filter(grads, eqx.is_array))
    assert any(jnp.any(g != 0.0) for g in leaf_grads)

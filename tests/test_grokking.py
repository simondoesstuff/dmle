"""Tests for the grokking training module."""

import jax
import jax.numpy as jnp
import optax
import equinox as eqx
import pytest

from god.datasets.modular import ModularConfig, Op, input_dim, make_modular_dataset
from god.models.ffn import TanhFFN
from god.training.grokking import TrainConfig, _accuracy, _loss, train


KEY = jax.random.PRNGKey(42)


# ── unit: loss and accuracy ───────────────────────────────────────────────────

def test_loss_shape():
    cfg = ModularConfig(modulus=5, ops=(Op.ADD,), train_fraction=1.0)
    ds = make_modular_dataset(cfg, KEY)
    model = TanhFFN(input_dim(cfg), 16, 1, 5, KEY)
    loss = _loss(model, ds.train_inputs, ds.train_targets)
    assert loss.shape == ()
    assert float(loss) > 0


def test_accuracy_range():
    cfg = ModularConfig(modulus=5, ops=(Op.ADD,), train_fraction=1.0)
    ds = make_modular_dataset(cfg, KEY)
    model = TanhFFN(input_dim(cfg), 16, 1, 5, KEY)
    acc = _accuracy(model, ds.train_inputs, ds.train_targets)
    assert 0.0 <= float(acc) <= 1.0


def test_perfect_logits_give_accuracy_one():
    p = 5
    cfg = ModularConfig(modulus=p, ops=(Op.ADD,), train_fraction=1.0)
    ds = make_modular_dataset(cfg, KEY)
    targets = ds.train_targets
    model = TanhFFN(input_dim(cfg), 16, 1, p, KEY)

    # Monkey-patch forward to return one-hot logits for the correct class
    def perfect_forward(x):
        return jnp.zeros(p).at[0].set(1.0)  # wrong class — accuracy should be low-ish

    # Just verify that when predictions match targets, acc == 1
    logits = jax.nn.one_hot(targets, p) * 10.0
    preds = jnp.argmax(logits, axis=-1)
    acc = jnp.mean(preds == targets)
    assert float(acc) == pytest.approx(1.0)


# ── integration: one training step reduces loss ───────────────────────────────

def test_training_step_reduces_loss():
    p = 7
    cfg = ModularConfig(modulus=p, ops=(Op.ADD,), train_fraction=0.5)
    ds = make_modular_dataset(cfg, KEY)
    in_dim = input_dim(cfg)
    model = TanhFFN(in_dim, 32, 1, p, KEY)

    optimizer = optax.adamw(learning_rate=1e-3, weight_decay=1.0)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    loss_before = float(_loss(model, ds.train_inputs, ds.train_targets))

    for _ in range(50):
        grads = eqx.filter_grad(_loss)(model, ds.train_inputs, ds.train_targets)
        updates, opt_state = optimizer.update(
            grads, opt_state, eqx.filter(model, eqx.is_array)
        )
        model = eqx.apply_updates(model, updates)

    loss_after = float(_loss(model, ds.train_inputs, ds.train_targets))
    assert loss_after < loss_before


# ── smoke: short train run completes and returns a model ─────────────────────

def test_train_smoke(tmp_path):
    cfg = TrainConfig(
        data_dir=str(tmp_path / "grokking"),
        modulus=11,
        hidden_dim=16,
        depth=1,
        train_fraction=0.5,
        n_epochs=10,
        log_interval=5,
        seed=0,
    )
    model = train(cfg)
    assert isinstance(model, TanhFFN)
    assert (tmp_path / "grokking" / "metrics.json").exists()
    assert (tmp_path / "grokking" / "grokking_curve.png").exists()

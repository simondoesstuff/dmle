"""Tests for the modular arithmetic dataset."""

import jax
import numpy as np
import pytest

from god.datasets.modular import ModularConfig, Op, input_dim, make_modular_dataset


KEY = jax.random.PRNGKey(0)


# ── input_dim ─────────────────────────────────────────────────────────────────

def test_input_dim_all_ops():
    config = ModularConfig(modulus=7, ops=(Op.ADD, Op.SUB, Op.MULT, Op.DIV))
    assert input_dim(config) == 2 * 7 + 4


def test_input_dim_single_op():
    config = ModularConfig(modulus=11, ops=(Op.ADD,))
    assert input_dim(config) == 2 * 11 + 1


def test_input_dim_two_ops():
    config = ModularConfig(modulus=11, ops=(Op.ADD, Op.SUB))
    assert input_dim(config) == 2 * 11 + 2


# ── dataset shapes ────────────────────────────────────────────────────────────

def test_dataset_shapes_add():
    config = ModularConfig(modulus=5, ops=(Op.ADD,), train_fraction=0.5)
    ds = make_modular_dataset(config, KEY)
    assert ds.train_inputs.shape[1] == input_dim(config)
    assert ds.train_targets.shape[0] == ds.train_inputs.shape[0]
    assert ds.test_targets.shape[0] == ds.test_inputs.shape[0]
    total = ds.train_inputs.shape[0] + ds.test_inputs.shape[0]
    assert total == 25  # 5x5 pairs


def test_train_fraction():
    config = ModularConfig(modulus=5, ops=(Op.ADD,), train_fraction=0.4)
    ds = make_modular_dataset(config, KEY)
    # int(25 * 0.4) = 10 train, 15 test
    assert ds.train_inputs.shape[0] == 10
    assert ds.test_inputs.shape[0] == 15


def test_all_in_train():
    config = ModularConfig(modulus=5, ops=(Op.ADD,), train_fraction=1.0)
    ds = make_modular_dataset(config, KEY)
    assert ds.train_inputs.shape[0] == 25
    assert ds.test_inputs.shape[0] == 0


# ── arithmetic correctness ────────────────────────────────────────────────────

def _find_pair(inp: np.ndarray, tgt: np.ndarray, a: int, b: int, p: int) -> int | None:
    """Return the target for (a, b) in a dataset with single op, or None."""
    for i in range(len(inp)):
        if inp[i, a] == 1.0 and inp[i, p + b] == 1.0:
            return int(tgt[i])
    return None


def test_add_correctness():
    p = 7
    config = ModularConfig(modulus=p, ops=(Op.ADD,), train_fraction=1.0)
    ds = make_modular_dataset(config, KEY)
    inp = np.array(ds.train_inputs)
    tgt = np.array(ds.train_targets)
    for a in range(p):
        for b in range(p):
            result = _find_pair(inp, tgt, a, b, p)
            assert result is not None
            assert result == (a + b) % p


def test_sub_correctness():
    p = 5
    config = ModularConfig(modulus=p, ops=(Op.SUB,), train_fraction=1.0)
    ds = make_modular_dataset(config, KEY)
    inp = np.array(ds.train_inputs)
    tgt = np.array(ds.train_targets)
    for a in range(p):
        for b in range(p):
            result = _find_pair(inp, tgt, a, b, p)
            assert result is not None
            assert result == (a - b) % p


def test_mult_correctness():
    p = 5
    config = ModularConfig(modulus=p, ops=(Op.MULT,), train_fraction=1.0)
    ds = make_modular_dataset(config, KEY)
    inp = np.array(ds.train_inputs)
    tgt = np.array(ds.train_targets)
    for a in range(p):
        for b in range(p):
            result = _find_pair(inp, tgt, a, b, p)
            assert result is not None
            assert result == (a * b) % p


def test_div_correctness():
    p = 7
    config = ModularConfig(modulus=p, ops=(Op.DIV,), train_fraction=1.0)
    ds = make_modular_dataset(config, KEY)
    inp = np.array(ds.train_inputs)
    tgt = np.array(ds.train_targets)
    for a in range(p):
        for b in range(1, p):  # b != 0
            result = _find_pair(inp, tgt, a, b, p)
            assert result is not None
            assert result == (a * pow(b, -1, p)) % p


# ── division edge cases ───────────────────────────────────────────────────────

def test_div_excludes_b0():
    p = 5
    config = ModularConfig(modulus=p, ops=(Op.DIV,), train_fraction=0.0)
    ds = make_modular_dataset(config, KEY)
    # DIV: p*(p-1) = 5*4 = 20 valid pairs (b=0 excluded)
    total = ds.train_inputs.shape[0] + ds.test_inputs.shape[0]
    assert total == 20


def test_div_b0_not_in_dataset():
    p = 7
    config = ModularConfig(modulus=p, ops=(Op.DIV,), train_fraction=0.5)
    ds = make_modular_dataset(config, KEY)
    # No row should have one_hot(b=0) set in the B position
    for split in (ds.train_inputs, ds.test_inputs):
        inp = np.array(split)
        for i in range(len(inp)):
            assert inp[i, p + 0] == 0.0, f"Row {i} has b=0 in DIV dataset"


# ── configurable modulus and ops ──────────────────────────────────────────────

def test_configurable_modulus():
    config = ModularConfig(modulus=11, ops=(Op.ADD, Op.MULT), train_fraction=0.5)
    ds = make_modular_dataset(config, KEY)
    assert ds.train_inputs.shape[1] == input_dim(config)
    total = ds.train_inputs.shape[0] + ds.test_inputs.shape[0]
    assert total == 11 * 11 * 2  # 242


def test_single_op_mult():
    p = 7
    config = ModularConfig(modulus=p, ops=(Op.MULT,), train_fraction=0.3)
    ds = make_modular_dataset(config, KEY)
    assert input_dim(config) == 2 * p + 1
    total = ds.train_inputs.shape[0] + ds.test_inputs.shape[0]
    assert total == p * p


def test_default_config():
    config = ModularConfig()
    ds = make_modular_dataset(config, KEY)
    assert ds.train_inputs.shape[1] == input_dim(config)
    assert input_dim(config) == 2 * 97 + 4
    # ADD + SUB + MULT: 97*97*3, DIV: 97*96 — total
    expected_total = 97 * 97 * 3 + 97 * 96
    total = ds.train_inputs.shape[0] + ds.test_inputs.shape[0]
    assert total == expected_total

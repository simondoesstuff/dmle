"""Modular arithmetic dataset from the Grokking paper.

Generates all valid (A op B) mod p triples for configurable ops and modulus.
Inputs are one-hot encoded: [one_hot(A, p) | one_hot(B, p) | one_hot(op, n_ops)].
Targets are integer class labels in {0, ..., p-1} (suitable for cross-entropy).

Division: computes A * B^{-1} mod p; skips B=0 (and any B without an inverse
if p is not prime). Use a prime modulus for fully-defined division.

Reference: Power et al. (2022), "Grokking: Generalization Beyond Overfitting
on Small Algorithmic Datasets".
"""

import enum
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import PRNGKeyArray

from god.datasets.base import Dataset


class Op(enum.Enum):
    ADD = "add"
    SUB = "sub"
    MULT = "mult"
    DIV = "div"


@dataclass(frozen=True)
class ModularConfig:
    modulus: int = 97
    ops: tuple[Op, ...] = (Op.ADD, Op.SUB, Op.MULT, Op.DIV)
    train_fraction: float = 0.3


def input_dim(config: ModularConfig) -> int:
    """Dimension of the one-hot encoded input vector."""
    return 2 * config.modulus + len(config.ops)


def _apply_op(a: int, b: int, op: Op, p: int) -> int | None:
    if op == Op.ADD:
        return (a + b) % p
    elif op == Op.SUB:
        return (a - b) % p
    elif op == Op.MULT:
        return (a * b) % p
    elif op == Op.DIV:
        if b == 0:
            return None
        try:
            return (a * pow(b, -1, p)) % p
        except ValueError:
            return None


def make_modular_dataset(config: ModularConfig, key: PRNGKeyArray) -> Dataset:
    """Enumerate all valid (A op B mod p) pairs and split into train/test.

    Args:
        config: Dataset configuration (modulus, ops, train_fraction).
        key: JAX PRNG key for the random train/test split.

    Returns:
        Dataset with integer targets in {0, ..., p-1}.
    """
    p = config.modulus
    n_ops = len(config.ops)
    op_list = list(config.ops)
    dim = 2 * p + n_ops

    inputs_list: list[np.ndarray] = []
    targets_list: list[int] = []

    for op_idx, op in enumerate(op_list):
        for a in range(p):
            for b in range(p):
                result = _apply_op(a, b, op, p)
                if result is None:
                    continue
                enc = np.zeros(dim, dtype=np.float32)
                enc[a] = 1.0
                enc[p + b] = 1.0
                enc[2 * p + op_idx] = 1.0
                inputs_list.append(enc)
                targets_list.append(result)

    all_inputs = jnp.array(np.stack(inputs_list))
    all_targets = jnp.array(np.array(targets_list, dtype=np.int32))

    N = len(targets_list)
    n_train = int(N * config.train_fraction)
    perm = jax.random.permutation(key, N)
    train_idx = perm[:n_train]
    test_idx = perm[n_train:]

    return Dataset(
        train_inputs=all_inputs[train_idx],
        train_targets=all_targets[train_idx],
        test_inputs=all_inputs[test_idx],
        test_targets=all_targets[test_idx],
    )

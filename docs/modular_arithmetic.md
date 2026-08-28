# Modular Arithmetic Dataset

Algorithmic dataset from the Grokking paper (Power et al., 2022) for studying generalisation dynamics in neural networks.

## Task

Given integers A and B drawn from `{0, …, p-1}`, predict `(A op B) mod p` for one of four operations. This is a classification task with `p` classes.

## Generating the Dataset

All valid `(A, op, B)` triples are enumerated exhaustively, then split into train/test by a random permutation.

```python
from god.datasets.modular import ModularConfig, Op, make_modular_dataset
import jax

config = ModularConfig(modulus=97, ops=(Op.ADD, Op.SUB, Op.MULT, Op.DIV), train_fraction=0.3)
dataset = make_modular_dataset(config, jax.random.PRNGKey(0))
# dataset.train_inputs  (n_train, 2*p + n_ops)
# dataset.train_targets (n_train,)  — integer labels in {0, …, p-1}
# dataset.test_inputs   (n_test, 2*p + n_ops)
# dataset.test_targets  (n_test,)
```

## Input Encoding

One-hot concatenation:

```
input = [one_hot(A, p) | one_hot(B, p) | one_hot(op_idx, n_ops)]
```

Dimension: `2 * p + len(ops)` — e.g. 198 for p=97 with all four ops.

```python
from god.datasets.modular import input_dim
print(input_dim(config))  # 198
```

## Operations

| Op | Formula | Valid pairs |
|----|---------|-------------|
| `ADD` | `(A + B) mod p` | p² |
| `SUB` | `(A - B) mod p` | p² |
| `MULT` | `(A × B) mod p` | p² |
| `DIV` | `A × B⁻¹ mod p` | p × (p-1) — B=0 excluded |

For **division**, `B⁻¹ mod p` is the modular inverse. Use a **prime modulus** to guarantee every non-zero B has a unique inverse. Non-invertible B values (when p is composite) are silently skipped.

## Configuration

```python
@dataclass(frozen=True)
class ModularConfig:
    modulus: int = 97              # p — use a prime for well-defined division
    ops: tuple[Op, ...] = (Op.ADD, Op.SUB, Op.MULT, Op.DIV)
    train_fraction: float = 0.3   # fraction of all pairs used for training
```

Common variants:

```python
# Single operation
config = ModularConfig(modulus=97, ops=(Op.ADD,))

# Small modulus for debugging
config = ModularConfig(modulus=7, ops=(Op.ADD, Op.MULT), train_fraction=0.5)

# Grokking paper default (no division)
config = ModularConfig(modulus=97, ops=(Op.ADD, Op.SUB, Op.MULT), train_fraction=0.3)
```

## Train/Test Split

The split is a random permutation of all valid pairs (`train_fraction` of them go to train). This matches the Grokking paper setup where random subsets are used rather than structural splits.

## Reference

Power, A., Mosconi, Y., Misra, V., et al. (2022). *Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets*. ICLR 2022 workshop.

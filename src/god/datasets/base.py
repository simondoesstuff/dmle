from dataclasses import dataclass

from jaxtyping import Array


@dataclass
class Dataset:
    train_inputs: Array   # (n_train, input_dim)
    train_targets: Array  # (n_train,)
    test_inputs: Array    # (n_test, input_dim)
    test_targets: Array   # (n_test,)

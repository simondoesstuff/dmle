"""Run the best-performing HNN config with frequent checkpointing for dynamics analysis.

Config: lam=1e-2, smin=0.01, smax=1.0, n_mc=32 — reached 73% test at 100k and climbing.
"""
from god.training.grokking_hnn import GrokkingHNNConfig, train

train(GrokkingHNNConfig(
    data_dir="data/hnn_best",
    lambda_complexity=1e-2,
    softmin_temp=0.01,
    softmax_temp=1.0,
    n_mc=32,
    n_epochs=200_000,
    checkpoint_interval=2000,
    log_interval=2000,
    train_fraction=0.3,
    seed=0,
))

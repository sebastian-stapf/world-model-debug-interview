"""Train/validation partitions and minibatches."""

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from .dataset import TransitionDataset


def split_indices(dataset: TransitionDataset, validation_fraction: float, seed: int):
    """Return training and validation indices; validate on unseen episodes."""
    if not 0 < validation_fraction < 1 or len(dataset.frames) < 2:
        raise ValueError("Need at least two episodes and a validation fraction in (0,1)")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(dataset))
    n_validation = max(1, min(len(order) - 1, round(len(order) * validation_fraction)))
    validation = order[:n_validation].tolist()
    training = order[n_validation:].tolist()
    return training, validation


def make_loaders(dataset: TransitionDataset, config: dict):
    training, validation = split_indices(dataset, config["validation_fraction"], config["seed"])
    common = {"batch_size": config["batch_size"], "num_workers": config["num_workers"]}
    train = DataLoader(
        Subset(dataset, training), shuffle=True,
        generator=torch.Generator().manual_seed(config["seed"]), **common,
    )
    val = DataLoader(Subset(dataset, validation), shuffle=False, **common)
    return train, val

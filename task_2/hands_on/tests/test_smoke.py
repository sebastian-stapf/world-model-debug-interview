"""Public smoke tests only. Add behavioural tests as part of your investigation."""

import numpy as np
import torch

from world_model.dataset import TransitionDataset
from world_model.environment import collect
from world_model.model import WorldModel
from world_model.sampling import sample_next
from world_model.trainer import loss_for_batch


def test_data_shapes_and_normalization(tmp_path):
    path = tmp_path / "episodes.npz"
    np.savez_compressed(path, **collect(4, 8, 42))
    dataset = TransitionDataset(path)
    sample = dataset[0]
    assert len(dataset) == 28
    assert sample["context"].shape == (2, 1, 24, 24)
    assert sample["target"].shape == (1, 24, 24)
    assert -1 <= sample["context"].min() <= sample["context"].max() <= 1
    assert sample["action"].dtype == torch.int64


def test_forward_backward_and_sampling_are_finite():
    torch.set_num_threads(2)
    model = WorldModel(width=8)
    batch = {"context": torch.randn(3, 2, 1, 24, 24), "target": torch.randn(3, 1, 24, 24), "action": torch.tensor([0, 1, 2])}
    loss = loss_for_batch(model, batch)
    loss.backward()
    assert torch.isfinite(loss)
    assert model.out.weight.grad is not None
    output = sample_next(model.eval(), batch["context"], batch["action"], steps=2)
    assert output.shape == batch["target"].shape
    assert torch.isfinite(output).all()


def test_environment_is_repeatable():
    first, second = collect(2, 8, 11), collect(2, 8, 11)
    assert np.array_equal(first["frames"], second["frames"])
    assert np.array_equal(first["actions"], second["actions"])

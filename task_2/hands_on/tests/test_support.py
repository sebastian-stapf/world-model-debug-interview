"""Checks for the sampler and checkpoint infrastructure outside the exercise."""

import numpy as np
import torch

from world_model.environment import collect
from world_model.model import WorldModel
from world_model.sampling import sample_next
from world_model.utils import load_checkpoint, provenance


def test_euler_integrates_a_known_constant_field():
    torch.manual_seed(1)
    context = torch.zeros(2, 2, 1, 24, 24)
    noise = torch.randn(2, 1, 24, 24)
    destination = torch.rand_like(noise) * 2 - 1

    def field(x, time, observed, action):
        return destination - noise

    generated = sample_next(field, context, torch.zeros(2, dtype=torch.long), steps=32, initial_noise=noise)
    torch.testing.assert_close(generated, destination, atol=5e-6, rtol=5e-6)


def test_checkpoint_round_trip_with_safe_loading(tmp_path):
    archive = tmp_path / "data.npz"
    np.savez_compressed(archive, **collect(2, 4, 3))
    model = WorldModel(8)
    config = {"width": 8}
    path = tmp_path / "checkpoint.pt"
    torch.save({"state_dict": model.state_dict(), "config": config, "provenance": provenance(archive, config, "cpu")}, path)
    restored, restored_config = load_checkpoint(path, "cpu")
    assert restored_config == config
    for name, value in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], value)

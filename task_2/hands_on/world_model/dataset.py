"""Transition windows for next-frame modelling."""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class TransitionDataset(Dataset):
    """Expose (two-frame context ending at t, action at t, frame at t+1).

    Archives contain frames[e, t] and actions[e, t]. An action is issued after
    observing frames[e, t] and produces frames[e, t+1]. Episodes never cross.
    """

    def __init__(self, path: str | Path):
        with np.load(path, allow_pickle=False) as archive:
            self.frames = archive["frames"].copy()
            self.actions = archive["actions"].copy()
        if self.frames.ndim != 5 or self.frames.shape[2:] != (1, 24, 24):
            raise ValueError("Expected frames [episodes, T+1, 1, 24, 24]")
        if self.actions.shape != (len(self.frames), self.frames.shape[1] - 1):
            raise ValueError("Expected one action per transition")
        if self.frames.dtype != np.uint8 or not np.isin(self.actions, [0, 1, 2]).all():
            raise ValueError("Expected uint8 frames and actions in {0,1,2}")
        self.keys = [(e, t) for e in range(len(self.frames)) for t in range(1, len(self.actions[e]))]

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        episode, t = self.keys[index]
        context = torch.from_numpy(self.frames[episode, t - 1:t + 1].copy()).float() / 127.5 - 1
        target = torch.from_numpy(self.frames[episode, t + 1].copy()).float() / 127.5 - 1
        action = int(self.actions[episode, t - 1])
        return {
            "context": context,
            "target": target,
            "action": torch.tensor(action, dtype=torch.long),
            "episode": torch.tensor(episode),
            "time": torch.tensor(t),
        }

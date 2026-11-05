"""Evaluate long-horizon predictions from a fixed initial context."""

import argparse
import json
from pathlib import Path

import torch

from .dataset import TransitionDataset
from .sampling import sample_next
from .utils import load_checkpoint, seed_everything


def image_mse(prediction, target):
    """Mean squared error in display space [0,1], per batch item and time."""
    return ((prediction - target) / 2).square().mean(dim=(-3, -2, -1))


@torch.no_grad()
def rollout(model, context, actions, future, steps=32, seed=0):
    """Measure open-loop predictions; future [B,H,1,24,24] supplies metric targets."""
    generator = torch.Generator(device=context.device).manual_seed(seed)
    predictions = []
    for h in range(actions.shape[1]):
        prediction = sample_next(model, context, actions[:, h], steps, generator)
        predictions.append(prediction)
        context = torch.cat([context[:, 1:], future[:, h:h + 1]], dim=1)
    return torch.stack(predictions, dim=1)


def episode_batch(dataset, episodes, horizon, device="cpu", start=1):
    if horizon < 1 or start < 1 or start + horizon >= dataset.frames.shape[1]:
        raise ValueError("Requested trajectory exceeds episode bounds")
    context = torch.from_numpy(dataset.frames[episodes, start - 1:start + 1].copy()).float() / 127.5 - 1
    future = torch.from_numpy(dataset.frames[episodes, start + 1:start + horizon + 1].copy()).float() / 127.5 - 1
    actions = torch.from_numpy(dataset.actions[episodes, start:start + horizon].copy()).long()
    return context.to(device), actions.to(device), future.to(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("data/heldout_pong.npz"))
    parser.add_argument("--output", type=Path, default=Path("outputs/eval.json"))
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    seed_everything(17)
    torch.set_num_threads(4)
    model, config = load_checkpoint(args.checkpoint, args.device)
    dataset = TransitionDataset(args.data)
    context, actions, future = episode_batch(dataset, list(range(min(32, len(dataset.frames)))), args.horizon, args.device)
    prediction = rollout(model, context, actions, future, config["ode_steps"], seed=23)
    metrics = {"open_loop_mse_by_horizon": image_mse(prediction, future).mean(0).tolist()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

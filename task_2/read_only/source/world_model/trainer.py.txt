"""Conditional flow matching on straight paths between noise and future frames."""

import argparse
import json
from pathlib import Path
import time

import torch
from torch.nn import functional as F

from .dataloader import make_loaders
from .dataset import TransitionDataset
from .model import WorldModel
from .utils import load_config, provenance, seed_everything, to_device


def flow_matching_batch(target, generator=None):
    noise = torch.randn(target.shape, device=target.device, generator=generator)
    time_values = torch.rand(len(target), device=target.device, generator=generator)
    t = time_values[:, None, None, None]
    x_t = (1 - t) * noise + t * target
    velocity = target
    return x_t, time_values, velocity


def loss_for_batch(model, batch, generator=None):
    x_t, time_values, velocity = flow_matching_batch(batch["target"], generator)
    prediction = model(x_t, time_values, batch["context"], batch["action"])
    return F.mse_loss(prediction, velocity)


@torch.no_grad()
def validation_loss(model, loader, device):
    model.eval()
    generator = torch.Generator(device=device).manual_seed(12345)
    total, count = 0.0, 0
    for batch in loader:
        batch = to_device(batch, device)
        n = len(batch["target"])
        total += float(loss_for_batch(model, batch, generator)) * n
        count += n
    model.train()
    return total / count


def train(data: Path, output: Path, config: dict, device: str):
    seed_everything(config["seed"])
    torch.set_num_threads(4)
    dataset = TransitionDataset(data)
    train_loader, val_loader = make_loaders(dataset, config)
    model = WorldModel(config["width"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=1e-4)
    output.mkdir(parents=True, exist_ok=True)
    metadata = provenance(data, config, device)
    (output / "provenance.json").write_text(json.dumps(metadata, indent=2) + "\n")
    iterator = iter(train_loader)
    records, recent = [], []
    started = time.perf_counter()
    model.train()
    for step in range(1, config["steps"] + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            batch = next(iterator)
        batch = to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        loss = loss_for_batch(model, batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        recent.append(float(loss.detach()))
        if step == 1 or step % config["log_every"] == 0 or step == config["steps"]:
            record = {"step": step, "train_loss": sum(recent) / len(recent), "elapsed_seconds": time.perf_counter() - started}
            recent.clear()
            if step == 1 or step % config["eval_every"] == 0 or step == config["steps"]:
                record["validation_loss"] = validation_loss(model, val_loader, device)
            records.append(record)
            print(json.dumps(record), flush=True)
            (output / "history.json").write_text(json.dumps(records, indent=2) + "\n")
    torch.save({"state_dict": model.state_dict(), "config": config, "provenance": metadata}, output / "model.pt")
    summary = {"seconds": time.perf_counter() - started, "steps": config["steps"], "parameters": sum(p.numel() for p in model.parameters())}
    if device.startswith("cuda"):
        summary["peak_gpu_memory_bytes"] = torch.cuda.max_memory_allocated()
    (output / "training_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/tiny_pong.npz"))
    parser.add_argument("--output", type=Path, default=Path("outputs/run"))
    parser.add_argument("--config", type=Path, default=Path("configs/default.json"))
    parser.add_argument("--steps", type=int)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.steps is not None:
        config["steps"] = args.steps
    train(args.data, args.output, config, args.device)


if __name__ == "__main__":
    main()

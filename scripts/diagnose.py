"""Produce a reproducible observation pack; no source edits or repairs."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
import torch

from world_model.dataloader import split_indices
from world_model.dataset import TransitionDataset
from world_model.environment import ACTION_NAMES
from world_model.eval import episode_batch, image_mse, rollout
from world_model.sampling import generate
from world_model.utils import load_checkpoint, seed_everything


plt.rcParams.update({
    "figure.facecolor": "#f5f7fb", "axes.facecolor": "#ffffff", "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 11, "axes.labelcolor": "#24324a",
    "text.color": "#17243b", "axes.titleweight": "bold", "savefig.facecolor": "#f5f7fb",
})


def tile(frame, scale=6):
    array = ((frame.detach().cpu().squeeze().numpy() + 1) * 127.5).round().clip(0, 255).astype(np.uint8)
    return Image.fromarray(array).convert("RGB").resize((24 * scale, 24 * scale), Image.Resampling.NEAREST)


def paddle_y(frame):
    return float(torch.where(frame[0, :, 1] > 0)[0].float().mean())


def data_panel(dataset, output):
    # The first six action changes give a more varied inspection sheet than a
    # run of identical commands. Selection uses the raw log, not model outputs.
    selected = [i for i, (e, t) in enumerate(dataset.keys) if dataset.actions[e, t] != dataset.actions[e, t - 1]][:6]
    canvas = Image.new("RGB", (6 * 156 + 20, 378), "#f5f7fb")
    draw = ImageDraw.Draw(canvas)
    draw.text((16, 12), "Loaded transitions: latest context (top) -> next frame (bottom)", fill="#17243b")
    for col, idx in enumerate(selected):
        sample = dataset[idx]
        x = 12 + col * 156
        draw.text((x, 36), f"e={int(sample['episode'])}, t={int(sample['time'])}", fill="#17243b")
        canvas.paste(tile(sample["context"][-1]), (x, 58))
        draw.text((x, 209), "action: " + ACTION_NAMES[int(sample["action"])], fill="#17243b")
        canvas.paste(tile(sample["target"]), (x, 232))
    canvas.save(output / "transitions.png")
    mismatch = 0
    for i in range(len(dataset)):
        sample = dataset[i]
        expected = np.clip(paddle_y(sample["context"][-1]) + (0, -2, 2)[int(sample["action"])], 4, 19)
        mismatch += abs(paddle_y(sample["target"]) - expected) > 0.01
    return {"checked_transitions": len(dataset), "paddle_command_disagreement_fraction": mismatch / len(dataset)}


def prediction_panels(context, future, evaluated, generated, output):
    horizon = future.shape[1]
    display_steps = sorted({h for h in [0, 1, 3, 5, 7, horizon - 1] if h < horizon})
    rows = [("Recorded future", future), ("Evaluation path", evaluated), ("Generation path", generated)]
    canvas = Image.new("RGB", (116 + len(display_steps) * 150, 40 + 3 * 156), "#f5f7fb")
    draw = ImageDraw.Draw(canvas)
    for col, h in enumerate(display_steps):
        draw.text((124 + col * 150, 12), f"horizon {h + 1}", fill="#17243b")
    for row, (label, sequence) in enumerate(rows):
        draw.text((8, 72 + row * 156), label, fill="#17243b")
        for col, h in enumerate(display_steps):
            canvas.paste(tile(sequence[0, h]), (116 + col * 150, 36 + row * 156))
    canvas.save(output / "generations.png")
    animation = []
    for h in range(horizon):
        frame = Image.new("RGB", (620, 200), "#f5f7fb")
        draw = ImageDraw.Draw(frame)
        for col, (label, sequence) in enumerate(rows):
            draw.text((12 + col * 204, 8), label, fill="#17243b")
            frame.paste(tile(sequence[0, h]), (30 + col * 204, 36))
        draw.text((12, 184), f"horizon {h + 1}", fill="#17243b")
        animation.append(frame)
    animation[0].save(output / "rollout.gif", save_all=True, append_images=animation[1:], duration=250, loop=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("data/tiny_pong.npz"))
    parser.add_argument("--heldout", type=Path, default=Path("data/heldout_pong.npz"))
    parser.add_argument("--output", type=Path, default=Path("assets"))
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    seed_everything(17)
    torch.set_num_threads(4)
    args.output.mkdir(parents=True, exist_ok=True)
    model, config = load_checkpoint(args.checkpoint, args.device)
    dataset = TransitionDataset(args.data)
    heldout = TransitionDataset(args.heldout)
    training, validation = split_indices(dataset, config["validation_fraction"], config["seed"])
    train_episodes = {dataset.keys[i][0] for i in training}
    val_episodes = {dataset.keys[i][0] for i in validation}
    metrics = {"split": {
        "train_windows": len(training), "validation_windows": len(validation),
        "train_episodes": len(train_episodes), "validation_episodes": len(val_episodes),
        "shared_episode_ids": len(train_episodes & val_episodes),
    }}
    metrics["data_probe"] = data_panel(dataset, args.output)
    context, actions, future = episode_batch(heldout, list(range(min(32, len(heldout.frames)))), config["horizon"], args.device)
    with torch.no_grad():
        evaluated = rollout(model, context, actions, future, config["ode_steps"], seed=23)
        generated = generate(model, context, actions, config["ode_steps"], seed=23)
        metrics["evaluation_path_mse"] = image_mse(evaluated, future).mean(0).tolist()
        metrics["generation_path_mse"] = image_mse(generated, future).mean(0).tolist()
        persistence = context[:, -1:].expand_as(future)
        metrics["persistence_mse"] = image_mse(persistence, future).mean(0).tolist()
        # Identical image/time/noise, three different actions in the same batch.
        probe_context = context[:1].repeat(3, 1, 1, 1, 1)
        probe_x = torch.randn_like(context[:1, -1]).repeat(3, 1, 1, 1)
        probe_t = torch.full((3,), 0.5, device=args.device)
        probe_actions = torch.arange(3, device=args.device)
        velocity = model(probe_x, probe_t, probe_context, probe_actions)
        other_actions = probe_actions.clone()
        other_actions[1:] = 0
        changed = model(probe_x, probe_t, probe_context, other_actions)
        metrics["action_probe"] = {
            "max_velocity_difference_across_actions": float((velocity - velocity[:1]).abs().max()),
            "same_item_delta_when_other_actions_change": float((changed[0] - velocity[0]).abs().max()),
        }
    prediction_panels(context, future, evaluated, generated, args.output)
    history = json.loads((args.checkpoint.parent / "history.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), constrained_layout=True)
    axes[0].plot([r["step"] for r in history], [r["train_loss"] for r in history], color="#2563eb", label="Training")
    val = [r for r in history if "validation_loss" in r]
    axes[0].plot([r["step"] for r in val], [r["validation_loss"] for r in val], "o-", color="#d05b27", label="Validation")
    axes[0].set(title="Logged regression objective", xlabel="Optimizer step", ylabel="Mean squared loss", yscale="log")
    axes[0].legend()
    hs = range(1, config["horizon"] + 1)
    for key, label, color in [("evaluation_path_mse", "Evaluation path", "#2563eb"), ("generation_path_mse", "Generation path", "#d05b27"), ("persistence_mse", "Repeat last observed frame", "#65758b")]:
        axes[1].plot(hs, metrics[key], "o-", label=label, color=color, markersize=4)
    axes[1].set(title="Same held-out episodes and noise seeds", xlabel="Prediction horizon", ylabel="Pixel MSE in [0,1]")
    axes[1].legend(fontsize=9)
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.savefig(args.output / "diagnostics.png", dpi=170)
    plt.close(fig)
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    for name in ["history.json", "provenance.json", "training_summary.json"]:
        (args.output / name).write_bytes((args.checkpoint.parent / name).read_bytes())
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

"""Generate reproducible episodes without ROMs or external datasets."""

import argparse
from pathlib import Path

import numpy as np

from world_model.environment import collect


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/tiny_pong.npz"))
    parser.add_argument("--episodes", type=int, default=256)
    parser.add_argument("--transitions", type=int, default=48)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **collect(args.episodes, args.transitions, args.seed))
    print(f"Saved {args.episodes} episodes to {args.output}")


if __name__ == "__main__":
    main()

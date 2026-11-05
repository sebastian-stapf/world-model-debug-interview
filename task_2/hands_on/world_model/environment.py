"""A deterministic, ROM-free Pong-like environment. Not an Atari emulator."""

from dataclasses import dataclass

import numpy as np


ACTION_NAMES = ("STAY", "UP", "DOWN")
SIZE = 24
PADDLE_HALF = 3
PADDLE_STEP = 2


@dataclass
class State:
    x: int
    y: int
    vx: int
    vy: int
    paddle: int


class TinyPong:
    """Ball bounces in a closed court; the left paddle is controlled by actions.

    Paddle contact reverses vertical ball velocity, so actions affect both the
    paddle and subsequent ball motion. No score, resets or hidden random events
    occur within an episode. Two consecutive frames reveal the motion direction.
    """

    def __init__(self, seed: int):
        self.rng = np.random.default_rng(seed)
        self.state = State(12, 12, 1, 1, 12)

    def reset(self) -> np.ndarray:
        self.state = State(
            x=int(self.rng.integers(5, 19)),
            y=int(self.rng.integers(3, 20)),
            vx=int(self.rng.choice([-1, 1])),
            vy=int(self.rng.choice([-1, 1])),
            paddle=int(self.rng.integers(4, 20)),
        )
        return self.render()

    def step(self, action: int) -> np.ndarray:
        if action not in (0, 1, 2):
            raise ValueError("Action must be 0 (stay), 1 (up), or 2 (down)")
        s = self.state
        s.paddle = int(np.clip(s.paddle + (0, -PADDLE_STEP, PADDLE_STEP)[action], 4, 19))
        if s.y + s.vy < 2 or s.y + s.vy > 20:
            s.vy *= -1
        if s.x + s.vx < 4 or s.x + s.vx > 19:
            s.vx *= -1
            if s.x == 4 and abs(s.y - s.paddle) <= PADDLE_HALF:
                s.vy *= -1
                if s.y + s.vy < 2 or s.y + s.vy > 20:
                    s.vy *= -1
        s.x += s.vx
        s.y += s.vy
        return self.render()

    def render(self) -> np.ndarray:
        frame = np.zeros((1, SIZE, SIZE), dtype=np.uint8)
        frame[0, 0, :] = 80
        frame[0, -1, :] = 80
        frame[0, :, -2] = 80
        s = self.state
        frame[0, s.paddle - PADDLE_HALF:s.paddle + PADDLE_HALF + 1, 1:3] = 255
        frame[0, s.y:s.y + 2, s.x:s.x + 2] = 255
        return frame


def collect(episodes: int, transitions: int, seed: int) -> dict[str, np.ndarray]:
    if episodes < 2 or transitions < 3:
        raise ValueError("Need at least 2 episodes and 3 transitions")
    frames = np.empty((episodes, transitions + 1, 1, SIZE, SIZE), dtype=np.uint8)
    actions = np.empty((episodes, transitions), dtype=np.int64)
    for episode in range(episodes):
        env = TinyPong(seed + episode)
        policy = np.random.default_rng(seed + 1_000_000 + episode)
        frames[episode, 0] = env.reset()
        for t in range(transitions):
            action = int(policy.integers(0, 3))
            actions[episode, t] = action
            frames[episode, t + 1] = env.step(action)
    return {"frames": frames, "actions": actions, "seed": np.array(seed)}

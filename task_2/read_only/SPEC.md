# Model and data contract

Given two observed frames `(o[t-1], o[t])` and an action `a[t]`, predict `o[t+1]`.
At deployment, generated frames become the context for later predictions under
a supplied action sequence. Training and validation should use separate episodes.
A sample's prediction should not depend on unrelated examples in its batch.

Images are 24 x 24 grayscale, normalized to `[-1,1]`. The network predicts a
velocity field along a straight interpolation path:

```text
z ~ Normal(0, I),       tau ~ Uniform(0, 1)
x_tau = (1 - tau) z + tau o[t+1]
loss = E[ || v_theta(x_tau, tau, context, action) - d(x_tau)/d(tau) ||^2 ]
```

Sampling starts from Gaussian noise at `tau=0` and integrates
`dx/dtau = v_theta(...)` to `tau=1` using explicit Euler steps. Environment time
`t` and flow time `tau` are different variables.

| Tensor | Shape |
| --- | --- |
| Context | `[B, 2, 1, 24, 24]` |
| Target / noisy frame / predicted velocity | `[B, 1, 24, 24]` |
| Flow time | `[B]` |
| Action IDs | `[B]` |
| Rollout actions | `[B, horizon]` |
| Recorded future / predicted sequence | `[B, horizon, 1, 24, 24]` |

## Data contract

TinyPong is a deterministic Pong-like toy, not an Atari emulator. It has a ball,
a controllable left paddle and closed court boundaries. Paddle contact at the
left boundary changes the ball's vertical direction. Two consecutive frames
reveal the motion direction. Episodes are fixed length and contain no resets;
the collection policy chooses actions independently.

| Stored array | Shape and interpretation |
| --- | --- |
| `frames` | `uint8 [episode, T+1, 1, 24, 24]`; includes the initial observation |
| `actions` | `int64 [episode, T]`; action at t is issued after observing frame t and produces frame t+1 |
| `seed` | First environment seed; episodes use successive seeds |

| Action | ID | Paddle displacement |
| --- | --- | --- |
| STAY | 0 | 0 pixels |
| UP | 1 | -2 image rows |
| DOWN | 2 | +2 image rows |

The paddle centre is clipped to rows `[4,19]`. The training archive contains 256
episodes of 48 transitions (seed 2026). The independent diagnostic archive has
32 episodes (seed 90000). The raw collector is trusted; loading and batching
training examples are part of the investigation.


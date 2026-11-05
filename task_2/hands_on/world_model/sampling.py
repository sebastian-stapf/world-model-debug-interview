"""Integrate a velocity field from Gaussian noise (t=0) to an image (t=1)."""

import torch


@torch.no_grad()
def sample_next(model, context, action, steps: int = 32, generator=None, initial_noise=None):
    if steps < 1:
        raise ValueError("ODE steps must be positive")
    shape = context[:, -1].shape
    x = torch.randn(shape, device=context.device, generator=generator) if initial_noise is None else initial_noise.clone()
    if x.shape != shape:
        raise ValueError("Noise and target frame shapes must match")
    dt = 1.0 / steps
    for i in range(steps):
        time = torch.full((len(x),), i * dt, device=x.device)
        x = x + dt * model(x, time, context, action)
    return x.clamp(-1, 1)


@torch.no_grad()
def generate(model, context, actions, steps: int = 32, seed: int = 0):
    """Deploy the model autoregressively. actions: [B, horizon]."""
    generator = torch.Generator(device=context.device).manual_seed(seed)
    generated = []
    for h in range(actions.shape[1]):
        prediction = sample_next(model, context, actions[:, h], steps, generator)
        generated.append(prediction)
        context = torch.cat([context[:, 1:], prediction[:, None]], dim=1)
    return torch.stack(generated, dim=1)

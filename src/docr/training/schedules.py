from __future__ import annotations

import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, LRScheduler


def build_scheduler(
    optimizer: Optimizer,
    name: str,
    max_steps: int,
    warmup_steps: int | float = 0,
) -> LRScheduler:
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if isinstance(warmup_steps, float) and 0.0 < warmup_steps < 1.0:
        warmup_steps = max(1, round(max_steps * warmup_steps))
    elif isinstance(warmup_steps, float) and not warmup_steps.is_integer():
        raise ValueError("fractional warmup_steps must satisfy 0 < warmup_steps < 1")
    warmup_steps = int(warmup_steps)
    if warmup_steps < 0 or warmup_steps >= max_steps:
        raise ValueError("warmup_steps must satisfy 0 <= warmup_steps < max_steps")

    def lr_multiplier(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        if name == "constant":
            return 1.0
        if name == "cosine":
            progress = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
            progress = min(max(progress, 0.0), 1.0)
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        raise ValueError(f"Unknown scheduler: {name}")

    if name in {"cosine", "constant"}:
        return LambdaLR(optimizer, lr_lambda=lr_multiplier)
    raise ValueError(f"Unknown scheduler: {name}")

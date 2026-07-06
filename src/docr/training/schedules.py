from __future__ import annotations

from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, LRScheduler


def build_scheduler(optimizer: Optimizer, name: str, max_steps: int) -> LRScheduler:
    if name == "cosine":
        return CosineAnnealingLR(optimizer, T_max=max_steps)
    if name == "constant":
        return LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
    raise ValueError(f"Unknown scheduler: {name}")


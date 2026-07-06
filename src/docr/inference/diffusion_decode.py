from __future__ import annotations

import torch


@torch.no_grad()
def diffusion_decode(
    model: torch.nn.Module,
    images: torch.Tensor,
    canvas_ids: torch.Tensor,
    steps: int = 1,
) -> torch.Tensor:
    current = canvas_ids
    for _ in range(steps):
        output = model(images=images, input_ids=current, mode="diffusion")
        current = output.logits.argmax(dim=-1)
    return current


from __future__ import annotations

import torch

from docr.inference.ar_decode import ar_decode
from docr.inference.diffusion_decode import diffusion_decode


@torch.no_grad()
def draft_refine_decode(
    model: torch.nn.Module,
    images: torch.Tensor,
    canvas_ids: torch.Tensor,
    diffusion_steps: int = 1,
) -> torch.Tensor:
    draft = diffusion_decode(model, images, canvas_ids, steps=diffusion_steps)
    return ar_decode(model, images, draft)


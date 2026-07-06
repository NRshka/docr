from __future__ import annotations

import torch


@torch.no_grad()
def ar_decode(model: torch.nn.Module, images: torch.Tensor, prompt_ids: torch.Tensor) -> torch.Tensor:
    output = model(images=images, input_ids=prompt_ids, mode="ar")
    return output.logits.argmax(dim=-1)


from __future__ import annotations

import torch
from torch.nn import functional as F


def language_model_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        labels.reshape(-1),
        ignore_index=ignore_index,
    )


def diffusion_denoising_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    prediction_mask: torch.Tensor,
) -> torch.Tensor:
    masked_labels = labels.masked_fill(~prediction_mask, -100)
    return language_model_loss(logits, masked_labels, ignore_index=-100)


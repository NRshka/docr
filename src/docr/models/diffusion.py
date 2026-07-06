from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DiscreteDiffusionSchedule:
    timesteps: int
    min_mask_ratio: float = 0.0
    max_mask_ratio: float = 1.0

    def mask_ratio(self, timestep: int | torch.Tensor) -> float | torch.Tensor:
        if isinstance(timestep, torch.Tensor):
            denom = max(self.timesteps - 1, 1)
            progress = timestep.float().clamp(0, self.timesteps - 1) / denom
            return self.min_mask_ratio + progress * (self.max_mask_ratio - self.min_mask_ratio)
        clipped = min(max(timestep, 0), self.timesteps - 1)
        denom = max(self.timesteps - 1, 1)
        progress = clipped / denom
        return self.min_mask_ratio + progress * (self.max_mask_ratio - self.min_mask_ratio)


def corrupt_with_mask(
    input_ids: torch.Tensor,
    timestep: int,
    schedule: DiscreteDiffusionSchedule,
    mask_token_id: int,
    special_token_ids: set[int] | None = None,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply absorbing-mask corruption.

    Returns corrupted ids and a boolean mask of positions selected for prediction.
    """

    if input_ids.ndim != 2:
        raise ValueError("input_ids must have shape [batch, seq_len]")

    ratio = float(schedule.mask_ratio(timestep))
    random_values = torch.rand(input_ids.shape, device=input_ids.device, generator=generator)
    prediction_mask = random_values < ratio

    if special_token_ids:
        protected = torch.zeros_like(prediction_mask)
        for token_id in special_token_ids:
            protected |= input_ids == token_id
        prediction_mask &= ~protected

    corrupted = input_ids.clone()
    corrupted[prediction_mask] = mask_token_id
    return corrupted, prediction_mask


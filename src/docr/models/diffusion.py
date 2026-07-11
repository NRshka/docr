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


@dataclass(frozen=True)
class BlockCorruptionBatch:
    noisy_ids: torch.Tensor
    targets: torch.Tensor
    prediction_mask: torch.Tensor
    valid_mask: torch.Tensor
    starts: torch.Tensor
    positions: torch.Tensor
    timesteps: torch.Tensor
    mask_ratios: torch.Tensor


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


def corrupt_contiguous_blocks(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    block_length: int,
    schedule: DiscreteDiffusionSchedule,
    mask_token_id: int,
    special_token_ids: set[int] | None = None,
    generator: torch.Generator | None = None,
) -> BlockCorruptionBatch:
    """Sample one next-token draft block and independent noise level per example."""

    if input_ids.ndim != 2:
        raise ValueError("input_ids must have shape [batch, seq_len]")
    if block_length <= 0:
        raise ValueError("block_length must be positive")
    batch_size, sequence_length = input_ids.shape
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    else:
        attention_mask = attention_mask.to(device=input_ids.device, dtype=torch.bool)

    valid_lengths = attention_mask.sum(dim=1).long()
    if torch.any(valid_lengths <= 0):
        raise ValueError("every sample needs at least one valid text token")
    starts = torch.empty(batch_size, dtype=torch.long, device=input_ids.device)
    eligible_input = attention_mask.clone()
    if special_token_ids:
        for token_id in special_token_ids:
            eligible_input &= input_ids != token_id
    timesteps = torch.randint(
        low=1,
        high=max(schedule.timesteps, 2),
        size=(batch_size,),
        device=input_ids.device,
        generator=generator,
    )
    for sample_idx in range(batch_size):
        eligible_positions = eligible_input[sample_idx].nonzero(as_tuple=False).flatten()
        if eligible_positions.numel() == 0:
            raise ValueError("every sample needs at least one non-special diffusion target")
        selected = torch.randint(
            low=0,
            high=eligible_positions.numel(),
            size=(1,),
            device=input_ids.device,
            generator=generator,
        )
        starts[sample_idx] = eligible_positions[selected]

    offsets = torch.arange(block_length, device=input_ids.device).unsqueeze(0)
    positions = starts.unsqueeze(1) + offsets
    valid_mask = positions < valid_lengths.unsqueeze(1)
    gather_positions = positions.clamp_max(max(sequence_length - 1, 0))
    targets = input_ids.gather(1, gather_positions)
    ratios = schedule.mask_ratio(timesteps).to(device=input_ids.device)
    random_values = torch.rand(
        (batch_size, block_length),
        device=input_ids.device,
        generator=generator,
    )
    prediction_mask = valid_mask & (random_values < ratios.unsqueeze(1))

    if special_token_ids:
        protected = torch.zeros_like(prediction_mask)
        for token_id in special_token_ids:
            protected |= targets == token_id
        prediction_mask &= ~protected
    else:
        protected = torch.zeros_like(prediction_mask)

    eligible = valid_mask & ~protected
    for sample_idx in range(batch_size):
        if not prediction_mask[sample_idx].any() and eligible[sample_idx].any():
            first_eligible = eligible[sample_idx].nonzero(as_tuple=False)[0, 0]
            prediction_mask[sample_idx, first_eligible] = True

    noisy_ids = targets.clone()
    noisy_ids[~valid_mask] = mask_token_id
    noisy_ids[prediction_mask] = mask_token_id
    return BlockCorruptionBatch(
        noisy_ids=noisy_ids,
        targets=targets,
        prediction_mask=prediction_mask,
        valid_mask=valid_mask,
        starts=starts,
        positions=positions,
        timesteps=timesteps,
        mask_ratios=ratios.float(),
    )

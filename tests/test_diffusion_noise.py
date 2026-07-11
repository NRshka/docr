import torch

from docr.models.diffusion import (
    DiscreteDiffusionSchedule,
    corrupt_contiguous_blocks,
    corrupt_with_mask,
)


def test_corruption_is_deterministic_with_fixed_seed():
    ids = torch.arange(12).view(2, 6)
    schedule = DiscreteDiffusionSchedule(timesteps=4)
    gen_a = torch.Generator().manual_seed(7)
    gen_b = torch.Generator().manual_seed(7)
    corrupted_a, mask_a = corrupt_with_mask(ids, 2, schedule, mask_token_id=99, generator=gen_a)
    corrupted_b, mask_b = corrupt_with_mask(ids, 2, schedule, mask_token_id=99, generator=gen_b)
    assert torch.equal(corrupted_a, corrupted_b)
    assert torch.equal(mask_a, mask_b)


def test_special_tokens_are_protected():
    ids = torch.tensor([[1, 2, 3, 4]])
    schedule = DiscreteDiffusionSchedule(timesteps=2)
    corrupted, mask = corrupt_with_mask(
        ids,
        1,
        schedule,
        mask_token_id=99,
        special_token_ids={2, 4},
        generator=torch.Generator().manual_seed(1),
    )
    assert corrupted[0, 1] == 2
    assert corrupted[0, 3] == 4
    assert not mask[0, 1]
    assert not mask[0, 3]


def test_mask_ratio_matches_timestep_extremes():
    schedule = DiscreteDiffusionSchedule(timesteps=5)
    assert schedule.mask_ratio(0) == 0.0
    assert schedule.mask_ratio(4) == 1.0


def test_block_corruption_is_reproducible_and_respects_valid_tokens():
    ids = torch.tensor([[1, 2, 3, 4, 0, 0], [5, 6, 7, 8, 9, 0]])
    attention = ids != 0
    schedule = DiscreteDiffusionSchedule(timesteps=8, min_mask_ratio=0.2, max_mask_ratio=0.9)
    first = corrupt_contiguous_blocks(
        ids,
        attention,
        block_length=4,
        schedule=schedule,
        mask_token_id=31,
        special_token_ids={0, 1},
        generator=torch.Generator().manual_seed(19),
    )
    second = corrupt_contiguous_blocks(
        ids,
        attention,
        block_length=4,
        schedule=schedule,
        mask_token_id=31,
        special_token_ids={0, 1},
        generator=torch.Generator().manual_seed(19),
    )

    assert torch.equal(first.starts, second.starts)
    assert torch.equal(first.timesteps, second.timesteps)
    assert torch.equal(first.noisy_ids, second.noisy_ids)
    assert torch.equal(first.prediction_mask, second.prediction_mask)
    assert not (first.prediction_mask & ~first.valid_mask).any()
    assert not (first.prediction_mask & (first.targets == 0)).any()
    assert not (first.prediction_mask & (first.targets == 1)).any()


def test_block_corruption_samples_noise_per_example():
    ids = torch.arange(1, 33).view(4, 8)
    block = corrupt_contiguous_blocks(
        ids,
        torch.ones_like(ids, dtype=torch.bool),
        block_length=4,
        schedule=DiscreteDiffusionSchedule(timesteps=32, min_mask_ratio=0.05, max_mask_ratio=0.95),
        mask_token_id=99,
        generator=torch.Generator().manual_seed(7),
    )

    assert block.timesteps.unique().numel() > 1
    assert block.mask_ratios.unique().numel() > 1

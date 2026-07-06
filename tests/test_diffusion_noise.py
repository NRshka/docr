import torch

from docr.models.diffusion import DiscreteDiffusionSchedule, corrupt_with_mask


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


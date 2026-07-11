import torch

from docr.training.ar_corruption import corrupt_ar_context


def test_ar_corruption_masks_only_attended_non_special_tokens():
    ids = torch.tensor([[10, 11, 2, 2, 2], [20, 31, 21, 2, 2]])
    attention = torch.tensor(
        [[True, True, True, False, False], [True, True, True, True, False]]
    )
    result = corrupt_ar_context(
        ids,
        attention,
        mask_token_id=31,
        probability=1.0,
        special_token_ids={2, 31},
    )
    assert result.eligible_mask.tolist() == [
        [True, True, False, False, False],
        [True, False, True, False, False],
    ]
    assert result.input_ids.tolist() == [[31, 31, 2, 2, 2], [31, 31, 31, 2, 2]]
    assert torch.equal(result.input_ids[~result.corruption_mask], ids[~result.corruption_mask])


def test_ar_corruption_is_reproducible_and_does_not_mutate_labels():
    ids = torch.tensor([[10, 11, 12, 2]])
    first = corrupt_ar_context(
        ids,
        torch.ones_like(ids, dtype=torch.bool),
        mask_token_id=31,
        probability=0.5,
        special_token_ids={2},
        generator=torch.Generator().manual_seed(7),
    )
    second = corrupt_ar_context(
        ids,
        torch.ones_like(ids, dtype=torch.bool),
        mask_token_id=31,
        probability=0.5,
        special_token_ids={2},
        generator=torch.Generator().manual_seed(7),
    )
    assert torch.equal(first.input_ids, second.input_ids)
    assert ids.tolist() == [[10, 11, 12, 2]]


def test_ar_corruption_validates_probability_and_shapes():
    ids = torch.ones(1, 3, dtype=torch.long)
    for probability in (-0.1, 1.1):
        try:
            corrupt_ar_context(ids, None, mask_token_id=3, probability=probability)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid probability was accepted")

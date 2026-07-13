import torch

from docr.training.ar_corruption import build_token_category_vocabulary, corrupt_ar_context


class StubTokenizer:
    pieces = ["<eos>", "apple", " pear", "12", "34", "{", "}", " ", "�"]

    def __len__(self):
        return len(self.pieces)

    def decode(self, ids, **kwargs):
        del kwargs
        return "".join(self.pieces[token_id] for token_id in ids)


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


def test_structured_corruption_preserves_categories_and_special_tokens():
    vocabulary = build_token_category_vocabulary(StubTokenizer(), special_token_ids={0})
    assert [pool.tolist() for pool in vocabulary.pools] == [[1, 2], [3, 4], [5, 6]]
    ids = torch.tensor([[1, 3, 5, 0, 7, 8]])
    result = corrupt_ar_context(
        ids,
        torch.ones_like(ids, dtype=torch.bool),
        mask_token_id=None,
        probability=1.0,
        special_token_ids={0},
        replacement_vocabulary=vocabulary,
        generator=torch.Generator().manual_seed(3),
    )
    assert result.input_ids.tolist() == [[2, 4, 6, 0, 7, 8]]
    assert result.eligible_mask.tolist() == [[True, True, True, False, False, False]]
    assert result.corruption_mask.tolist() == [[True, True, True, False, False, False]]


def test_structured_corruption_never_keeps_selected_original_token():
    vocabulary = build_token_category_vocabulary(StubTokenizer(), special_token_ids={0})
    ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
    result = corrupt_ar_context(
        ids,
        None,
        mask_token_id=None,
        probability=1.0,
        replacement_vocabulary=vocabulary,
        generator=torch.Generator().manual_seed(9),
    )
    assert torch.all(result.input_ids != ids)

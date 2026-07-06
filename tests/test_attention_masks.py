import torch

from docr.models.attention_masks import (
    build_ar_causal_mask,
    build_canvas_self_attention_mask,
    build_canvas_visual_cross_attention_mask,
    build_unified_canvas_visual_mask,
)


def test_ar_causal_mask_blocks_future_tokens():
    mask = build_ar_causal_mask(4)
    assert mask[2, 0]
    assert mask[2, 2]
    assert not mask[2, 3]


def test_canvas_mask_allows_only_same_canvas():
    mask = build_canvas_self_attention_mask(num_canvases=2, canvas_length=3)
    assert mask[0, 2]
    assert mask[3, 5]
    assert not mask[0, 3]
    assert not mask[5, 2]


def test_cross_attention_allows_all_visual_tokens():
    mask = build_canvas_visual_cross_attention_mask(
        num_canvases=2,
        canvas_length=3,
        num_visual_tokens=4,
    )
    assert mask.shape == (6, 4)
    assert torch.all(mask)


def test_unified_mask_blocks_canvas_to_canvas_leakage():
    mask = build_unified_canvas_visual_mask(num_visual_tokens=2, num_canvases=2, canvas_length=2)
    assert mask[2, 0]
    assert mask[2, 3]
    assert not mask[2, 4]


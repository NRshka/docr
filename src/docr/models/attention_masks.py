from __future__ import annotations

import torch


def build_ar_causal_mask(seq_len: int, device: torch.device | None = None) -> torch.Tensor:
    """Return a boolean mask where True means a query can attend to a key."""

    return torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool, device=device))


def build_canvas_self_attention_mask(
    num_canvases: int,
    canvas_length: int,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Block-diagonal bidirectional mask for independent diffusion canvases."""

    total = num_canvases * canvas_length
    mask = torch.zeros((total, total), dtype=torch.bool, device=device)
    for canvas_idx in range(num_canvases):
        start = canvas_idx * canvas_length
        end = start + canvas_length
        mask[start:end, start:end] = True
    return mask


def build_canvas_visual_cross_attention_mask(
    num_canvases: int,
    canvas_length: int,
    num_visual_tokens: int,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Allow every canvas token to attend to every visual token."""

    return torch.ones(
        (num_canvases * canvas_length, num_visual_tokens),
        dtype=torch.bool,
        device=device,
    )


def build_unified_canvas_visual_mask(
    num_visual_tokens: int,
    num_canvases: int,
    canvas_length: int,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Unified mask for [visual_tokens, canvas_tokens] sequences.

    Visual tokens attend to visual tokens. Canvas tokens attend to all visual tokens and to
    tokens inside the same canvas. Canvas tokens do not attend to other canvases.
    """

    text_len = num_canvases * canvas_length
    total = num_visual_tokens + text_len
    mask = torch.zeros((total, total), dtype=torch.bool, device=device)
    mask[:num_visual_tokens, :num_visual_tokens] = True
    mask[num_visual_tokens:, :num_visual_tokens] = True
    canvas_mask = build_canvas_self_attention_mask(num_canvases, canvas_length, device=device)
    mask[num_visual_tokens:, num_visual_tokens:] = canvas_mask
    return mask


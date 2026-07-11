from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ARContextCorruption:
    input_ids: torch.Tensor
    corruption_mask: torch.Tensor
    eligible_mask: torch.Tensor


def corrupt_ar_context(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    *,
    mask_token_id: int,
    probability: float,
    special_token_ids: set[int] | None = None,
    generator: torch.Generator | None = None,
) -> ARContextCorruption:
    """Mask valid content tokens while leaving clean next-token labels unchanged.

    With the decoder's shifted AR logits, changing input position i changes context for targets
    after i; it does not replace the clean label at i. Padding and every configured special token
    (notably EOS) are ineligible by construction.
    """

    if input_ids.ndim != 2:
        raise ValueError("input_ids must have shape [batch, sequence]")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be in [0, 1]")
    if attention_mask is None:
        eligible = torch.ones_like(input_ids, dtype=torch.bool)
    else:
        if attention_mask.shape != input_ids.shape:
            raise ValueError("attention_mask must have the same shape as input_ids")
        eligible = attention_mask.to(device=input_ids.device, dtype=torch.bool).clone()
    for token_id in special_token_ids or set():
        eligible &= input_ids != token_id

    sampled = torch.rand(input_ids.shape, device=input_ids.device, generator=generator)
    corruption_mask = eligible & (sampled < probability)
    corrupted = input_ids.clone()
    corrupted[corruption_mask] = mask_token_id
    return ARContextCorruption(
        input_ids=corrupted,
        corruption_mask=corruption_mask,
        eligible_mask=eligible,
    )

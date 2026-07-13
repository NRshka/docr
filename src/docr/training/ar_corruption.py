from __future__ import annotations

from dataclasses import dataclass
import unicodedata
from typing import Any

import torch


@dataclass(frozen=True)
class ARContextCorruption:
    input_ids: torch.Tensor
    corruption_mask: torch.Tensor
    eligible_mask: torch.Tensor


@dataclass(frozen=True)
class TokenCategoryVocabulary:
    category_by_token: torch.Tensor
    pools: tuple[torch.Tensor, ...]
    names: tuple[str, ...]


def _token_category(text: str) -> str | None:
    content = text.strip()
    # Fast tokenizers commonly decode invalid standalone byte pieces as U+FFFD.
    # Such pieces are not meaningful punctuation and must never be sampled.
    if not content or "\ufffd" in content:
        return None
    if any(character.isalpha() for character in content):
        return "alphabetic"
    if any(character.isdigit() for character in content):
        return "numeric"
    if all(unicodedata.category(character)[0] in {"P", "S"} for character in content):
        return "punctuation"
    return None


def build_token_category_vocabulary(
    tokenizer: Any,
    special_token_ids: set[int] | None = None,
) -> TokenCategoryVocabulary:
    """Group ordinary tokenizer IDs by their decoded semantic character class."""

    names = ("alphabetic", "numeric", "punctuation")
    name_to_index = {name: index for index, name in enumerate(names)}
    special = special_token_ids or set()
    category_by_token = torch.full((len(tokenizer),), -1, dtype=torch.long)
    grouped: list[list[int]] = [[] for _ in names]
    for token_id in range(len(tokenizer)):
        if token_id in special:
            continue
        decoded = tokenizer.decode(
            [token_id],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        category = _token_category(decoded)
        if category is None:
            continue
        category_idx = name_to_index[category]
        category_by_token[token_id] = category_idx
        grouped[category_idx].append(token_id)
    pools = tuple(torch.tensor(token_ids, dtype=torch.long) for token_ids in grouped)
    return TokenCategoryVocabulary(category_by_token, pools, names)


def corrupt_ar_context(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    *,
    mask_token_id: int | None,
    probability: float,
    special_token_ids: set[int] | None = None,
    replacement_vocabulary: TokenCategoryVocabulary | None = None,
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

    token_categories = None
    if replacement_vocabulary is not None:
        token_categories = replacement_vocabulary.category_by_token.to(input_ids.device)
        if input_ids.max() >= token_categories.numel() or input_ids.min() < 0:
            raise ValueError("input_ids contain IDs outside the replacement vocabulary")
        categories = token_categories[input_ids]
        eligible &= categories >= 0
        for category_idx, pool in enumerate(replacement_vocabulary.pools):
            if pool.numel() < 2:
                eligible &= categories != category_idx
    elif mask_token_id is None:
        raise ValueError("mask_token_id is required without a replacement vocabulary")

    sampled = torch.rand(input_ids.shape, device=input_ids.device, generator=generator)
    corruption_mask = eligible & (sampled < probability)
    corrupted = input_ids.clone()
    if replacement_vocabulary is None:
        corrupted[corruption_mask] = int(mask_token_id)
    else:
        categories = token_categories[input_ids]
        for category_idx, pool_cpu in enumerate(replacement_vocabulary.pools):
            active = corruption_mask & (categories == category_idx)
            count = int(active.sum().item())
            if count == 0:
                continue
            pool = pool_cpu.to(input_ids.device)
            sampled_indices = torch.randint(
                pool.numel(), (count,), device=input_ids.device, generator=generator
            )
            replacements = pool[sampled_indices]
            originals = input_ids[active]
            identical = replacements == originals
            if identical.any():
                replacements[identical] = pool[(sampled_indices[identical] + 1) % pool.numel()]
            corrupted[active] = replacements
    return ARContextCorruption(
        input_ids=corrupted,
        corruption_mask=corruption_mask,
        eligible_mask=eligible,
    )

from __future__ import annotations

from collections import Counter


def repeated_ngram_fraction(token_ids: list[int], n: int = 4) -> float:
    if n <= 0:
        raise ValueError("n must be positive")
    if len(token_ids) < n:
        return 0.0
    ngrams = [tuple(token_ids[index : index + n]) for index in range(len(token_ids) - n + 1)]
    counts = Counter(ngrams)
    repeated = sum(count - 1 for count in counts.values() if count > 1)
    return repeated / len(ngrams)


def generation_length_metrics(
    generated_ids: list[int],
    target_ids: list[int],
    *,
    eos_token_id: int | None,
    max_new_tokens: int,
) -> dict[str, float]:
    generated_length = len(generated_ids)
    target_length = len(target_ids)
    emitted_eos = (
        eos_token_id is not None
        and generated_length > 0
        and generated_ids[-1] == eos_token_id
    )
    length_error = generated_length - target_length
    return {
        "eos_emitted": float(emitted_eos),
        "hit_max_length": float(generated_length >= max_new_tokens and not emitted_eos),
        "generated_tokens": float(generated_length),
        "target_tokens": float(target_length),
        "length_ratio": generated_length / max(target_length, 1),
        "signed_length_error": float(length_error),
        "relative_absolute_length_error": abs(length_error) / max(target_length, 1),
        "early_termination": float(emitted_eos and generated_length < target_length),
        "late_termination": float(emitted_eos and generated_length > target_length),
        "repeated_4gram_fraction": repeated_ngram_fraction(generated_ids, n=4),
    }

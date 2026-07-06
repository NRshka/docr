from __future__ import annotations

import re


def edit_distance(reference: list[str], hypothesis: list[str]) -> int:
    prev = list(range(len(hypothesis) + 1))
    for i, ref_item in enumerate(reference, start=1):
        curr = [i]
        for j, hyp_item in enumerate(hypothesis, start=1):
            cost = 0 if ref_item == hyp_item else 1
            curr.append(min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost))
        prev = curr
    return prev[-1]


def character_error_rate(prediction: str, target: str) -> float:
    if not target:
        return 0.0 if not prediction else 1.0
    return edit_distance(list(target), list(prediction)) / len(target)


def word_error_rate(prediction: str, target: str) -> float:
    target_words = target.split()
    prediction_words = prediction.split()
    if not target_words:
        return 0.0 if not prediction_words else 1.0
    return edit_distance(target_words, prediction_words) / len(target_words)


def numeric_exact_match(prediction: str, target: str) -> float:
    pattern = re.compile(r"[-+]?\d+(?:[.,]\d+)?")
    target_numbers = pattern.findall(target)
    prediction_numbers = pattern.findall(prediction)
    if not target_numbers:
        return 1.0 if not prediction_numbers else 0.0
    return 1.0 if target_numbers == prediction_numbers else 0.0


def ocr_metrics(prediction: str, target: str) -> dict[str, float]:
    return {
        "cer": character_error_rate(prediction, target),
        "wer": word_error_rate(prediction, target),
        "numeric_exact_match": numeric_exact_match(prediction, target),
    }


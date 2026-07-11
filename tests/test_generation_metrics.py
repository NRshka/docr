import pytest

from docr.evaluation.generation import generation_length_metrics, repeated_ngram_fraction


def test_generation_length_metrics_detects_eos_and_late_termination():
    metrics = generation_length_metrics(
        [4, 5, 6, 2], [4, 5, 2], eos_token_id=2, max_new_tokens=10
    )
    assert metrics["eos_emitted"] == 1.0
    assert metrics["late_termination"] == 1.0
    assert metrics["early_termination"] == 0.0
    assert metrics["length_ratio"] == pytest.approx(4 / 3)


def test_generation_length_metrics_detects_unterminated_cap():
    metrics = generation_length_metrics([4, 5, 6], [4, 2], eos_token_id=2, max_new_tokens=3)
    assert metrics["eos_emitted"] == 0.0
    assert metrics["hit_max_length"] == 1.0


def test_repeated_ngram_fraction_counts_repetitions():
    assert repeated_ngram_fraction([1, 2, 3], n=4) == 0.0
    assert repeated_ngram_fraction([1, 2, 1, 2, 1, 2], n=2) == pytest.approx(3 / 5)

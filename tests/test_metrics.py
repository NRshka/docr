from docr.evaluation.metrics import character_error_rate, numeric_exact_match, word_error_rate


def test_character_error_rate_exact_and_empty():
    assert character_error_rate("abc", "abc") == 0.0
    assert character_error_rate("", "") == 0.0
    assert character_error_rate("x", "") == 1.0


def test_word_error_rate():
    assert word_error_rate("hello world", "hello world") == 0.0
    assert word_error_rate("hello", "hello world") == 0.5


def test_numeric_exact_match():
    assert numeric_exact_match("invoice 123 total 4.5", "invoice 123 total 4.5") == 1.0
    assert numeric_exact_match("invoice 124", "invoice 123") == 0.0
    assert numeric_exact_match("no numbers", "no numbers") == 1.0

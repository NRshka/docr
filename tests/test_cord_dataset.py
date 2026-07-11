import json

import pytest

from docr.data.dataset import cord_ground_truth_to_text, encode_text_with_eos


class StubTokenizer:
    eos_token_id = 2

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        assert not add_special_tokens
        return [int(piece) for piece in text.split()]


def test_cord_ground_truth_line_text_from_valid_lines():
    payload = {
        "gt_parse": {"total": {"total_price": "12.00"}},
        "valid_line": [
            {"words": [{"text": "TOTAL"}, {"text": "12.00"}]},
            {"words": [{"text": "CASH"}, {"text": "20.00"}]},
        ],
    }
    assert cord_ground_truth_to_text(json.dumps(payload), "line_text") == "TOTAL 12.00\nCASH 20.00"


def test_cord_ground_truth_line_text_falls_back_to_json():
    payload = {"gt_parse": {"total": {"total_price": "12.00"}}}
    assert cord_ground_truth_to_text(json.dumps(payload), "line_text") == (
        '{"total": {"total_price": "12.00"}}'
    )


def test_cord_ground_truth_raw_json():
    payload = {"gt_parse": {"menu": [{"nm": "Coffee"}]}}
    assert cord_ground_truth_to_text(json.dumps(payload), "raw_json") == (
        '{"menu": [{"nm": "Coffee"}]}'
    )


def test_target_encoding_appends_exactly_one_eos():
    tokenizer = StubTokenizer()
    assert encode_text_with_eos("4 5", tokenizer) == [4, 5, 2]
    assert encode_text_with_eos("4 5 2", tokenizer) == [4, 5, 2]


def test_target_encoding_reserves_eos_when_truncated():
    assert encode_text_with_eos("4 5 6 7", StubTokenizer(), max_text_length=3) == [4, 5, 2]
    assert encode_text_with_eos("4", StubTokenizer(), max_text_length=1) == [2]


def test_target_encoding_rejects_no_room_for_eos():
    with pytest.raises(ValueError, match="must be positive"):
        encode_text_with_eos("4", StubTokenizer(), max_text_length=0)

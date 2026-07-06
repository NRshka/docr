import json

from docr.data.dataset import cord_ground_truth_to_text


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

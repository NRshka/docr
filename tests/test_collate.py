import torch

from docr.data.collate import OCRCollator


def test_collate_pads_text_and_adds_canvas():
    collator = OCRCollator(pad_token_id=0, canvas_length=5, mask_token_id=99)
    batch = collator(
        [
            {"image": torch.zeros(3, 4, 4), "input_ids": torch.tensor([1, 2]), "text": "ab"},
            {"image": torch.zeros(3, 4, 4), "input_ids": torch.tensor([3, 4, 5]), "text": "cde"},
        ]
    )
    assert batch["images"].shape == (2, 3, 4, 4)
    assert batch["input_ids"].tolist() == [[1, 2, 0], [3, 4, 5]]
    assert batch["attention_mask"].tolist() == [[True, True, False], [True, True, True]]
    assert batch["canvas_input_ids"].tolist() == [[1, 2, 0, 99, 99], [3, 4, 5, 99, 99]]
    assert "texts" in batch


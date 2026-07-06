from __future__ import annotations

from typing import Any

import torch


def pad_1d(sequences: list[torch.Tensor], pad_value: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    if not sequences:
        raise ValueError("Cannot collate an empty batch")
    max_len = max(seq.numel() for seq in sequences)
    batch = torch.full((len(sequences), max_len), pad_value, dtype=sequences[0].dtype)
    mask = torch.zeros((len(sequences), max_len), dtype=torch.bool)
    for row, seq in enumerate(sequences):
        length = seq.numel()
        batch[row, :length] = seq
        mask[row, :length] = True
    return batch, mask


class OCRCollator:
    def __init__(
        self,
        pad_token_id: int = 0,
        canvas_length: int | None = None,
        mask_token_id: int | None = None,
    ) -> None:
        self.pad_token_id = pad_token_id
        self.canvas_length = canvas_length
        self.mask_token_id = mask_token_id if mask_token_id is not None else pad_token_id

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        images = torch.stack([example["image"] for example in examples])
        input_ids, attention_mask = pad_1d(
            [example["input_ids"] for example in examples],
            pad_value=self.pad_token_id,
        )
        batch: dict[str, Any] = {
            "images": images,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "texts": [example["text"] for example in examples],
            "image_paths": [example.get("image_path") for example in examples],
            "doc_ids": [example.get("doc_id") for example in examples],
            "page_ids": [example.get("page_id") for example in examples],
            "layout_paths": [example.get("layout_path") for example in examples],
            "metadata": [example.get("metadata", {}) for example in examples],
        }
        if self.canvas_length is not None:
            canvas = torch.full(
                (len(examples), self.canvas_length),
                self.mask_token_id,
                dtype=input_ids.dtype,
            )
            labels = torch.full_like(canvas, self.pad_token_id)
            copy_len = min(input_ids.shape[1], self.canvas_length)
            canvas[:, :copy_len] = input_ids[:, :copy_len]
            labels[:, :copy_len] = input_ids[:, :copy_len]
            batch["canvas_input_ids"] = canvas
            batch["canvas_labels"] = labels
        return batch


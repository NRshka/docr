from __future__ import annotations

import json
from itertools import islice
from pathlib import Path
from typing import Any
from typing import Protocol

import torch
from torch.utils.data import Dataset

from docr.data.manifest import ManifestRecord, read_manifest
from docr.data.transforms import build_image_transform, load_rgb_image


class TokenizerLike(Protocol):
    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]: ...


def encode_text_fallback(text: str, max_text_length: int | None = None) -> list[int]:
    ids = list(text.encode("utf-8"))
    if max_text_length is not None:
        ids = ids[:max_text_length]
    return ids


class ManifestOCRDataset(Dataset[dict]):
    def __init__(
        self,
        manifest_path: str | Path,
        image_root: str | Path = ".",
        image_size: tuple[int, int] = (1024, 768),
        tokenizer: TokenizerLike | None = None,
        max_text_length: int | None = None,
    ) -> None:
        self.records = read_manifest(manifest_path)
        self.image_root = Path(image_root)
        self.transform = build_image_transform(image_size)
        self.tokenizer = tokenizer
        self.max_text_length = max_text_length

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        image_path = Path(record.image_path)
        if not image_path.is_absolute():
            image_path = self.image_root / image_path

        image = self.transform(load_rgb_image(image_path))
        token_ids = self._encode_text(record.text)

        return {
            "image": image,
            "input_ids": torch.tensor(token_ids, dtype=torch.long),
            "text": record.text,
            "image_path": str(image_path),
            "doc_id": record.doc_id,
            "page_id": record.page_id,
            "layout_path": record.layout_path,
            "metadata": record.metadata,
        }

    def _encode_text(self, text: str) -> list[int]:
        if self.tokenizer is None:
            ids = encode_text_fallback(text)
        else:
            ids = self.tokenizer.encode(text, add_special_tokens=True)
        if self.max_text_length is not None:
            ids = ids[: self.max_text_length]
        return ids


class SyntheticOCRDataset(Dataset[dict]):
    def __init__(
        self,
        num_samples: int = 16,
        image_size: tuple[int, int] = (64, 64),
        text: str = "synthetic ocr sample 123",
    ) -> None:
        self.num_samples = num_samples
        self.image_size = image_size
        self.text = text

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> dict:
        width, height = self.image_size
        image = torch.zeros(3, height, width, dtype=torch.float32)
        token_ids = torch.tensor(encode_text_fallback(f"{self.text} {index}"), dtype=torch.long)
        return {
            "image": image,
            "input_ids": token_ids,
            "text": f"{self.text} {index}",
            "image_path": None,
            "doc_id": f"synthetic-{index}",
            "page_id": 0,
            "layout_path": None,
            "metadata": {},
        }


class HFCordV2OCRDataset(Dataset[dict]):
    """Small Hugging Face CORD-v2 adapter.

    The dataset exposes receipt images and a JSON `ground_truth` string. `target_mode=line_text`
    tries to flatten OCR words from `valid_line`; `target_mode=raw_json` keeps the full JSON
    target for document parsing experiments.
    """

    def __init__(
        self,
        dataset_name: str = "naver-clova-ix/cord-v2",
        dataset_path: str | Path | None = None,
        split: str = "train",
        image_size: tuple[int, int] = (1024, 768),
        target_mode: str = "line_text",
        load_from_disk: bool = False,
        streaming: bool = True,
        max_samples: int | None = 8,
        tokenizer: TokenizerLike | None = None,
        max_text_length: int | None = None,
    ) -> None:
        try:
            from datasets import load_dataset, load_from_disk as hf_load_from_disk
        except ImportError as exc:
            raise ImportError(
                "HFCordV2OCRDataset requires the `datasets` package. "
                "Install it with `python -m pip install datasets`."
            ) from exc

        if load_from_disk:
            if dataset_path is None:
                raise ValueError("dataset_path is required when load_from_disk=true")
            dataset_dict = hf_load_from_disk(str(dataset_path))
            dataset = dataset_dict[split] if hasattr(dataset_dict, "keys") else dataset_dict
        else:
            dataset = load_dataset(dataset_name, split=split, streaming=streaming)

        if streaming and not load_from_disk:
            if max_samples is None:
                raise ValueError("CORD-v2 streaming mode requires max_samples for this scaffold")
            self.records = list(islice(dataset, max_samples))
        else:
            if max_samples is not None:
                dataset = dataset.select(range(min(max_samples, len(dataset))))
            self.records = list(dataset)

        self.transform = build_image_transform(image_size)
        self.target_mode = target_mode
        self.tokenizer = tokenizer
        self.max_text_length = max_text_length

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        image = self.transform(record["image"].convert("RGB"))
        text = cord_ground_truth_to_text(record["ground_truth"], target_mode=self.target_mode)
        token_ids = self._encode_text(text)
        metadata = extract_cord_metadata(record["ground_truth"])
        return {
            "image": image,
            "input_ids": torch.tensor(token_ids, dtype=torch.long),
            "text": text,
            "image_path": None,
            "doc_id": metadata.get("image_id"),
            "page_id": metadata.get("split"),
            "layout_path": None,
            "metadata": metadata,
        }

    def _encode_text(self, text: str) -> list[int]:
        if self.tokenizer is None:
            return encode_text_fallback(text, self.max_text_length)
        ids = self.tokenizer.encode(text, add_special_tokens=True)
        if self.max_text_length is not None:
            ids = ids[: self.max_text_length]
        return ids


def cord_ground_truth_to_text(ground_truth: str, target_mode: str = "line_text") -> str:
    parsed = json.loads(ground_truth)
    if target_mode == "raw_json":
        return json.dumps(parsed.get("gt_parse", parsed), ensure_ascii=False, sort_keys=True)
    if target_mode == "line_text":
        lines = _extract_valid_line_text(parsed)
        if lines:
            return "\n".join(lines)
        return json.dumps(parsed.get("gt_parse", parsed), ensure_ascii=False, sort_keys=True)
    raise ValueError(f"Unknown CORD target_mode: {target_mode}")


def extract_cord_metadata(ground_truth: str) -> dict[str, Any]:
    parsed = json.loads(ground_truth)
    meta = parsed.get("meta", {})
    return meta if isinstance(meta, dict) else {}


def _extract_valid_line_text(parsed: dict[str, Any]) -> list[str]:
    valid_lines = parsed.get("valid_line", [])
    lines: list[str] = []
    if not isinstance(valid_lines, list):
        return lines
    for line in valid_lines:
        words = line.get("words", []) if isinstance(line, dict) else []
        texts = []
        for word in words:
            if not isinstance(word, dict):
                continue
            text = word.get("text") or word.get("value")
            if text:
                texts.append(str(text))
        if texts:
            lines.append(" ".join(texts))
    return lines

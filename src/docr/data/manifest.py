from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ManifestRecord:
    image_path: str
    text: str
    doc_id: str | None = None
    page_id: str | int | None = None
    layout_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def parse_manifest_line(line: str, line_number: int) -> ManifestRecord:
    try:
        payload = json.loads(line)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON on manifest line {line_number}: {exc}") from exc

    missing = {"image_path", "text"} - set(payload)
    if missing:
        names = ", ".join(sorted(missing))
        raise ValueError(f"Manifest line {line_number} missing required field(s): {names}")

    metadata = payload.get("metadata") or {}
    if not isinstance(metadata, dict):
        raise ValueError(f"Manifest line {line_number} field 'metadata' must be an object")

    return ManifestRecord(
        image_path=str(payload["image_path"]),
        text=str(payload["text"]),
        doc_id=payload.get("doc_id"),
        page_id=payload.get("page_id"),
        layout_path=payload.get("layout_path"),
        metadata=metadata,
    )


def read_manifest(path: str | Path) -> list[ManifestRecord]:
    manifest_path = Path(path)
    records: list[ManifestRecord] = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if stripped:
                records.append(parse_manifest_line(stripped, line_number))
    return records


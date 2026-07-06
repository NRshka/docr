from __future__ import annotations

import json
from pathlib import Path


def write_json_report(path: str | Path, metrics: dict[str, float]) -> None:
    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")


def write_markdown_report(path: str | Path, metrics: dict[str, float]) -> None:
    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# OCR Evaluation", ""]
    for name, value in sorted(metrics.items()):
        lines.append(f"- `{name}`: {value:.6f}")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


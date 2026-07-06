from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a JSONL OCR manifest from images/text files.")
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--text-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    text_dir = Path(args.text_dir)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for image_path in sorted(image_dir.iterdir()):
        if not image_path.is_file():
            continue
        text_path = text_dir / f"{image_path.stem}.txt"
        if not text_path.exists():
            continue
        rows.append({"image_path": str(image_path), "text": text_path.read_text(encoding="utf-8")})

    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()


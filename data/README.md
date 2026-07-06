# Data Directory

This repository expects local data files, not committed datasets.

- `raw/`: source document images and sidecar text files.
- `manifests/`: JSONL manifests used by training/evaluation.
- `processed/`: optional cached or preprocessed artifacts.

Manifest rows require `image_path` and `text`. Optional fields are `doc_id`, `page_id`,
`layout_path`, and `metadata`.


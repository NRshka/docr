# dOCR

Research scaffold for diffusion-assisted OCR over document scans.

The project explores a compressed visual encoder paired with a text decoder that can run in:

- autoregressive mode for reliable OCR and refinement,
- masked diffusion mode for canvas-based drafting,
- draft/refine mode where diffusion proposes text and AR decoding verifies or repairs it.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the research design.

## Setup

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

## Common Commands

```bash
python scripts/train.py
python scripts/train.py model=diffusion_ocr train=diffusion
python scripts/train.py data=cord_v2 data.max_samples=8 train.max_steps=2 train.batch_size=1 model.image_size=[256,256] model.visual_tokens=32 model.vision.hidden_size=64 model.decoder.hidden_size=64
python scripts/train.py data=cord_v2 model.vision.backbone=sam model.vision.freeze_backbone=true model.image_size=[1024,1024] model.visual_tokens=64 train.batch_size=1
python scripts/eval.py checkpoint_path=outputs/checkpoints/latest.pt
python scripts/infer.py image_path=data/raw/example.png mode=ar
```

The CORD-v2 command streams a tiny subset from Hugging Face by default. For larger local/GPU
runs, increase `data.max_samples`, `train.max_steps`, and the model dimensions.

## Data

The default dataset format is JSONL. Each row requires:

```json
{"image_path": "data/raw/page.png", "text": "recognized text"}
```

Optional fields:

- `doc_id`
- `page_id`
- `layout_path`
- `metadata`

Use `scripts/prepare_manifest.py` to create a simple manifest from an image folder and a text folder.

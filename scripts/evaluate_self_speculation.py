from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from docr.data.collate import OCRCollator
from docr.data.dataset import HFCordV2OCRDataset, ManifestOCRDataset
from docr.evaluation.metrics import ocr_metrics
from docr.inference.self_speculative import greedy_ar_decode, linear_self_speculative_decode
from docr.models.factory import build_model
from docr.utils.tokenizer import build_tokenizer, tokenizer_pad_id


def load_lightning_model(model: torch.nn.Module, path: str | Path) -> None:
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=False)
    state = checkpoint.get("state_dict", checkpoint)
    state = {key.removeprefix("model."): value for key, value in state.items() if key.startswith("model.")}
    if not state:
        raise ValueError("checkpoint contains no model.* state_dict entries")
    model.load_state_dict(state, strict=True)


def build_validation_dataset(cfg: DictConfig, tokenizer: Any):
    image_size = tuple(cfg.model.image_size)
    common = {
        "image_size": image_size,
        "tokenizer": tokenizer,
        "max_text_length": cfg.data.max_text_length,
        "preserve_aspect_ratio": bool(cfg.model.vision.get("preserve_aspect_ratio", False)),
        "image_normalization": str(cfg.model.vision.get("image_normalization", "none")),
    }
    if cfg.data.name == "cord_v2":
        return HFCordV2OCRDataset(
            dataset_name=cfg.data.dataset_name,
            dataset_path=cfg.data.get("dataset_path", None),
            split=cfg.data.get("val_split", "validation"),
            target_mode=cfg.data.target_mode,
            load_from_disk=bool(cfg.data.get("load_from_disk", False)),
            streaming=bool(cfg.data.get("streaming", False)),
            max_samples=cfg.phase4.get("max_samples", cfg.data.get("val_max_samples", None)),
            **common,
        )
    if cfg.data.name == "manifest":
        return ManifestOCRDataset(
            manifest_path=cfg.data.val_manifest, image_root=cfg.data.image_root, **common
        )
    raise ValueError(f"Phase-4 evaluation does not support dataset {cfg.data.name!r}")


def decode_text(tokenizer: Any, token_ids: torch.Tensor) -> str:
    ids = token_ids[0].tolist()
    if tokenizer is None:
        return bytes(token for token in ids if 0 <= token < 256).decode("utf-8", errors="replace")
    return tokenizer.decode(ids, skip_special_tokens=True)


def valid_json(text: str) -> float:
    try:
        json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return 0.0
    return 1.0


def mean(values: list[float]) -> float:
    return sum(values) / max(len(values), 1)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    if cfg.checkpoint_path is None:
        raise ValueError("checkpoint_path must point to the completed D02 Lightning checkpoint")
    torch.set_float32_matmul_precision(str(cfg.train.get("float32_matmul_precision", "high")))
    configured_device = str(cfg.device)
    if configured_device == "auto":
        configured_device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(configured_device)
    tokenizer = build_tokenizer(cfg)
    query_token_id = tokenizer_pad_id(tokenizer)
    mask_token_id = getattr(tokenizer, "mask_token_id", None)
    if mask_token_id is None:
        mask_token_id = cfg.model.diffusion.get("mask_token_id", query_token_id)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)

    model = build_model(cfg)
    if tokenizer is not None and hasattr(model.decoder, "lm"):
        model.decoder.lm.resize_token_embeddings(len(tokenizer))
    load_lightning_model(model, str(cfg.checkpoint_path))
    model.to(device).eval()

    dataset = build_validation_dataset(cfg, tokenizer)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=int(cfg.data.get("num_workers", 0)),
        collate_fn=OCRCollator(pad_token_id=query_token_id),
    )
    widths = [int(width) for width in cfg.phase4.draft_widths]
    records: list[dict[str, Any]] = []
    aggregates: dict[str, list[float]] = {}

    def add_metric(name: str, value: float) -> None:
        aggregates.setdefault(name, []).append(float(value))

    for sample_index, batch in enumerate(loader):
        image = batch["images"].to(device)
        target = batch["texts"][0]
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            baseline = greedy_ar_decode(
                model,
                image,
                query_token_id=query_token_id,
                max_new_tokens=int(cfg.phase4.max_new_tokens),
                eos_token_id=eos_token_id,
            )
        if device.type == "cuda":
            torch.cuda.synchronize()
        ar_seconds = time.perf_counter() - started
        ar_text = decode_text(tokenizer, baseline.token_ids)
        ar_scores = ocr_metrics(ar_text, target)
        for name, value in ar_scores.items():
            add_metric(f"ar/{name}", value)
        add_metric("ar/json_valid", valid_json(ar_text))
        add_metric("ar/seconds", ar_seconds)
        record: dict[str, Any] = {
            "sample": sample_index,
            "target": target,
            "ar": {"text": ar_text, "seconds": ar_seconds, **ar_scores},
            "self_speculative": {},
        }

        for width in widths:
            if device.type == "cuda":
                torch.cuda.synchronize()
            started = time.perf_counter()
            with torch.autocast(
                device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
            ):
                result = linear_self_speculative_decode(
                    model,
                    image,
                    mask_token_id=int(mask_token_id),
                    query_token_id=query_token_id,
                    diffusion_timestep=int(cfg.model.diffusion.timesteps) - 1,
                    draft_width=width,
                    max_new_tokens=int(cfg.phase4.max_new_tokens),
                    eos_token_id=eos_token_id,
                )
            if device.type == "cuda":
                torch.cuda.synchronize()
            seconds = time.perf_counter() - started
            text = decode_text(tokenizer, result.token_ids)
            scores = ocr_metrics(text, target)
            prefix = f"self_spec/w{width}"
            agreement = [
                accepted / max(proposed, 1)
                for accepted, proposed in zip(
                    result.stats.accepted_by_offset, result.stats.proposed_by_offset
                )
            ]
            values = {
                **scores,
                "json_valid": valid_json(text),
                "seconds": seconds,
                "cycles": result.stats.cycles,
                "model_forwards": result.stats.model_forwards,
                "mean_accepted_prefix": result.stats.mean_accepted_prefix,
                "tokens_per_forward": result.stats.tokens_per_forward,
                "zero_acceptance_rate": result.stats.zero_acceptance_cycles
                / max(result.stats.cycles, 1),
                "exact_ar_match": float(torch.equal(result.token_ids, baseline.token_ids)),
            }
            for name, value in values.items():
                add_metric(f"{prefix}/{name}", float(value))
            for offset, value in enumerate(agreement):
                add_metric(f"{prefix}/draft_agreement_offset_{offset:02d}", value)
            record["self_speculative"][str(width)] = {
                "text": text,
                **values,
                "draft_agreement_by_offset": agreement,
            }
        records.append(record)
        print(f"evaluated={sample_index + 1}/{len(dataset)}")

    summary = {name: mean(values) for name, values in aggregates.items()}
    output_path = Path(str(cfg.phase4.output_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps({"config": OmegaConf.to_container(cfg, resolve=True), "summary": summary, "records": records}, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    print(f"report={output_path}")


if __name__ == "__main__":
    main()

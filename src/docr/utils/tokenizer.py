from __future__ import annotations

from typing import Any


def build_tokenizer(cfg: Any):
    implementation = str(cfg.model.decoder.get("implementation", "tiny"))
    if implementation in {"tiny", "diffusion_transformer"}:
        return None
    if bool(cfg.model.decoder.get("random_init", False)) and cfg.model.decoder.get("tokenizer_name", None) is None:
        return None

    from transformers import AutoTokenizer

    tokenizer_name = cfg.model.decoder.get("tokenizer_name", cfg.model.decoder.backbone_name)
    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_name),
        local_files_only=bool(cfg.model.decoder.get("local_files_only", False)),
        trust_remote_code=bool(cfg.model.decoder.get("trust_remote_code", False)),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    mask_token = cfg.model.get("diffusion", {}).get("mask_token", None)
    if mask_token is not None:
        tokenizer.add_special_tokens({"mask_token": str(mask_token)})
    return tokenizer


def tokenizer_pad_id(tokenizer: Any | None, fallback: int = 0) -> int:
    if tokenizer is None:
        return fallback
    if tokenizer.pad_token_id is None:
        return int(tokenizer.eos_token_id)
    return int(tokenizer.pad_token_id)

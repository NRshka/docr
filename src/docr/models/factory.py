from __future__ import annotations

from typing import Any

from docr.models.decoder import DiffusionTransformerDecoder, OCRModel, QwenVisualPrefixDecoder
from docr.models.decoder import TinyTextDecoder
from docr.models.vision_encoder import SamVisionEncoder, TinyVisionEncoder


def build_vision_encoder(cfg: Any):
    backbone = str(cfg.model.vision.get("backbone", "patch"))
    common = {
        "hidden_size": int(cfg.model.vision.hidden_size),
        "output_tokens": int(cfg.model.visual_tokens),
        "compressor_type": str(cfg.model.compressor.get("type", "perceiver")),
        "compressor_layers": int(cfg.model.compressor.get("num_layers", 2)),
        "compressor_heads": int(cfg.model.compressor.get("num_heads", 8)),
        "compressor_mlp_ratio": float(cfg.model.compressor.get("mlp_ratio", 4.0)),
        "compressor_dropout": float(cfg.model.compressor.get("dropout", 0.0)),
    }
    if backbone == "patch":
        return TinyVisionEncoder(
            patch_size=int(cfg.model.vision.patch_size),
            positional_embedding=str(cfg.model.vision.get("positional_embedding", "learned_2d")),
            max_grid_size=int(cfg.model.vision.get("max_grid_size", 256)),
            **common,
        )
    if backbone == "sam":
        return SamVisionEncoder(
            backbone_name=str(cfg.model.vision.get("backbone_name", "facebook/sam-vit-base")),
            freeze_backbone=bool(cfg.model.vision.get("freeze_backbone", True)),
            local_files_only=bool(cfg.model.vision.get("local_files_only", False)),
            trust_remote_code=bool(cfg.model.vision.get("trust_remote_code", False)),
            **common,
        )
    raise ValueError(f"Unknown vision backbone: {backbone}")


def build_model(cfg: Any) -> OCRModel:
    vision = build_vision_encoder(cfg)

    implementation = str(cfg.model.decoder.get("implementation", "tiny"))
    if implementation == "tiny":
        decoder = TinyTextDecoder(
            vocab_size=int(cfg.model.decoder.get("vocab_size", 256)),
            hidden_size=int(cfg.model.decoder.hidden_size),
        )
    elif implementation == "qwen_visual_prefix":
        decoder = QwenVisualPrefixDecoder(
            backbone_name=str(cfg.model.decoder.backbone_name),
            visual_hidden_size=int(cfg.model.vision.hidden_size),
            timesteps=int(cfg.model.diffusion.get("timesteps", 32)),
            freeze_lm=bool(cfg.model.decoder.get("freeze_lm", False)),
            local_files_only=bool(cfg.model.decoder.get("local_files_only", False)),
            trust_remote_code=bool(cfg.model.decoder.get("trust_remote_code", False)),
        )
    elif implementation == "diffusion_transformer":
        decoder = DiffusionTransformerDecoder(
            vocab_size=int(cfg.model.decoder.get("vocab_size", 256)),
            hidden_size=int(cfg.model.decoder.hidden_size),
            visual_hidden_size=int(cfg.model.vision.hidden_size),
            num_layers=int(cfg.model.decoder.get("num_layers", 4)),
            num_heads=int(cfg.model.decoder.get("num_heads", 8)),
            mlp_ratio=float(cfg.model.decoder.get("mlp_ratio", 4.0)),
            max_length=int(cfg.model.decoder.get("max_length", cfg.model.canvas.length)),
            timesteps=int(cfg.model.diffusion.timesteps),
            dropout=float(cfg.model.decoder.get("dropout", 0.0)),
        )
    else:
        raise ValueError(f"Unknown decoder implementation: {implementation}")

    return OCRModel(vision_encoder=vision, decoder=decoder)


def decoder_tokenizer_name(cfg: Any) -> str | None:
    tokenizer_name = cfg.model.decoder.get("tokenizer_name", None)
    if tokenizer_name:
        return str(tokenizer_name)
    backbone_name = cfg.model.decoder.get("backbone_name", None)
    return str(backbone_name) if backbone_name else None

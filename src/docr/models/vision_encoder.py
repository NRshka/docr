from __future__ import annotations

import torch
from torch import nn

from docr.models.token_compressor import Learned2DPositionEmbedding, MeanTokenCompressor
from docr.models.token_compressor import PerceiverTokenCompressor


def build_token_compressor(
    hidden_size: int,
    output_tokens: int,
    compressor_type: str = "perceiver",
    compressor_layers: int = 2,
    compressor_heads: int = 8,
    compressor_mlp_ratio: float = 4.0,
    compressor_dropout: float = 0.0,
) -> nn.Module:
    if compressor_type == "mean":
        return MeanTokenCompressor(hidden_size, output_tokens)
    if compressor_type == "perceiver":
        return PerceiverTokenCompressor(
            input_dim=hidden_size,
            output_tokens=output_tokens,
            num_layers=compressor_layers,
            num_heads=compressor_heads,
            mlp_ratio=compressor_mlp_ratio,
            dropout=compressor_dropout,
        )
    raise ValueError(f"Unknown compressor_type: {compressor_type}")


class TinyVisionEncoder(nn.Module):
    """Patch encoder with optional 2D positions and configurable token compressor."""

    def __init__(
        self,
        patch_size: int = 16,
        hidden_size: int = 512,
        output_tokens: int = 768,
        compressor_type: str = "perceiver",
        compressor_layers: int = 2,
        compressor_heads: int = 8,
        compressor_mlp_ratio: float = 4.0,
        compressor_dropout: float = 0.0,
        positional_embedding: str = "learned_2d",
        max_grid_size: int = 256,
    ) -> None:
        super().__init__()
        self.patch = nn.Conv2d(3, hidden_size, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(hidden_size)
        self.positional_embedding = positional_embedding
        if positional_embedding == "learned_2d":
            self.pos_embed = Learned2DPositionEmbedding(hidden_size, max_grid_size=max_grid_size)
        elif positional_embedding == "none":
            self.pos_embed = None
        else:
            raise ValueError(f"Unknown positional_embedding: {positional_embedding}")

        self.compressor = build_token_compressor(
            hidden_size=hidden_size,
            output_tokens=output_tokens,
            compressor_type=compressor_type,
            compressor_layers=compressor_layers,
            compressor_heads=compressor_heads,
            compressor_mlp_ratio=compressor_mlp_ratio,
            compressor_dropout=compressor_dropout,
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        patch_grid = self.patch(images)
        grid_height, grid_width = patch_grid.shape[-2:]
        patches = patch_grid.flatten(2).transpose(1, 2)
        if self.pos_embed is not None:
            patches = patches + self.pos_embed(grid_height, grid_width, patches.device)
        patches = self.norm(patches)
        return self.compressor(patches)


class SamVisionEncoder(nn.Module):
    """SAM vision backbone followed by projection and visual-token resampling."""

    def __init__(
        self,
        backbone_name: str = "facebook/sam-vit-base",
        hidden_size: int = 512,
        output_tokens: int = 768,
        freeze_backbone: bool = True,
        local_files_only: bool = False,
        trust_remote_code: bool = False,
        compressor_type: str = "perceiver",
        compressor_layers: int = 2,
        compressor_heads: int = 8,
        compressor_mlp_ratio: float = 4.0,
        compressor_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        from transformers import SamVisionModel

        self.backbone = SamVisionModel.from_pretrained(
            backbone_name,
            local_files_only=local_files_only,
            trust_remote_code=trust_remote_code,
        )
        backbone_dim = int(getattr(self.backbone.config, "output_channels", 0))
        if backbone_dim <= 0:
            backbone_dim = int(self.backbone.config.hidden_size)
        self.proj = nn.Linear(backbone_dim, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
        self.compressor = build_token_compressor(
            hidden_size=hidden_size,
            output_tokens=output_tokens,
            compressor_type=compressor_type,
            compressor_layers=compressor_layers,
            compressor_heads=compressor_heads,
            compressor_mlp_ratio=compressor_mlp_ratio,
            compressor_dropout=compressor_dropout,
        )

        if freeze_backbone:
            self.backbone.eval()
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False

    def train(self, mode: bool = True) -> "SamVisionEncoder":
        super().train(mode)
        if not any(parameter.requires_grad for parameter in self.backbone.parameters()):
            self.backbone.eval()
        return self

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if any(parameter.requires_grad for parameter in self.backbone.parameters()):
            features = self.backbone(pixel_values=images).last_hidden_state
        else:
            with torch.no_grad():
                features = self.backbone(pixel_values=images).last_hidden_state
        if features.ndim == 4:
            features = features.flatten(2).transpose(1, 2)
        elif features.ndim != 3:
            raise ValueError("SAM vision output must have shape [batch, channels, h, w] or [batch, seq, dim]")
        features = self.norm(self.proj(features))
        return self.compressor(features)

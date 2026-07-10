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


class CogViTOCRVisionEncoder(nn.Module):
    """GLM-OCR CogViT tower with its native spatial merger and no Perceiver.

    Input images must already be normalized and have height/width divisible by
    ``patch_size * spatial_merge_size``. Static images are duplicated across the
    temporal patch dimension exactly as the official GLM image processor does.
    """

    def __init__(
        self,
        backbone_name: str = "zai-org/GLM-OCR",
        freeze_backbone: bool = True,
        local_files_only: bool = False,
        trust_remote_code: bool = False,
        backbone: nn.Module | None = None,
    ) -> None:
        super().__init__()
        if backbone is None:
            from transformers import GlmOcrForConditionalGeneration

            full_model = GlmOcrForConditionalGeneration.from_pretrained(
                backbone_name,
                local_files_only=local_files_only,
                trust_remote_code=trust_remote_code,
            )
            backbone = full_model.model.visual
            del full_model
        self.backbone = backbone
        self.patch_size = int(self.backbone.config.patch_size)
        self.temporal_patch_size = int(self.backbone.config.temporal_patch_size)
        self.spatial_merge_size = int(self.backbone.config.spatial_merge_size)
        self.output_dim = int(self.backbone.config.out_hidden_size)

        if freeze_backbone:
            self.backbone.eval()
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False

    def train(self, mode: bool = True) -> "CogViTOCRVisionEncoder":
        super().train(mode)
        if not any(parameter.requires_grad for parameter in self.backbone.parameters()):
            self.backbone.eval()
        return self

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        flattened_patches, grid_thw = self._patchify(images)
        if any(parameter.requires_grad for parameter in self.backbone.parameters()):
            outputs = self.backbone(flattened_patches, grid_thw=grid_thw)
        else:
            with torch.no_grad():
                outputs = self.backbone(flattened_patches, grid_thw=grid_thw)

        tokens_per_image = (
            grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2]
        ) // self.spatial_merge_size**2
        if not torch.equal(tokens_per_image, tokens_per_image[:1].expand_as(tokens_per_image)):
            raise ValueError("Fixed-batch CogViT currently requires equal visual token counts")
        return outputs.pooler_output.view(images.shape[0], int(tokens_per_image[0]), self.output_dim)

    def _patchify(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("CogViT images must have shape [batch, 3, height, width]")
        batch_size, channels, height, width = images.shape
        merge_stride = self.patch_size * self.spatial_merge_size
        if height % merge_stride != 0 or width % merge_stride != 0:
            raise ValueError(
                f"CogViT image size must be divisible by {merge_stride}, got {height}x{width}"
            )

        patches = images.unsqueeze(1).repeat(1, self.temporal_patch_size, 1, 1, 1)
        grid_t = 1
        grid_h = height // self.patch_size
        grid_w = width // self.patch_size
        patches = patches.view(
            batch_size,
            grid_t,
            self.temporal_patch_size,
            channels,
            grid_h // self.spatial_merge_size,
            self.spatial_merge_size,
            self.patch_size,
            grid_w // self.spatial_merge_size,
            self.spatial_merge_size,
            self.patch_size,
        )
        patches = patches.permute(0, 1, 4, 7, 5, 8, 3, 2, 6, 9)
        flattened = patches.reshape(
            batch_size * grid_t * grid_h * grid_w,
            channels * self.temporal_patch_size * self.patch_size * self.patch_size,
        )
        grid_thw = torch.tensor(
            [[grid_t, grid_h, grid_w]] * batch_size,
            dtype=torch.long,
            device=images.device,
        )
        return flattened, grid_thw

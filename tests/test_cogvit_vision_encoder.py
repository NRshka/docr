from types import SimpleNamespace

import torch

from docr.models.vision_encoder import CogViTOCRVisionEncoder


class FakeCogViT(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(
            patch_size=2,
            temporal_patch_size=2,
            spatial_merge_size=2,
            out_hidden_size=6,
        )
        self.scale = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, flattened_patches, grid_thw):
        token_count = int((grid_thw.prod(dim=-1) // 4).sum())
        pooled = torch.ones(token_count, 6, device=flattened_patches.device) * self.scale
        return SimpleNamespace(pooler_output=pooled)


def test_cogvit_uses_native_merged_tokens_without_compressor():
    encoder = CogViTOCRVisionEncoder(backbone=FakeCogViT(), freeze_backbone=True)
    output = encoder(torch.randn(2, 3, 8, 12))

    assert output.shape == (2, 6, 6)
    assert not hasattr(encoder, "compressor")
    assert not encoder.backbone.scale.requires_grad


def test_cogvit_patch_order_matches_expected_shape():
    encoder = CogViTOCRVisionEncoder(backbone=FakeCogViT(), freeze_backbone=False)
    flattened, grid = encoder._patchify(torch.randn(2, 3, 8, 12))

    assert flattened.shape == (48, 24)
    assert torch.equal(grid, torch.tensor([[1, 4, 6], [1, 4, 6]]))

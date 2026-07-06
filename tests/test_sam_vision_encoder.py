import sys
import types

import torch
from torch import nn

from omegaconf import OmegaConf

from docr.models.factory import build_vision_encoder
from docr.models.vision_encoder import SamVisionEncoder


class FakeSamOutput:
    def __init__(self, last_hidden_state):
        self.last_hidden_state = last_hidden_state


class FakeSamVisionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = types.SimpleNamespace(output_channels=16, hidden_size=32)
        self.conv = nn.Conv2d(3, 16, kernel_size=16, stride=16)

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        del args, kwargs
        return cls()

    def forward(self, pixel_values):
        return FakeSamOutput(last_hidden_state=self.conv(pixel_values))


def install_fake_transformers(monkeypatch):
    fake_module = types.SimpleNamespace(SamVisionModel=FakeSamVisionModel)
    monkeypatch.setitem(sys.modules, "transformers", fake_module)


def test_sam_vision_encoder_outputs_visual_tokens(monkeypatch):
    install_fake_transformers(monkeypatch)
    encoder = SamVisionEncoder(
        backbone_name="fake-sam",
        hidden_size=32,
        output_tokens=6,
        compressor_layers=1,
        compressor_heads=4,
        freeze_backbone=True,
    )
    output = encoder(torch.randn(2, 3, 64, 64))
    assert output.shape == (2, 6, 32)
    assert not any(parameter.requires_grad for parameter in encoder.backbone.parameters())


def test_sam_vision_encoder_can_unfreeze_backbone(monkeypatch):
    install_fake_transformers(monkeypatch)
    encoder = SamVisionEncoder(
        backbone_name="fake-sam",
        hidden_size=32,
        output_tokens=4,
        compressor_layers=1,
        compressor_heads=4,
        freeze_backbone=False,
    )
    output = encoder(torch.randn(1, 3, 64, 64))
    output.mean().backward()
    assert any(parameter.grad is not None for parameter in encoder.backbone.parameters())


def test_factory_builds_sam_vision_encoder(monkeypatch):
    install_fake_transformers(monkeypatch)
    cfg = OmegaConf.create(
        {
            "model": {
                "visual_tokens": 3,
                "vision": {
                    "backbone": "sam",
                    "backbone_name": "fake-sam",
                    "freeze_backbone": True,
                    "local_files_only": False,
                    "trust_remote_code": False,
                    "hidden_size": 32,
                },
                "compressor": {
                    "type": "perceiver",
                    "num_layers": 1,
                    "num_heads": 4,
                    "mlp_ratio": 4.0,
                    "dropout": 0.0,
                },
            }
        }
    )
    encoder = build_vision_encoder(cfg)
    assert isinstance(encoder, SamVisionEncoder)
    assert encoder(torch.randn(1, 3, 64, 64)).shape == (1, 3, 32)

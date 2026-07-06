import torch

from docr.models.vision_encoder import TinyVisionEncoder


def test_vision_encoder_outputs_configured_token_count():
    encoder = TinyVisionEncoder(
        patch_size=8,
        hidden_size=32,
        output_tokens=5,
        compressor_type="perceiver",
        compressor_layers=1,
        compressor_heads=4,
    )
    output = encoder(torch.randn(2, 3, 32, 32))
    assert output.shape == (2, 5, 32)


def test_vision_encoder_token_count_is_configurable():
    short_encoder = TinyVisionEncoder(
        patch_size=8,
        hidden_size=16,
        output_tokens=3,
        compressor_type="perceiver",
        compressor_layers=1,
        compressor_heads=4,
    )
    long_encoder = TinyVisionEncoder(
        patch_size=8,
        hidden_size=16,
        output_tokens=7,
        compressor_type="perceiver",
        compressor_layers=1,
        compressor_heads=4,
    )
    image = torch.randn(1, 3, 32, 32)
    assert short_encoder(image).shape[1] == 3
    assert long_encoder(image).shape[1] == 7


def test_vision_encoder_changes_for_different_images():
    encoder = TinyVisionEncoder(
        patch_size=8,
        hidden_size=32,
        output_tokens=4,
        compressor_type="perceiver",
        compressor_layers=1,
        compressor_heads=4,
    )
    first = torch.zeros(1, 3, 32, 32)
    second = torch.zeros(1, 3, 32, 32)
    second[:, :, :16, :16] = 1.0
    assert not torch.allclose(encoder(first), encoder(second))


def test_vision_encoder_allows_gradients():
    encoder = TinyVisionEncoder(
        patch_size=8,
        hidden_size=32,
        output_tokens=4,
        compressor_type="perceiver",
        compressor_layers=1,
        compressor_heads=4,
    )
    output = encoder(torch.randn(2, 3, 32, 32))
    output.square().mean().backward()
    assert encoder.patch.weight.grad is not None
    assert encoder.compressor.query_tokens.grad is not None

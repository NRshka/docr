import pytest
import torch

from docr.models.qwen_unified import GatedVisualCrossAttention, UnifiedQwenDecoder


def test_zero_gate_is_exact_identity_and_gate_receives_gradient():
    module = GatedVisualCrossAttention(hidden_size=16, num_heads=4)
    hidden = torch.randn(2, 5, 16, requires_grad=True)
    visual = torch.randn(2, 7, 16)

    output = module(hidden, visual)

    assert torch.equal(output, hidden)
    output.sum().backward()
    assert module.gate.grad is not None
    assert torch.isfinite(module.gate.grad)


def test_open_gate_changes_text_states_and_attention_parameters_receive_gradients():
    module = GatedVisualCrossAttention(hidden_size=16, num_heads=4)
    module.gate.data.fill_(0.1)
    hidden = torch.randn(2, 5, 16, requires_grad=True)
    visual = torch.randn(2, 7, 16)

    output = module(hidden, visual)
    output.square().mean().backward()

    assert not torch.equal(output, hidden)
    assert module.attention.in_proj_weight.grad is not None
    assert torch.isfinite(module.attention.in_proj_weight.grad).all()


def test_cross_attention_layer_indices_support_negative_values():
    assert UnifiedQwenDecoder._resolve_cross_attention_layers([-1, -3, 2], 6) == (2, 3, 5)
    with pytest.raises(ValueError, match="outside"):
        UnifiedQwenDecoder._resolve_cross_attention_layers([-7], 6)
    with pytest.raises(ValueError, match="unique"):
        UnifiedQwenDecoder._resolve_cross_attention_layers([1, -5], 6)


def test_unified_decoder_runs_with_inserted_cross_attention():
    from transformers import Qwen2Config

    config = Qwen2Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
    )
    decoder = UnifiedQwenDecoder(
        visual_hidden_size=16,
        cross_attention_layers=[-1],
        cross_attention_heads=4,
        config=config,
    )
    output = decoder(
        input_ids=torch.tensor([[1, 2, 3]]),
        visual_tokens=torch.randn(1, 4, 16),
        mode="ar",
    )
    assert output.logits.shape == (1, 3, 32)
    dual = decoder.forward_dual_stream(
        clean_input_ids=torch.tensor([[1, 2, 3]]),
        noisy_block_ids=torch.tensor([[4, 5]]),
        block_starts=torch.tensor([1]),
        visual_tokens=torch.randn(1, 4, 16),
        timestep=torch.tensor([1]),
    )
    assert dual.ar_logits.shape == (1, 3, 32)
    assert dual.diffusion_logits.shape == (1, 2, 32)


def test_pooled_visual_prefix_keeps_full_cross_attention_memory():
    from transformers import Qwen2Config

    config = Qwen2Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
    )
    decoder = UnifiedQwenDecoder(
        visual_hidden_size=16,
        visual_prefix_mode="pooled",
        cross_attention_layers=[-1],
        cross_attention_heads=4,
        config=config,
    )
    visual = torch.randn(1, 7, 16)
    projected = decoder.visual_proj(visual)
    prefix = decoder._build_visual_prefix(projected)
    assert prefix.shape == (1, 1, 16)
    assert torch.allclose(prefix, projected.mean(dim=1, keepdim=True))

    output = decoder(
        input_ids=torch.tensor([[1, 2, 3]]),
        visual_tokens=visual,
        mode="ar",
    )
    assert output.logits.shape == (1, 3, 32)


def test_visual_prefix_mode_is_validated():
    from transformers import Qwen2Config

    with pytest.raises(ValueError, match="visual_prefix_mode"):
        UnifiedQwenDecoder(
            visual_hidden_size=16,
            visual_prefix_mode="invalid",
            config=Qwen2Config(
                vocab_size=32,
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=1,
                num_attention_heads=4,
                num_key_value_heads=2,
            ),
        )

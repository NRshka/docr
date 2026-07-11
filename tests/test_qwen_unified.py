import torch
from transformers import Qwen2Config

from docr.models.decoder import OCRModel
from docr.models.diffusion import DiscreteDiffusionSchedule
from docr.models.qwen_unified import (
    UnifiedQwenDecoder,
    build_dual_stream_allowed_mask,
    build_unified_qwen_allowed_mask,
)
from docr.models.vision_encoder import TinyVisionEncoder
from docr.training.trainer import OCRLightningModule


def tiny_qwen_config() -> Qwen2Config:
    return Qwen2Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )


def test_ar_mask_blocks_future_text_and_allows_visual_prefix():
    mask = build_unified_qwen_allowed_mask(
        mode="ar",
        num_visual_tokens=2,
        text_length=4,
    )

    assert mask[2, 0]
    assert mask[2, 1]
    assert mask[2, 2]
    assert not mask[2, 3]
    assert not mask[2, 5]
    assert mask[5, 4]
    assert mask[5, 5]


def test_diffusion_mask_allows_same_canvas_and_blocks_cross_canvas():
    mask = build_unified_qwen_allowed_mask(
        mode="diffusion",
        num_visual_tokens=2,
        text_length=6,
        num_canvases=2,
        canvas_length=3,
    )

    assert mask[2, 0]
    assert mask[2, 1]
    assert mask[2, 4]
    assert mask[4, 2]
    assert not mask[2, 5]
    assert not mask[5, 2]
    assert mask[5, 7]
    assert mask[7, 5]


def test_unified_qwen_ar_and_diffusion_forward_shapes():
    decoder = UnifiedQwenDecoder(
        visual_hidden_size=16,
        timesteps=4,
        num_canvases=2,
        canvas_length=3,
        config=tiny_qwen_config(),
    )
    visual_tokens = torch.randn(2, 3, 16)
    input_ids = torch.tensor([[1, 2, 3, 4, 0, 0], [5, 6, 7, 8, 9, 10]])
    attention_mask = torch.tensor(
        [[True, True, True, True, False, False], [True, True, True, True, True, True]]
    )

    ar_output = decoder(
        input_ids=input_ids,
        visual_tokens=visual_tokens,
        attention_mask=attention_mask,
        mode="ar",
    )
    diffusion_output = decoder(
        input_ids=input_ids,
        visual_tokens=visual_tokens,
        attention_mask=attention_mask,
        timestep=torch.tensor([1, 2]),
        mode="diffusion",
    )

    assert ar_output.logits.shape == (2, 6, 32)
    assert diffusion_output.logits.shape == (2, 6, 32)


def test_joint_lightning_step_with_unified_qwen():
    vision = TinyVisionEncoder(
        patch_size=8,
        hidden_size=16,
        output_tokens=3,
        compressor_layers=1,
        compressor_heads=4,
    )
    decoder = UnifiedQwenDecoder(
        visual_hidden_size=16,
        timesteps=4,
        num_canvases=1,
        canvas_length=6,
        config=tiny_qwen_config(),
    )
    module = OCRLightningModule(
        model=OCRModel(vision, decoder),
        mode="joint",
        diffusion_schedule=DiscreteDiffusionSchedule(timesteps=4, min_mask_ratio=0.5),
        mask_token_id=31,
        special_token_ids={0},
        ar_loss_weight=0.5,
        diffusion_loss_weight=1.0,
    )
    batch = {
        "images": torch.randn(2, 3, 32, 32),
        "input_ids": torch.tensor([[1, 2, 3, 0, 0, 0], [4, 5, 6, 7, 0, 0]]),
        "attention_mask": torch.tensor(
            [[True, True, True, False, False, False], [True, True, True, True, False, False]]
        ),
    }

    loss, metrics = module._joint_step(
        batch["images"],
        batch["input_ids"],
        batch["attention_mask"],
    )

    assert loss.requires_grad
    assert "train/loss_ar" in metrics
    assert "train/loss_diffusion" in metrics
    assert "train/loss_diffusion_weighted" in metrics


def test_dual_stream_mask_blocks_target_and_suffix_leakage():
    mask = build_dual_stream_allowed_mask(
        num_visual_tokens=2,
        clean_attention_mask=torch.tensor([[True, True, True, True, True, False]]),
        block_starts=torch.tensor([2]),
        noisy_block_mask=torch.tensor([[True, True, True]]),
    )[0]
    clean_start = 2
    noisy_start = 2 + 6

    assert mask[noisy_start, 0]
    assert mask[noisy_start, clean_start]
    assert mask[noisy_start, clean_start + 1]
    assert not mask[noisy_start, clean_start + 2]
    assert not mask[noisy_start, clean_start + 4]
    assert mask[noisy_start, noisy_start + 2]
    assert mask[noisy_start + 2, noisy_start]
    assert not mask[clean_start, noisy_start]


def test_dual_stream_ar_matches_causal_forward_and_suffix_does_not_leak():
    torch.manual_seed(3)
    decoder = UnifiedQwenDecoder(
        visual_hidden_size=16,
        timesteps=4,
        config=tiny_qwen_config(),
    ).eval()
    visual_tokens = torch.randn(1, 3, 16)
    clean_ids = torch.tensor([[1, 4, 5, 6, 7, 2]])
    clean_mask = torch.ones_like(clean_ids, dtype=torch.bool)
    noisy_ids = torch.tensor([[31, 31]])
    block_starts = torch.tensor([2])
    block_mask = torch.ones_like(noisy_ids, dtype=torch.bool)

    causal = decoder(
        input_ids=clean_ids,
        visual_tokens=visual_tokens,
        attention_mask=clean_mask,
        mode="ar",
    )
    dual = decoder.forward_dual_stream(
        clean_input_ids=clean_ids,
        noisy_block_ids=noisy_ids,
        block_starts=block_starts,
        visual_tokens=visual_tokens,
        clean_attention_mask=clean_mask,
        noisy_block_mask=block_mask,
        timestep=torch.tensor([2]),
    )
    changed_suffix = clean_ids.clone()
    changed_suffix[:, 2:] = torch.tensor([[9, 10, 11, 12]])
    suffix_dual = decoder.forward_dual_stream(
        clean_input_ids=changed_suffix,
        noisy_block_ids=noisy_ids,
        block_starts=block_starts,
        visual_tokens=visual_tokens,
        clean_attention_mask=clean_mask,
        noisy_block_mask=block_mask,
        timestep=torch.tensor([2]),
    )

    assert torch.allclose(dual.ar_logits, causal.logits, atol=1e-5, rtol=1e-5)
    assert torch.allclose(
        dual.diffusion_logits,
        suffix_dual.diffusion_logits,
        atol=1e-5,
        rtol=1e-5,
    )


def test_dual_stream_noisy_block_uses_clean_prefix():
    torch.manual_seed(5)
    decoder = UnifiedQwenDecoder(
        visual_hidden_size=16,
        timesteps=4,
        config=tiny_qwen_config(),
    ).eval()
    visual_tokens = torch.randn(1, 2, 16)
    kwargs = {
        "noisy_block_ids": torch.tensor([[31, 31]]),
        "block_starts": torch.tensor([2]),
        "visual_tokens": visual_tokens,
        "clean_attention_mask": torch.ones(1, 5, dtype=torch.bool),
        "noisy_block_mask": torch.ones(1, 2, dtype=torch.bool),
        "timestep": torch.tensor([1]),
    }
    first = decoder.forward_dual_stream(
        clean_input_ids=torch.tensor([[1, 4, 5, 6, 2]]), **kwargs
    )
    second = decoder.forward_dual_stream(
        clean_input_ids=torch.tensor([[1, 8, 5, 6, 2]]), **kwargs
    )

    assert not torch.allclose(first.diffusion_logits, second.diffusion_logits)


def test_tri_mode_joint_step_uses_block_diffusion():
    vision = TinyVisionEncoder(
        patch_size=8,
        hidden_size=16,
        output_tokens=3,
        compressor_layers=1,
        compressor_heads=4,
    )
    decoder = UnifiedQwenDecoder(
        visual_hidden_size=16,
        timesteps=4,
        config=tiny_qwen_config(),
    )
    module = OCRLightningModule(
        model=OCRModel(vision, decoder),
        mode="tri_mode_joint",
        diffusion_schedule=DiscreteDiffusionSchedule(timesteps=4, min_mask_ratio=0.5),
        mask_token_id=31,
        special_token_ids={0},
        ar_loss_weight=1.0,
        diffusion_loss_weight=0.3,
        diffusion_block_length=3,
    )
    images = torch.randn(2, 3, 32, 32)
    ids = torch.tensor([[1, 3, 4, 5, 0], [1, 6, 7, 8, 2]])
    attention = ids != 0

    loss, metrics = module._tri_mode_step(
        images,
        ids,
        attention,
        generator=torch.Generator().manual_seed(11),
    )

    expected = metrics["train/loss_ar"] + 0.3 * metrics["train/loss_diffusion"]
    assert loss.requires_grad
    assert torch.allclose(metrics["train/loss"], expected)
    assert metrics["train/block_length"] == 3
    assert metrics["train/masked_tokens"] > 0


def test_validation_generator_is_independent_of_global_rng():
    vision = TinyVisionEncoder(
        patch_size=8,
        hidden_size=16,
        output_tokens=3,
        compressor_layers=1,
        compressor_heads=4,
    )
    decoder = UnifiedQwenDecoder(
        visual_hidden_size=16,
        timesteps=4,
        config=tiny_qwen_config(),
    )
    module = OCRLightningModule(model=OCRModel(vision, decoder))
    generator_a = module._validation_generator(3, torch.device("cpu"))
    _ = torch.rand(100)
    generator_b = module._validation_generator(3, torch.device("cpu"))

    assert torch.equal(torch.rand(10, generator=generator_a), torch.rand(10, generator=generator_b))

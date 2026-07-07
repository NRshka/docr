import torch

from docr.models.decoder import DiffusionTransformerDecoder, OCRModel
from docr.models.diffusion import DiscreteDiffusionSchedule
from docr.models.vision_encoder import TinyVisionEncoder
from docr.training.trainer import OCRLightningModule


def test_diffusion_training_step_logs_mask_metrics():
    vision = TinyVisionEncoder(
        patch_size=8,
        hidden_size=16,
        output_tokens=3,
        compressor_layers=1,
        compressor_heads=4,
    )
    decoder = DiffusionTransformerDecoder(
        vocab_size=32,
        hidden_size=16,
        visual_hidden_size=16,
        num_layers=1,
        num_heads=4,
        max_length=6,
        timesteps=4,
    )
    module = OCRLightningModule(
        model=OCRModel(vision, decoder),
        mode="diffusion",
        diffusion_schedule=DiscreteDiffusionSchedule(timesteps=4, min_mask_ratio=0.5),
        mask_token_id=31,
        special_token_ids={0},
        probe_interval=1,
        probe_timesteps=[1, 3],
        probe_visual_ablations=["normal", "blank"],
    )
    batch = {
        "images": torch.randn(2, 3, 32, 32),
        "input_ids": torch.tensor([[1, 2, 3, 0, 0, 0], [4, 5, 6, 7, 0, 0]]),
        "attention_mask": torch.tensor(
            [[True, True, True, False, False, False], [True, True, True, True, False, False]]
        ),
    }
    loss, metrics = module._diffusion_step(
        batch["images"],
        batch["input_ids"],
        batch["attention_mask"],
    )
    assert loss.requires_grad
    assert "train/loss" in metrics
    assert "train/loss_diffusion" in metrics
    assert "train/diffusion_timestep" in metrics
    assert "train/diffusion_mask_ratio" in metrics
    assert "train/masked_tokens" in metrics
    assert "train/masked_token_fraction" in metrics
    assert "probe/normal_loss_t01" in metrics
    assert "probe/blank_loss_t01" in metrics
    assert "probe/blank_delta_t01" in metrics
    assert metrics["train/masked_tokens"] > 0


def test_diffusion_validation_logs_denoising_diagnostics():
    vision = TinyVisionEncoder(
        patch_size=8,
        hidden_size=16,
        output_tokens=3,
        compressor_layers=1,
        compressor_heads=4,
    )
    decoder = DiffusionTransformerDecoder(
        vocab_size=32,
        hidden_size=16,
        visual_hidden_size=16,
        num_layers=1,
        num_heads=4,
        max_length=6,
        timesteps=4,
    )
    module = OCRLightningModule(
        model=OCRModel(vision, decoder),
        mode="diffusion",
        diffusion_schedule=DiscreteDiffusionSchedule(timesteps=4, min_mask_ratio=0.5),
        mask_token_id=31,
        special_token_ids={0},
        validation_probe_timesteps=[1],
        validation_visual_ablations=["normal", "blank"],
        log_to_logger=False,
    )
    batch = {
        "images": torch.randn(2, 3, 32, 32),
        "input_ids": torch.tensor([[1, 2, 3, 0, 0, 0], [4, 5, 6, 7, 0, 0]]),
        "attention_mask": torch.tensor(
            [[True, True, True, False, False, False], [True, True, True, True, False, False]]
        ),
    }

    loss = module.validation_step(batch, batch_idx=0)

    assert loss.requires_grad
    assert "val/token_acc_diffusion_masked" in module.last_val_metrics
    assert "val/diffusion_loss_normal_t01" in module.last_val_metrics
    assert "val/denoise_acc_normal_t01" in module.last_val_metrics
    assert "val/visual_ablation_delta_blank_t01" in module.last_val_metrics

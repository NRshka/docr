import torch

from docr.models.decoder import DiffusionTransformerDecoder, OCRModel
from docr.models.diffusion import DiscreteDiffusionSchedule
from docr.models.vision_encoder import TinyVisionEncoder
from docr.training.trainer import OCRTrainer


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
    trainer = OCRTrainer(
        model=OCRModel(vision, decoder),
        mode="diffusion",
        diffusion_schedule=DiscreteDiffusionSchedule(timesteps=4, min_mask_ratio=0.5),
        mask_token_id=31,
        special_token_ids={0},
    )
    batch = {
        "images": torch.randn(2, 3, 32, 32),
        "input_ids": torch.tensor([[1, 2, 3, 0, 0, 0], [4, 5, 6, 7, 0, 0]]),
        "attention_mask": torch.tensor(
            [[True, True, True, False, False, False], [True, True, True, True, False, False]]
        ),
    }
    metrics = trainer.train_step(batch)
    assert trainer.state.step == 1
    assert "train/loss" in metrics
    assert "train/diffusion_timestep" in metrics
    assert "train/diffusion_mask_ratio" in metrics
    assert "train/masked_tokens" in metrics
    assert metrics["train/masked_tokens"] > 0

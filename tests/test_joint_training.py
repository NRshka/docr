import torch

from docr.models.decoder import OCRModel, TinyTextDecoder
from docr.models.diffusion import DiscreteDiffusionSchedule
from docr.models.vision_encoder import TinyVisionEncoder
from docr.training.trainer import OCRLightningModule


def test_joint_training_step_logs_ar_and_diffusion_losses():
    vision = TinyVisionEncoder(
        patch_size=8,
        hidden_size=16,
        output_tokens=3,
        compressor_layers=1,
        compressor_heads=4,
    )
    decoder = TinyTextDecoder(vocab_size=32, hidden_size=16)
    module = OCRLightningModule(
        model=OCRModel(vision, decoder),
        mode="joint",
        diffusion_schedule=DiscreteDiffusionSchedule(timesteps=4, min_mask_ratio=0.5),
        mask_token_id=31,
        special_token_ids={0},
        ar_loss_weight=0.25,
        diffusion_loss_weight=1.5,
        probe_interval=1,
        probe_timesteps=[1],
        probe_visual_ablations=["normal"],
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

    expected = 0.25 * metrics["train/loss_ar"] + 1.5 * metrics["train/loss_diffusion"]
    assert loss.requires_grad
    assert torch.allclose(metrics["train/loss"], expected)
    assert "train/loss_ar_weighted" in metrics
    assert "train/loss_diffusion_weighted" in metrics
    assert "train/diffusion_timestep" in metrics
    assert "train/masked_token_fraction" in metrics
    assert "probe/normal_loss_t01" in metrics

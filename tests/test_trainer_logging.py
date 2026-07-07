import torch

from docr.training.trainer import OCRLightningModule


class TinyTrainModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(3, 8)

    def forward(self, images, input_ids, attention_mask=None, mode="ar"):
        del attention_mask, mode
        pooled = images.mean(dim=(2, 3)).unsqueeze(1)
        logits = self.proj(pooled).expand(input_ids.shape[0], input_ids.shape[1], 8)
        return type("Output", (), {"logits": logits})


def test_lightning_module_computes_ar_loss_metrics():
    module = OCRLightningModule(
        model=TinyTrainModel(),
        learning_rate=1e-3,
    )
    batch = {
        "images": torch.randn(2, 3, 4, 4),
        "input_ids": torch.tensor([[1, 2, 0], [3, 0, 0]]),
        "attention_mask": torch.tensor([[True, True, False], [True, False, False]]),
    }
    loss, metrics = module._ar_step(batch["images"], batch["input_ids"], batch["attention_mask"])
    assert loss.requires_grad
    assert "train/loss" in metrics
    assert "train/loss_ar" in metrics
    assert "train/text_tokens" in metrics
    assert "train/tokens_per_sample" in metrics
    assert float(metrics["train/text_tokens"]) == 3.0


def test_lightning_module_logs_validation_metrics():
    module = OCRLightningModule(
        model=TinyTrainModel(),
        learning_rate=1e-3,
        log_to_logger=False,
    )
    batch = {
        "images": torch.randn(2, 3, 4, 4),
        "input_ids": torch.tensor([[1, 2, 0], [3, 0, 0]]),
        "attention_mask": torch.tensor([[True, True, False], [True, False, False]]),
    }

    loss = module.validation_step(batch, batch_idx=0)

    assert loss.requires_grad
    assert "val/loss" in module.last_val_metrics
    assert "val/loss_ar" in module.last_val_metrics
    assert "val/text_tokens" in module.last_val_metrics
    assert module.last_val_metrics["val/text_tokens"] == 3.0

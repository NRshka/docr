import torch

from docr.training.trainer import OCRTrainer


class TinyTrainModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(3, 8)

    def forward(self, images, input_ids, attention_mask=None, mode="ar"):
        del attention_mask, mode
        pooled = images.mean(dim=(2, 3)).unsqueeze(1)
        logits = self.proj(pooled).expand(input_ids.shape[0], input_ids.shape[1], 8)
        return type("Output", (), {"logits": logits})


class MemoryLogger:
    def __init__(self):
        self.calls = []

    def log(self, metrics, step):
        self.calls.append((step, metrics))


def test_trainer_logs_loss_metrics_at_interval():
    logger = MemoryLogger()
    trainer = OCRTrainer(
        model=TinyTrainModel(),
        learning_rate=1e-3,
        logger=logger,
        log_interval=1,
    )
    batch = {
        "images": torch.randn(2, 3, 4, 4),
        "input_ids": torch.tensor([[1, 2, 0], [3, 0, 0]]),
        "attention_mask": torch.tensor([[True, True, False], [True, False, False]]),
    }
    metrics = trainer.train_step(batch)
    assert "train/loss" in metrics
    assert "train/lr" in metrics
    assert "train/text_tokens" in metrics
    assert metrics["train/text_tokens"] == 3.0
    assert logger.calls[0][0] == 1
    assert "train/loss" in logger.calls[0][1]

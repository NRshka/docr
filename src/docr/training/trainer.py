from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
from typing import Any

import torch
from torch.optim import AdamW

from docr.models.diffusion import DiscreteDiffusionSchedule, corrupt_with_mask
from docr.training.losses import diffusion_denoising_loss
from docr.training.losses import language_model_loss


@dataclass
class TrainState:
    step: int = 0


class OCRTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        device: str | torch.device = "cpu",
        logger: Any | None = None,
        log_interval: int = 10,
        mode: str = "ar",
        diffusion_schedule: DiscreteDiffusionSchedule | None = None,
        mask_token_id: int | None = None,
        special_token_ids: set[int] | None = None,
    ) -> None:
        self.model = model.to(device)
        self.device = torch.device(device)
        self.optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.logger = logger
        self.log_interval = log_interval
        self.mode = mode
        self.diffusion_schedule = diffusion_schedule
        self.mask_token_id = mask_token_id
        self.special_token_ids = special_token_ids or set()
        self.state = TrainState()

    def train_step(self, batch: dict) -> dict[str, float]:
        self.model.train()
        images = batch["images"].to(self.device)
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        if self.mode == "diffusion":
            return self._diffusion_train_step(images, input_ids, attention_mask)
        return self._ar_train_step(images, input_ids, attention_mask)

    def _ar_train_step(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> dict[str, float]:
        labels = input_ids.clone()
        if attention_mask is not None:
            labels = labels.masked_fill(~attention_mask, -100)
        output = self.model(
            images=images,
            input_ids=input_ids,
            attention_mask=attention_mask,
            mode="ar",
        )
        loss = language_model_loss(output.logits, labels, ignore_index=-100)
        return self._finish_step(loss, images=images, labels=labels)

    def _diffusion_train_step(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> dict[str, float]:
        if self.diffusion_schedule is None or self.mask_token_id is None:
            raise ValueError("Diffusion training requires diffusion_schedule and mask_token_id")

        timestep_value = int(
            torch.randint(
                low=1,
                high=max(self.diffusion_schedule.timesteps, 2),
                size=(1,),
                device=self.device,
            ).item()
        )
        corrupted, prediction_mask = corrupt_with_mask(
            input_ids,
            timestep=timestep_value,
            schedule=self.diffusion_schedule,
            mask_token_id=self.mask_token_id,
            special_token_ids=self.special_token_ids,
        )
        if attention_mask is not None:
            prediction_mask &= attention_mask.bool()
        if not prediction_mask.any():
            prediction_mask = attention_mask.bool() if attention_mask is not None else torch.ones_like(input_ids).bool()

        timestep = torch.full((input_ids.shape[0],), timestep_value, dtype=torch.long, device=self.device)
        output = self.model(
            images=images,
            input_ids=corrupted,
            attention_mask=attention_mask,
            timestep=timestep,
            mode="diffusion",
        )
        loss = diffusion_denoising_loss(output.logits, input_ids, prediction_mask)
        metrics = self._finish_step(loss, images=images, labels=input_ids, log_now=False)
        metrics.update(
            {
                "train/diffusion_timestep": float(timestep_value),
                "train/diffusion_mask_ratio": float(self.diffusion_schedule.mask_ratio(timestep_value)),
                "train/masked_tokens": float(prediction_mask.sum().detach().cpu()),
            }
        )
        self._log_metrics(metrics)
        return metrics

    def _finish_step(
        self,
        loss: torch.Tensor,
        images: torch.Tensor,
        labels: torch.Tensor,
        log_now: bool = True,
    ) -> dict[str, float]:
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.state.step += 1
        metrics = {
            "train/loss": float(loss.detach().cpu()),
            "train/lr": float(self.optimizer.param_groups[0]["lr"]),
            "train/batch_size": float(images.shape[0]),
            "train/text_tokens": float((labels != -100).sum().detach().cpu()),
        }
        if log_now:
            self._log_metrics(metrics)
        return metrics

    def _log_metrics(self, metrics: dict[str, float]) -> None:
        if self.logger is not None and self.state.step % self.log_interval == 0:
            self.logger.log(metrics, step=self.state.step)

    def fit(self, batches: Iterable[dict], max_steps: int) -> None:
        while self.state.step < max_steps:
            progressed = False
            for batch in batches:
                metrics = self.train_step(batch)
                if self.state.step % self.log_interval == 0:
                    print(
                        f"step={self.state.step} "
                        f"loss={metrics['train/loss']:.4f} "
                        f"lr={metrics['train/lr']:.2e}"
                    )
                progressed = True
                if self.state.step >= max_steps:
                    break
            if not progressed:
                raise ValueError("Training loader produced no batches")

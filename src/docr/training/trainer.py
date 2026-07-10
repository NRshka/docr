from __future__ import annotations

from typing import Any

import lightning as L
import torch
from torch.optim import AdamW

from docr.evaluation.metrics import ocr_metrics
from docr.models.diffusion import DiscreteDiffusionSchedule, corrupt_with_mask
from docr.training.losses import diffusion_denoising_loss, language_model_loss
from docr.training.schedules import build_scheduler


class OCRLightningModule(L.LightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
        learning_rate: float = 1e-4,
        pretrained_learning_rate: float | None = None,
        weight_decay: float = 0.01,
        scheduler_name: str = "constant",
        warmup_steps: int = 0,
        max_steps: int = 1000,
        mode: str = "ar",
        diffusion_schedule: DiscreteDiffusionSchedule | None = None,
        mask_token_id: int | None = None,
        special_token_ids: set[int] | None = None,
        probe_interval: int = 0,
        probe_timesteps: list[int] | None = None,
        probe_visual_ablations: list[str] | None = None,
        ar_loss_weight: float = 1.0,
        diffusion_loss_weight: float = 1.0,
        tokenizer: Any | None = None,
        validation_probe_timesteps: list[int] | None = None,
        validation_visual_ablations: list[str] | None = None,
        log_to_logger: bool = True,
    ) -> None:
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.pretrained_learning_rate = (
            learning_rate if pretrained_learning_rate is None else pretrained_learning_rate
        )
        self.weight_decay = weight_decay
        self.scheduler_name = scheduler_name
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.mode = mode
        self.diffusion_schedule = diffusion_schedule
        self.mask_token_id = mask_token_id
        self.special_token_ids = special_token_ids or set()
        self.probe_interval = probe_interval
        self.probe_timesteps = probe_timesteps or []
        self.probe_visual_ablations = probe_visual_ablations or []
        self.ar_loss_weight = ar_loss_weight
        self.diffusion_loss_weight = diffusion_loss_weight
        self.tokenizer = tokenizer
        self.validation_probe_timesteps = validation_probe_timesteps or []
        self.validation_visual_ablations = validation_visual_ablations or []
        self.log_to_logger = log_to_logger
        self.last_metrics: dict[str, float] = {}
        self.last_val_metrics: dict[str, float] = {}
        self.last_batch_diagnostics = "unavailable"
        self.save_hyperparameters(ignore=["model", "diffusion_schedule", "special_token_ids"])

    def configure_optimizers(self):
        pretrained = []
        new_modules = []
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad:
                continue
            if ".decoder.lm." in name or ".vision_encoder.backbone." in name:
                pretrained.append(parameter)
            else:
                new_modules.append(parameter)

        parameter_groups = []
        if pretrained:
            parameter_groups.append(
                {"params": pretrained, "lr": self.pretrained_learning_rate, "name": "pretrained"}
            )
        if new_modules:
            parameter_groups.append(
                {"params": new_modules, "lr": self.learning_rate, "name": "new_modules"}
            )
        optimizer = AdamW(parameter_groups, weight_decay=self.weight_decay)
        scheduler = build_scheduler(
            optimizer,
            name=self.scheduler_name,
            max_steps=self.max_steps,
            warmup_steps=self.warmup_steps,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        images = batch["images"]
        input_ids = batch["input_ids"]
        attention_mask = batch.get("attention_mask", None)
        valid_lengths = (
            attention_mask.sum(dim=1).detach().cpu().tolist()
            if attention_mask is not None
            else [input_ids.shape[1]] * input_ids.shape[0]
        )
        self.last_batch_diagnostics = (
            f"batch_idx={batch_idx} image_shape={tuple(images.shape)} "
            f"text_shape={tuple(input_ids.shape)} valid_lengths={valid_lengths} "
            f"doc_ids={batch.get('doc_ids', [])}"
        )
        if self.mode == "joint":
            loss, metrics = self._joint_step(images, input_ids, attention_mask)
        elif self.mode == "diffusion":
            loss, metrics = self._diffusion_step(images, input_ids, attention_mask)
        else:
            loss, metrics = self._ar_step(images, input_ids, attention_mask)
        self._require_finite(loss, "training loss")
        self._log_train_metrics(metrics, batch_size=images.shape[0])
        return loss

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        del batch_idx
        images = batch["images"]
        input_ids = batch["input_ids"]
        attention_mask = batch.get("attention_mask", None)
        if self.mode == "joint":
            loss, metrics = self._joint_step(images, input_ids, attention_mask, include_probes=False)
        elif self.mode == "diffusion":
            loss, metrics = self._diffusion_step(
                images,
                input_ids,
                attention_mask,
                include_probes=False,
            )
        else:
            loss, metrics = self._ar_step(images, input_ids, attention_mask)
        self._require_finite(loss, "validation loss")
        metrics.update(self._validation_quality_metrics(images, input_ids, attention_mask))
        val_metrics = self._with_metric_prefix(metrics, "val")
        self._log_val_metrics(val_metrics, batch_size=images.shape[0])
        return loss

    def on_after_backward(self) -> None:
        for name, parameter in self.named_parameters():
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                raise FloatingPointError(
                    f"Non-finite gradient at global_step={self.global_step} parameter={name}; "
                    f"{self.last_batch_diagnostics}"
                )

    def on_before_optimizer_step(self, optimizer) -> None:
        del optimizer
        gradients = [
            parameter.grad.detach().float().norm(2)
            for parameter in self.parameters()
            if parameter.grad is not None
        ]
        if not gradients:
            return
        grad_norm = torch.stack(gradients).norm(2)
        self._require_finite(grad_norm, "gradient norm")
        self.log(
            "train/grad_norm",
            grad_norm,
            on_step=True,
            on_epoch=False,
            logger=self.log_to_logger,
            sync_dist=True,
        )

    def _require_finite(self, value: torch.Tensor, label: str) -> None:
        if not torch.isfinite(value).all():
            raise FloatingPointError(f"Non-finite {label} at global_step={self.global_step}")

    def _ar_step(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        visual_tokens: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        labels = input_ids.clone()
        if attention_mask is not None:
            labels = labels.masked_fill(~attention_mask.bool(), -100)
        if visual_tokens is None:
            output = self.model(
                images=images,
                input_ids=input_ids,
                attention_mask=attention_mask,
                mode="ar",
            )
        else:
            output = self.model.decode_text(
                input_ids=input_ids,
                visual_tokens=visual_tokens,
                attention_mask=attention_mask,
                mode="ar",
            )
        loss = language_model_loss(output.logits, labels, ignore_index=-100)
        metrics = self._base_metrics(loss, images=images, labels=labels)
        metrics["train/loss_ar"] = loss.detach()
        metrics["train/token_acc_ar"] = self._token_accuracy(output.logits, labels)
        return loss, metrics

    def _diffusion_step(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        visual_tokens: torch.Tensor | None = None,
        include_probes: bool = True,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if self.diffusion_schedule is None or self.mask_token_id is None:
            raise ValueError("Diffusion training requires diffusion_schedule and mask_token_id")

        timestep_value = int(
            torch.randint(
                low=1,
                high=max(self.diffusion_schedule.timesteps, 2),
                size=(1,),
                device=input_ids.device,
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
            prediction_mask = (
                attention_mask.bool() if attention_mask is not None else torch.ones_like(input_ids).bool()
            )

        timestep = torch.full(
            (input_ids.shape[0],),
            timestep_value,
            dtype=torch.long,
            device=input_ids.device,
        )
        if visual_tokens is None:
            output = self.model(
                images=images,
                input_ids=corrupted,
                attention_mask=attention_mask,
                timestep=timestep,
                mode="diffusion",
            )
        else:
            output = self.model.decode_text(
                input_ids=corrupted,
                visual_tokens=visual_tokens,
                attention_mask=attention_mask,
                timestep=timestep,
                mode="diffusion",
            )
        loss = diffusion_denoising_loss(output.logits, input_ids, prediction_mask)
        labels = input_ids.masked_fill(~prediction_mask, -100)
        metrics = self._base_metrics(loss, images=images, labels=labels)
        text_tokens = (
            attention_mask.sum()
            if attention_mask is not None
            else torch.tensor(input_ids.numel(), device=input_ids.device)
        ).float()
        masked_tokens = prediction_mask.sum().float()
        metrics.update(
            {
                "train/loss_diffusion": loss.detach(),
                "train/token_acc_diffusion_masked": self._token_accuracy(output.logits, labels),
                "train/diffusion_timestep": torch.tensor(float(timestep_value), device=input_ids.device),
                "train/diffusion_mask_ratio": torch.tensor(
                    float(self.diffusion_schedule.mask_ratio(timestep_value)),
                    device=input_ids.device,
                ),
                "train/masked_tokens": masked_tokens.detach(),
                "train/masked_token_fraction": (masked_tokens / text_tokens.clamp_min(1.0)).detach(),
                "train/text_tokens": text_tokens.detach(),
            }
        )
        if include_probes and self._should_probe():
            metrics.update(self._diffusion_probe_metrics(images, input_ids, attention_mask, visual_tokens))
        return loss, metrics

    def _joint_step(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        include_probes: bool = True,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        visual_tokens = self.model.encode_images(images)
        ar_loss, ar_metrics = self._ar_step(
            images,
            input_ids,
            attention_mask,
            visual_tokens=visual_tokens,
        )
        diffusion_loss, diffusion_metrics = self._diffusion_step(
            images,
            input_ids,
            attention_mask,
            visual_tokens=visual_tokens,
            include_probes=False,
        )
        weighted_ar = self.ar_loss_weight * ar_loss
        weighted_diffusion = self.diffusion_loss_weight * diffusion_loss
        loss = weighted_ar + weighted_diffusion
        metrics = {**diffusion_metrics}
        metrics.update(
            {
                "train/loss": loss.detach(),
                "train/loss_ar": ar_loss.detach(),
                "train/loss_diffusion": diffusion_loss.detach(),
                "train/loss_ar_weighted": weighted_ar.detach(),
                "train/loss_diffusion_weighted": weighted_diffusion.detach(),
                "train/token_acc_ar": ar_metrics["train/token_acc_ar"].detach(),
                "train/token_acc_diffusion_masked": diffusion_metrics[
                    "train/token_acc_diffusion_masked"
                ].detach(),
                "train/ar_loss_weight": torch.tensor(self.ar_loss_weight, device=loss.device),
                "train/diffusion_loss_weight": torch.tensor(
                    self.diffusion_loss_weight,
                    device=loss.device,
                ),
                "train/text_tokens_ar": ar_metrics["train/text_tokens"].detach(),
            }
        )
        if include_probes and self._should_probe():
            metrics.update(self._diffusion_probe_metrics(images, input_ids, attention_mask, visual_tokens))
        return loss, metrics

    def _base_metrics(
        self,
        loss: torch.Tensor,
        images: torch.Tensor,
        labels: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        text_tokens = (labels != -100).sum().float()
        return {
            "train/loss": loss.detach(),
            "train/batch_size": torch.tensor(float(images.shape[0]), device=loss.device),
            "train/text_tokens": text_tokens.detach(),
            "train/tokens_per_sample": (text_tokens / images.shape[0]).detach(),
        }

    def _token_accuracy(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        valid = labels != -100
        if not valid.any():
            return torch.tensor(0.0, device=logits.device)
        predictions = logits.argmax(dim=-1)
        return (predictions[valid] == labels[valid]).float().mean().detach()

    @torch.no_grad()
    def _validation_quality_metrics(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        metrics: dict[str, torch.Tensor] = {}
        if self.mode in {"ar", "joint"}:
            metrics.update(self._ar_ocr_metrics(images, input_ids, attention_mask))
        if self.diffusion_schedule is not None and self.mask_token_id is not None:
            metrics.update(self._validation_diffusion_probe_metrics(images, input_ids, attention_mask))
        return metrics

    @torch.no_grad()
    def _ar_ocr_metrics(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        output = self.model(
            images=images,
            input_ids=input_ids,
            attention_mask=attention_mask,
            mode="ar",
        )
        labels = input_ids.clone()
        if attention_mask is not None:
            labels = labels.masked_fill(~attention_mask.bool(), -100)
        predictions = output.logits.argmax(dim=-1)
        token_acc = self._token_accuracy(output.logits, labels)
        cer_values = []
        wer_values = []
        numeric_values = []
        exact_values = []
        normalized_edit_values = []
        for predicted_ids, target_ids, valid_mask in self._iter_valid_prediction_pairs(
            predictions,
            input_ids,
            attention_mask,
        ):
            prediction = self._decode_ids(predicted_ids)
            target = self._decode_ids(target_ids)
            sample_metrics = ocr_metrics(prediction, target)
            cer_values.append(sample_metrics["cer"])
            wer_values.append(sample_metrics["wer"])
            numeric_values.append(sample_metrics["numeric_exact_match"])
            exact_values.append(1.0 if prediction == target else 0.0)
            edit_denominator = max(len(prediction), len(target), 1)
            normalized_edit_values.append(sample_metrics["cer"] * len(target) / edit_denominator)
            del valid_mask

        device = input_ids.device
        return {
            "train/token_acc_ar_teacher_forced": token_acc,
            "train/cer_ar_teacher_forced": self._mean_float(cer_values, device),
            "train/wer_ar_teacher_forced": self._mean_float(wer_values, device),
            "train/numeric_exact_match_ar_teacher_forced": self._mean_float(numeric_values, device),
            "train/exact_match_ar_teacher_forced": self._mean_float(exact_values, device),
            "train/normalized_edit_ar_teacher_forced": self._mean_float(
                normalized_edit_values,
                device,
            ),
        }

    def _iter_valid_prediction_pairs(
        self,
        predictions: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        for predicted_ids, target_ids, valid_mask in zip(predictions, input_ids, attention_mask.bool()):
            yield predicted_ids[valid_mask], target_ids[valid_mask], valid_mask

    def _decode_ids(self, ids: torch.Tensor) -> str:
        values = [int(value) for value in ids.detach().cpu().tolist()]
        if self.tokenizer is not None:
            return self.tokenizer.decode(values, skip_special_tokens=True)
        byte_values = bytes(value for value in values if 0 <= value <= 255)
        return byte_values.decode("utf-8", errors="ignore")

    def _mean_float(self, values: list[float], device: torch.device) -> torch.Tensor:
        if not values:
            return torch.tensor(0.0, device=device)
        return torch.tensor(float(sum(values) / len(values)), device=device)

    @torch.no_grad()
    def _validation_diffusion_probe_metrics(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        if self.diffusion_schedule is None or self.mask_token_id is None:
            return {}
        timesteps = self.validation_probe_timesteps or self.probe_timesteps
        ablations = self.validation_visual_ablations or self.probe_visual_ablations or ["normal"]
        if not timesteps:
            timesteps = [self.diffusion_schedule.timesteps - 1]

        metrics: dict[str, torch.Tensor] = {}
        normal_losses: dict[int, torch.Tensor] = {}
        for timestep_value in timesteps:
            timestep_value = int(max(0, min(timestep_value, self.diffusion_schedule.timesteps - 1)))
            corrupted, prediction_mask = self._probe_corruption(input_ids, timestep_value)
            if attention_mask is not None:
                prediction_mask &= attention_mask.bool()
            if not prediction_mask.any():
                continue
            timestep = torch.full(
                (input_ids.shape[0],),
                timestep_value,
                dtype=torch.long,
                device=input_ids.device,
            )
            labels = input_ids.masked_fill(~prediction_mask, -100)
            for ablation in ablations:
                output = self.model(
                    images=self._ablate_images(images, ablation),
                    input_ids=corrupted,
                    attention_mask=attention_mask,
                    timestep=timestep,
                    mode="diffusion",
                )
                loss = diffusion_denoising_loss(output.logits, input_ids, prediction_mask).detach()
                metrics[f"train/diffusion_loss_{ablation}_t{timestep_value:02d}"] = loss
                metrics[f"train/denoise_acc_{ablation}_t{timestep_value:02d}"] = self._token_accuracy(
                    output.logits,
                    labels,
                )
                if ablation == "normal":
                    normal_losses[timestep_value] = loss
                elif timestep_value in normal_losses:
                    metrics[f"train/visual_ablation_delta_{ablation}_t{timestep_value:02d}"] = (
                        loss - normal_losses[timestep_value]
                    )
        return metrics

    def _log_train_metrics(self, metrics: dict[str, torch.Tensor], batch_size: int) -> None:
        optimizer = self.optimizers(use_pl_optimizer=False)
        if optimizer is not None:
            metrics["train/lr"] = torch.tensor(
                float(optimizer.param_groups[0]["lr"]),
                device=self.device,
            )
        self.last_metrics = {
            name: float(value.detach().cpu()) if isinstance(value, torch.Tensor) else float(value)
            for name, value in metrics.items()
        }
        self.log_dict(
            metrics,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=self.log_to_logger,
            sync_dist=True,
            batch_size=batch_size,
        )

    def _log_val_metrics(self, metrics: dict[str, torch.Tensor], batch_size: int) -> None:
        self.last_val_metrics = {
            name: float(value.detach().cpu()) if isinstance(value, torch.Tensor) else float(value)
            for name, value in metrics.items()
        }
        self.log_dict(
            metrics,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=self.log_to_logger,
            sync_dist=True,
            batch_size=batch_size,
        )

    def _with_metric_prefix(
        self,
        metrics: dict[str, torch.Tensor],
        prefix: str,
    ) -> dict[str, torch.Tensor]:
        return {
            name.replace("train/", f"{prefix}/", 1) if name.startswith("train/") else f"{prefix}/{name}": value
            for name, value in metrics.items()
        }

    def _should_probe(self) -> bool:
        step = int(self.global_step) + 1
        return (
            self.probe_interval > 0
            and step % self.probe_interval == 0
            and self.diffusion_schedule is not None
            and self.mask_token_id is not None
            and bool(self.probe_timesteps)
        )

    @torch.no_grad()
    def _diffusion_probe_metrics(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        visual_tokens: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if self.diffusion_schedule is None or self.mask_token_id is None:
            return {}

        was_training = self.model.training
        self.model.eval()
        metrics: dict[str, torch.Tensor] = {}
        normal_losses: dict[int, torch.Tensor] = {}

        for timestep_value in self.probe_timesteps:
            timestep_value = int(max(0, min(timestep_value, self.diffusion_schedule.timesteps - 1)))
            corrupted, prediction_mask = self._probe_corruption(input_ids, timestep_value)
            if attention_mask is not None:
                prediction_mask &= attention_mask.bool()
            if not prediction_mask.any():
                continue

            timestep = torch.full(
                (input_ids.shape[0],),
                timestep_value,
                dtype=torch.long,
                device=input_ids.device,
            )
            for ablation in self.probe_visual_ablations:
                if ablation == "normal" and visual_tokens is not None:
                    output = self.model.decode_text(
                        input_ids=corrupted,
                        visual_tokens=visual_tokens,
                        attention_mask=attention_mask,
                        timestep=timestep,
                        mode="diffusion",
                    )
                else:
                    probe_images = self._ablate_images(images, ablation)
                    output = self.model(
                        images=probe_images,
                        input_ids=corrupted,
                        attention_mask=attention_mask,
                        timestep=timestep,
                        mode="diffusion",
                    )
                loss = diffusion_denoising_loss(output.logits, input_ids, prediction_mask).detach()
                key = f"probe/{ablation}_loss_t{timestep_value:02d}"
                metrics[key] = loss
                if ablation == "normal":
                    normal_losses[timestep_value] = loss
                elif timestep_value in normal_losses:
                    metrics[f"probe/{ablation}_delta_t{timestep_value:02d}"] = (
                        loss - normal_losses[timestep_value]
                    )

        if was_training:
            self.model.train()
        return metrics

    def _probe_corruption(
        self,
        input_ids: torch.Tensor,
        timestep_value: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        generator = torch.Generator(device=input_ids.device)
        generator.manual_seed(10_000 + timestep_value)
        return corrupt_with_mask(
            input_ids,
            timestep=timestep_value,
            schedule=self.diffusion_schedule,
            mask_token_id=self.mask_token_id,
            special_token_ids=self.special_token_ids,
            generator=generator,
        )

    def _ablate_images(self, images: torch.Tensor, ablation: str) -> torch.Tensor:
        if ablation == "normal":
            return images
        if ablation == "blank":
            return torch.zeros_like(images)
        if ablation == "shuffled":
            if images.shape[0] == 1:
                return torch.zeros_like(images)
            return images.roll(shifts=1, dims=0)
        raise ValueError(f"Unknown visual ablation: {ablation}")


OCRTrainer = OCRLightningModule

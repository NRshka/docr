from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
os.environ.setdefault("WANDB_ERROR_REPORTING", "false")

import hydra
import lightning as L
import torch
from omegaconf import DictConfig
from omegaconf import OmegaConf
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader

from docr.data.collate import OCRCollator
from docr.data.dataset import HFCordV2OCRDataset, ManifestOCRDataset, SyntheticOCRDataset
from docr.models.factory import build_model
from docr.models.diffusion import DiscreteDiffusionSchedule
from docr.training.trainer import OCRLightningModule
from docr.utils.seed import seed_everything
from docr.utils.tokenizer import build_tokenizer, tokenizer_pad_id


def lightning_precision(precision: str) -> str:
    mapping = {
        "fp32": "32-true",
        "32": "32-true",
        "fp16": "16-mixed",
        "16": "16-mixed",
        "bf16": "bf16-mixed",
    }
    return mapping.get(str(precision), str(precision))


def build_lightning_logger(cfg: DictConfig):
    logging_cfg = cfg.logging
    if not bool(logging_cfg.enabled):
        print("wandb_logging=disabled")
        return False

    save_dir = Path(str(logging_cfg.get("dir", f"{cfg.output_dir}/wandb")))
    save_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("WANDB_DIR", str(save_dir))
    os.environ.setdefault("WANDB_CACHE_DIR", str(save_dir / "cache"))
    os.environ.setdefault("WANDB_CONFIG_DIR", str(save_dir / "config"))
    Path(os.environ["WANDB_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["WANDB_CONFIG_DIR"]).mkdir(parents=True, exist_ok=True)

    mode = "offline" if bool(logging_cfg.offline) else "online"
    print(
        "wandb_logging="
        f"{mode} entity={logging_cfg.entity} project={logging_cfg.project} "
        f"name={logging_cfg.run_name}"
    )
    return WandbLogger(
        project=str(logging_cfg.project),
        entity=str(logging_cfg.entity) if logging_cfg.entity is not None else None,
        name=str(logging_cfg.run_name) if logging_cfg.run_name is not None else None,
        tags=list(logging_cfg.get("tags", [])),
        save_dir=str(save_dir),
        dir=str(save_dir),
        offline=bool(logging_cfg.offline),
        config=OmegaConf.to_container(cfg, resolve=True),
    )


def build_dataset(cfg: DictConfig, tokenizer, split: str):
    image_size = tuple(cfg.model.image_size)
    if cfg.data.name == "synthetic":
        num_samples = cfg.data.get("num_train", 4) if split == "train" else cfg.data.get("num_val", 0)
        if int(num_samples) <= 0:
            return None
        return SyntheticOCRDataset(num_samples=num_samples, image_size=image_size)
    if cfg.data.name == "manifest":
        manifest_key = "train_manifest" if split == "train" else "val_manifest"
        manifest_path = Path(str(cfg.data.get(manifest_key, "")))
        if split != "train" and not manifest_path.exists():
            print(f"validation=disabled missing_manifest={manifest_path}")
            return None
        return ManifestOCRDataset(
            manifest_path=manifest_path,
            image_root=cfg.data.image_root,
            image_size=image_size,
            tokenizer=tokenizer,
            max_text_length=cfg.data.max_text_length,
        )
    if cfg.data.name == "cord_v2":
        hf_split = cfg.data.split if split == "train" else cfg.data.get("val_split", "validation")
        max_samples = cfg.data.max_samples if split == "train" else cfg.data.get("val_max_samples", cfg.data.max_samples)
        return HFCordV2OCRDataset(
            dataset_name=cfg.data.dataset_name,
            dataset_path=cfg.data.get("dataset_path", None),
            split=hf_split,
            image_size=tuple(cfg.data.image_size),
            target_mode=cfg.data.target_mode,
            load_from_disk=bool(cfg.data.get("load_from_disk", False)),
            streaming=bool(cfg.data.streaming),
            max_samples=max_samples,
            tokenizer=tokenizer,
            max_text_length=cfg.data.max_text_length,
        )
    raise ValueError(f"Unknown dataset: {cfg.data.name}")


def build_loader(cfg: DictConfig, dataset, collator: OCRCollator, split: str):
    if dataset is None:
        return None
    batch_size = int(cfg.train.batch_size if split == "train" else cfg.train.get("val_batch_size", cfg.train.batch_size))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collator,
        num_workers=int(cfg.data.get("num_workers", 0)),
    )


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    seed_everything(int(cfg.seed))
    L.seed_everything(int(cfg.seed), workers=True)
    torch.set_float32_matmul_precision(str(cfg.train.get("float32_matmul_precision", "high")))
    if bool(cfg.train.get("suppress_accumulate_grad_stream_warning", False)):
        torch.autograd.graph.set_warn_on_accumulate_grad_stream_mismatch(False)
    tokenizer = build_tokenizer(cfg)
    pad_token_id = tokenizer_pad_id(tokenizer)

    collator = OCRCollator(
        pad_token_id=pad_token_id,
        canvas_length=int(cfg.model.canvas.length),
        mask_token_id=pad_token_id,
    )
    dataset = build_dataset(cfg, tokenizer, split="train")
    val_dataset = build_dataset(cfg, tokenizer, split="val") if bool(cfg.train.get("validate", True)) else None
    loader = build_loader(cfg, dataset, collator, split="train")
    val_loader = build_loader(cfg, val_dataset, collator, split="val")

    model = build_model(cfg)
    if tokenizer is not None and hasattr(model.decoder, "lm"):
        input_embeddings = model.decoder.lm.get_input_embeddings()
        if input_embeddings.num_embeddings != len(tokenizer):
            model.decoder.lm.resize_token_embeddings(len(tokenizer))

    diffusion_schedule = None
    mask_token_id = None
    special_token_ids = {pad_token_id}
    uses_diffusion = (
        cfg.train.name in {"diffusion", "joint", "draft_refine"}
        or cfg.model.decoder.mode in {"diffusion", "joint"}
        or float(cfg.model.loss_weights.get("diffusion", 0.0)) > 0.0
    )
    if uses_diffusion:
        diffusion_schedule = DiscreteDiffusionSchedule(
            timesteps=int(cfg.model.diffusion.timesteps),
            min_mask_ratio=float(cfg.model.diffusion.get("min_mask_ratio", 0.0)),
            max_mask_ratio=float(cfg.model.diffusion.get("max_mask_ratio", 1.0)),
        )
        configured_mask_token_id = cfg.model.diffusion.get("mask_token_id", None)
        if configured_mask_token_id is None:
            mask_token_id = getattr(tokenizer, "mask_token_id", None)
            if mask_token_id is None:
                mask_token_id = pad_token_id
        else:
            mask_token_id = int(configured_mask_token_id)

    lightning_module = OCRLightningModule(
        model=model,
        learning_rate=float(cfg.train.learning_rate),
        pretrained_learning_rate=float(
            cfg.train.get("pretrained_learning_rate", cfg.train.learning_rate)
        ),
        weight_decay=float(cfg.train.weight_decay),
        scheduler_name=str(cfg.train.get("scheduler", "constant")),
        warmup_steps=int(cfg.train.get("warmup_steps", 0)),
        max_steps=int(cfg.train.max_steps),
        mode=str(cfg.train.name),
        diffusion_schedule=diffusion_schedule,
        mask_token_id=mask_token_id,
        special_token_ids=special_token_ids,
        probe_interval=int(cfg.train.get("probe_interval", 0)),
        probe_timesteps=list(cfg.train.get("probe_timesteps", [])),
        probe_visual_ablations=list(cfg.train.get("probe_visual_ablations", [])),
        ar_loss_weight=float(cfg.model.loss_weights.get("ar", 1.0)),
        diffusion_loss_weight=float(cfg.model.loss_weights.get("diffusion", 1.0)),
        tokenizer=tokenizer,
        validation_probe_timesteps=list(cfg.train.get("val_probe_timesteps", cfg.train.get("probe_timesteps", []))),
        validation_visual_ablations=list(
            cfg.train.get("val_visual_ablations", cfg.train.get("probe_visual_ablations", []))
        ),
        log_to_logger=bool(cfg.logging.enabled),
    )

    callbacks = []
    checkpoint_interval = int(cfg.train.get("checkpoint_interval", 0))
    if checkpoint_interval > 0:
        callbacks.append(
            ModelCheckpoint(
                dirpath=str(Path(cfg.output_dir) / "checkpoints"),
                filename="step_{step}",
                every_n_train_steps=checkpoint_interval,
                save_last=True,
                save_top_k=-1,
            )
        )

    trainer = L.Trainer(
        accelerator=str(cfg.train.get("accelerator", "auto")),
        devices=cfg.train.get("devices", "auto"),
        strategy=str(cfg.train.get("strategy", "auto")),
        precision=lightning_precision(str(cfg.train.precision)),
        max_steps=int(cfg.train.max_steps),
        accumulate_grad_batches=int(cfg.train.gradient_accumulation_steps),
        gradient_clip_val=float(cfg.train.get("gradient_clip_val", 0.0)),
        gradient_clip_algorithm=str(cfg.train.get("gradient_clip_algorithm", "norm")),
        log_every_n_steps=int(cfg.train.log_interval),
        val_check_interval=cfg.train.get("val_check_interval", None),
        check_val_every_n_epoch=cfg.train.get("check_val_every_n_epoch", 1),
        num_sanity_val_steps=int(cfg.train.get("num_sanity_val_steps", 0)),
        logger=build_lightning_logger(cfg),
        callbacks=callbacks,
        enable_checkpointing=bool(callbacks),
        default_root_dir=str(cfg.output_dir),
    )
    trainer.fit(lightning_module, train_dataloaders=loader, val_dataloaders=val_loader)
    print(f"completed_steps={trainer.global_step}")


if __name__ == "__main__":
    main()

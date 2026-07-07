from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
os.environ.setdefault("WANDB_ERROR_REPORTING", "false")

import hydra
import lightning as L
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


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    seed_everything(int(cfg.seed))
    L.seed_everything(int(cfg.seed), workers=True)
    tokenizer = build_tokenizer(cfg)
    pad_token_id = tokenizer_pad_id(tokenizer)

    image_size = tuple(cfg.model.image_size)
    if cfg.data.name == "synthetic":
        dataset = SyntheticOCRDataset(num_samples=cfg.data.get("num_train", 4), image_size=image_size)
    elif cfg.data.name == "manifest":
        dataset = ManifestOCRDataset(
            manifest_path=cfg.data.train_manifest,
            image_root=cfg.data.image_root,
            image_size=image_size,
            tokenizer=tokenizer,
            max_text_length=cfg.data.max_text_length,
        )
    elif cfg.data.name == "cord_v2":
        dataset = HFCordV2OCRDataset(
            dataset_name=cfg.data.dataset_name,
            dataset_path=cfg.data.get("dataset_path", None),
            split=cfg.data.split,
            image_size=tuple(cfg.data.image_size),
            target_mode=cfg.data.target_mode,
            load_from_disk=bool(cfg.data.get("load_from_disk", False)),
            streaming=bool(cfg.data.streaming),
            max_samples=cfg.data.max_samples,
            tokenizer=tokenizer,
            max_text_length=cfg.data.max_text_length,
        )
    else:
        raise ValueError(f"Unknown dataset: {cfg.data.name}")

    collator = OCRCollator(
        pad_token_id=pad_token_id,
        canvas_length=int(cfg.model.canvas.length),
        mask_token_id=pad_token_id,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(cfg.train.batch_size),
        collate_fn=collator,
        num_workers=int(cfg.data.get("num_workers", 0)),
    )

    model = build_model(cfg)
    diffusion_schedule = None
    mask_token_id = None
    special_token_ids = {pad_token_id}
    if cfg.train.name == "diffusion" or cfg.model.decoder.mode == "diffusion":
        diffusion_schedule = DiscreteDiffusionSchedule(
            timesteps=int(cfg.model.diffusion.timesteps),
            min_mask_ratio=float(cfg.model.diffusion.get("min_mask_ratio", 0.0)),
            max_mask_ratio=float(cfg.model.diffusion.get("max_mask_ratio", 1.0)),
        )
        mask_token_id = int(cfg.model.diffusion.mask_token_id)

    lightning_module = OCRLightningModule(
        model=model,
        learning_rate=float(cfg.train.learning_rate),
        weight_decay=float(cfg.train.weight_decay),
        mode=str(cfg.train.name),
        diffusion_schedule=diffusion_schedule,
        mask_token_id=mask_token_id,
        special_token_ids=special_token_ids,
        probe_interval=int(cfg.train.get("probe_interval", 0)),
        probe_timesteps=list(cfg.train.get("probe_timesteps", [])),
        probe_visual_ablations=list(cfg.train.get("probe_visual_ablations", [])),
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
        log_every_n_steps=int(cfg.train.log_interval),
        logger=build_lightning_logger(cfg),
        callbacks=callbacks,
        enable_checkpointing=bool(callbacks),
        default_root_dir=str(cfg.output_dir),
    )
    trainer.fit(lightning_module, train_dataloaders=loader)
    print(f"completed_steps={trainer.global_step}")


if __name__ == "__main__":
    main()

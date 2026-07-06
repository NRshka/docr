from __future__ import annotations

import hydra
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from docr.data.collate import OCRCollator
from docr.data.dataset import HFCordV2OCRDataset, ManifestOCRDataset, SyntheticOCRDataset
from docr.models.factory import build_model
from docr.models.diffusion import DiscreteDiffusionSchedule
from docr.training.trainer import OCRTrainer
from docr.utils.logging import maybe_init_wandb
from docr.utils.seed import seed_everything
from docr.utils.tokenizer import build_tokenizer, tokenizer_pad_id
from docr.utils.torch import resolve_device


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    seed_everything(int(cfg.seed))
    device = resolve_device(str(cfg.device))
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
    logger = maybe_init_wandb(cfg)
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

    trainer = OCRTrainer(
        model=model,
        learning_rate=float(cfg.train.learning_rate),
        weight_decay=float(cfg.train.weight_decay),
        device=device,
        logger=logger,
        log_interval=int(cfg.train.log_interval),
        mode=str(cfg.train.name),
        diffusion_schedule=diffusion_schedule,
        mask_token_id=mask_token_id,
        special_token_ids=special_token_ids,
    )
    try:
        trainer.fit(loader, max_steps=int(cfg.train.max_steps))
    finally:
        if logger is not None:
            logger.finish()
    print(f"completed_steps={trainer.state.step}")


if __name__ == "__main__":
    main()

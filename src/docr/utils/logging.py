from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf


def maybe_init_wandb(cfg: Any) -> Any | None:
    logging_cfg = getattr(cfg, "logging", None)
    if logging_cfg is None or not getattr(logging_cfg, "enabled", False):
        return None

    wandb_dir = Path(getattr(logging_cfg, "dir", "outputs/wandb"))
    wandb_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("WANDB_DIR", str(wandb_dir))
    os.environ.setdefault("WANDB_CACHE_DIR", str(wandb_dir / "cache"))
    os.environ.setdefault("WANDB_CONFIG_DIR", str(wandb_dir / "config"))
    os.environ.setdefault("WANDB_ERROR_REPORTING", "false")
    Path(os.environ["WANDB_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["WANDB_CONFIG_DIR"]).mkdir(parents=True, exist_ok=True)

    import wandb

    mode = "offline" if getattr(logging_cfg, "offline", True) else "online"
    try:
        return wandb.init(
            project=getattr(logging_cfg, "project", "docr"),
            entity=getattr(logging_cfg, "entity", None),
            name=getattr(logging_cfg, "run_name", None),
            tags=list(getattr(logging_cfg, "tags", [])),
            mode=mode,
            dir=str(wandb_dir),
            config=OmegaConf.to_container(cfg, resolve=True),
            settings=wandb.Settings(
                start_method="thread",
                silent=bool(getattr(logging_cfg, "quiet", False)),
            ),
        )
    except Exception as exc:
        print(f"wandb_init_failed={exc!r}; continuing without WandB logging")
        return None

from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig, OmegaConf


def config_to_container(cfg: DictConfig) -> dict:
    return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


from pathlib import Path

import torch
from omegaconf import OmegaConf

from scripts.train import load_model_weights_only, resolve_val_check_interval


def test_optimizer_step_validation_interval_resolves_through_accumulation():
    cfg = OmegaConf.create(
        {
            "gradient_accumulation_steps": 4,
            "val_check_interval": 17,
            "val_check_interval_optimizer_steps": 250,
        }
    )

    assert resolve_val_check_interval(cfg) == 1000


def test_weights_only_initialization_strips_lightning_model_prefix(tmp_path: Path):
    source = torch.nn.Linear(3, 2)
    target = torch.nn.Linear(3, 2)
    with torch.no_grad():
        source.weight.fill_(2.5)
        source.bias.fill_(-0.75)
    checkpoint = {
        "state_dict": {f"model.{name}": value for name, value in source.state_dict().items()},
        "global_step": 500,
        "optimizer_states": [{"ignored": True}],
        "hyper_parameters": OmegaConf.create({"mode": "ar", "nested": {"value": 3}}),
    }
    path = tmp_path / "stage1.ckpt"
    torch.save(checkpoint, path)

    missing, unexpected = load_model_weights_only(target, path, strict=True)

    assert missing == []
    assert unexpected == []
    assert torch.equal(target.weight, source.weight)
    assert torch.equal(target.bias, source.bias)

import pytest
import torch

from docr.training.schedules import build_scheduler


def test_cosine_scheduler_warms_up_then_decays():
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.AdamW([parameter], lr=1.0)
    scheduler = build_scheduler(optimizer, name="cosine", max_steps=10, warmup_steps=2)

    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.5)
    optimizer.step()
    scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1.0)

    for _ in range(9):
        optimizer.step()
        scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.0)


def test_scheduler_rejects_invalid_warmup():
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.AdamW([parameter], lr=1.0)
    with pytest.raises(ValueError, match="warmup_steps"):
        build_scheduler(optimizer, name="cosine", max_steps=10, warmup_steps=10)


def test_fractional_warmup_is_interpreted_as_ratio():
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.AdamW([parameter], lr=1.0)
    build_scheduler(optimizer, name="constant", max_steps=100, warmup_steps=0.1)

    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.1)

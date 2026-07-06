from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def load_rgb_image(path: str | Path) -> Image.Image:
    return Image.open(path).convert("RGB")


class BasicImageTransform:
    """Small dependency-light image transform for early experiments."""

    def __init__(self, image_size: tuple[int, int]) -> None:
        self.image_size = image_size

    def __call__(self, image: Image.Image) -> torch.Tensor:
        resized = image.resize(self.image_size)
        array = np.asarray(resized, dtype=np.uint8)
        data = torch.from_numpy(array.copy()).permute(2, 0, 1).float()
        return data.div(255.0)


def build_image_transform(image_size: tuple[int, int]) -> Callable[[Image.Image], torch.Tensor]:
    return BasicImageTransform(image_size=image_size)

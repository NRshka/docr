from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image


def load_rgb_image(path: str | Path) -> Image.Image:
    return Image.open(path).convert("RGB")


class BasicImageTransform:
    """Small dependency-light image transform for early experiments."""

    NORMALIZATION_STATS = {
        "none": (None, None),
        # SAM uses the conventional ImageNet RGB statistics. Applying them to [0, 1]
        # tensors is equivalent to the processor's 0-255 mean/std values.
        "sam": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        "glm_ocr": (
            (0.48145466, 0.4578275, 0.40821073),
            (0.26862954, 0.26130258, 0.27577711),
        ),
    }

    def __init__(
        self,
        image_size: tuple[int, int],
        preserve_aspect_ratio: bool = False,
        normalization: str = "none",
    ) -> None:
        self.image_size = image_size
        self.preserve_aspect_ratio = preserve_aspect_ratio
        if normalization not in self.NORMALIZATION_STATS:
            raise ValueError(f"Unknown image normalization: {normalization}")
        self.normalization = normalization

    def __call__(self, image: Image.Image) -> torch.Tensor:
        target_width, target_height = self.image_size
        if self.preserve_aspect_ratio:
            scale = min(target_width / image.width, target_height / image.height)
            resized_width = max(1, min(target_width, round(image.width * scale)))
            resized_height = max(1, min(target_height, round(image.height * scale)))
            resized = image.resize((resized_width, resized_height), Image.Resampling.BICUBIC)
        else:
            resized = image.resize(self.image_size, Image.Resampling.BICUBIC)
        array = np.asarray(resized, dtype=np.uint8)
        data = torch.from_numpy(array.copy()).permute(2, 0, 1).float()
        data = data.div(255.0)
        mean, std = self.NORMALIZATION_STATS[self.normalization]
        if mean is not None and std is not None:
            mean_tensor = torch.tensor(mean, dtype=data.dtype).view(3, 1, 1)
            std_tensor = torch.tensor(std, dtype=data.dtype).view(3, 1, 1)
            data = (data - mean_tensor) / std_tensor

        if not self.preserve_aspect_ratio:
            return data
        # Pad after normalization so unused pixels are zero in feature space, matching the
        # convention used by pretrained vision processors.
        canvas = torch.zeros(3, target_height, target_width, dtype=data.dtype)
        canvas[:, : data.shape[1], : data.shape[2]] = data
        return canvas


def build_image_transform(
    image_size: tuple[int, int],
    preserve_aspect_ratio: bool = False,
    normalization: str = "none",
) -> BasicImageTransform:
    return BasicImageTransform(
        image_size=image_size,
        preserve_aspect_ratio=preserve_aspect_ratio,
        normalization=normalization,
    )

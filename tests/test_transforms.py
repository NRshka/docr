import pytest
import torch
from PIL import Image

from docr.data.transforms import BasicImageTransform


def test_aspect_ratio_resize_pads_in_normalized_space():
    image = Image.new("RGB", (8, 4), color=(255, 255, 255))
    transform = BasicImageTransform(
        image_size=(8, 8),
        preserve_aspect_ratio=True,
        normalization="sam",
    )

    output = transform(image)

    assert output.shape == (3, 8, 8)
    assert torch.all(output[:, :4, :] != 0)
    assert torch.equal(output[:, 4:, :], torch.zeros(3, 4, 8))
    expected = torch.tensor(
        [(1.0 - 0.485) / 0.229, (1.0 - 0.456) / 0.224, (1.0 - 0.406) / 0.225]
    )
    assert torch.allclose(output[:, 0, 0], expected)


def test_transform_rejects_unknown_normalization():
    with pytest.raises(ValueError, match="Unknown image normalization"):
        BasicImageTransform((8, 8), normalization="unknown")

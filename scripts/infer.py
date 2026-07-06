from __future__ import annotations

import hydra
import torch
from omegaconf import DictConfig

from docr.inference.ar_decode import ar_decode
from docr.inference.diffusion_decode import diffusion_decode
from docr.inference.draft_refine import draft_refine_decode
from docr.models.factory import build_model
from docr.utils.tokenizer import build_tokenizer, tokenizer_pad_id


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    mode = str(cfg.get("mode", cfg.model.decoder.mode))
    tokenizer = build_tokenizer(cfg)
    pad_token_id = tokenizer_pad_id(tokenizer)
    image_size = tuple(cfg.model.image_size)
    width, height = image_size
    images = torch.zeros(1, 3, height, width)
    prompt = torch.full((1, int(cfg.model.canvas.length)), pad_token_id, dtype=torch.long)
    model = build_model(cfg)

    if mode == "ar":
        token_ids = ar_decode(model, images, prompt)
    elif mode == "diffusion":
        token_ids = diffusion_decode(model, images, prompt, steps=1)
    elif mode == "draft_refine":
        token_ids = draft_refine_decode(model, images, prompt, diffusion_steps=1)
    else:
        raise ValueError(f"Unknown inference mode: {mode}")

    if tokenizer is None:
        text = "".join(chr(int(token_id)) for token_id in token_ids[0] if 0 <= int(token_id) < 128)
    else:
        text = tokenizer.decode(token_ids[0], skip_special_tokens=True)
    print(text)


if __name__ == "__main__":
    main()

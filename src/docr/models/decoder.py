from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class DecoderOutput:
    logits: torch.Tensor


class TinyTextDecoder(nn.Module):
    """Minimal decoder placeholder with AR and diffusion-compatible forward shape."""

    def __init__(self, vocab_size: int = 256, hidden_size: int = 512) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.visual_proj = nn.Linear(hidden_size, hidden_size)
        self.rnn = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.lm_head = nn.Linear(hidden_size, vocab_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        visual_tokens: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        timestep: torch.Tensor | None = None,
        mode: str = "ar",
    ) -> DecoderOutput:
        del attention_mask, timestep, mode
        hidden = self.embed(input_ids.clamp_min(0).clamp_max(self.embed.num_embeddings - 1))
        if visual_tokens is not None:
            visual_context = self.visual_proj(visual_tokens.mean(dim=1)).unsqueeze(1)
            hidden = hidden + visual_context
        outputs, _ = self.rnn(hidden)
        return DecoderOutput(logits=self.lm_head(outputs))


class DiffusionTransformerDecoder(nn.Module):
    """Small bidirectional denoiser for early diffusion OCR experiments."""

    def __init__(
        self,
        vocab_size: int = 256,
        hidden_size: int = 512,
        visual_hidden_size: int = 512,
        num_layers: int = 4,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        max_length: int = 1024,
        timesteps: int = 32,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.position_embed = nn.Embedding(max_length, hidden_size)
        self.timestep_embed = nn.Embedding(timesteps, hidden_size)
        self.visual_proj = nn.Linear(visual_hidden_size, hidden_size)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=int(hidden_size * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        visual_tokens: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        timestep: torch.Tensor | None = None,
        mode: str = "diffusion",
    ) -> DecoderOutput:
        if mode not in {"diffusion", "ar"}:
            raise ValueError(f"Unsupported decoder mode: {mode}")
        if input_ids.shape[1] > self.max_length:
            raise ValueError(f"input length {input_ids.shape[1]} exceeds max_length {self.max_length}")

        clipped_ids = input_ids.clamp_min(0).clamp_max(self.vocab_size - 1)
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        hidden = self.embed(clipped_ids) + self.position_embed(positions).unsqueeze(0)

        if timestep is None:
            timestep = torch.zeros(input_ids.shape[0], dtype=torch.long, device=input_ids.device)
        if timestep.ndim == 0:
            timestep = timestep.expand(input_ids.shape[0])
        hidden = hidden + self.timestep_embed(timestep.long()).unsqueeze(1)

        if visual_tokens is not None:
            visual_context = self.visual_proj(visual_tokens.mean(dim=1)).unsqueeze(1)
            hidden = hidden + visual_context

        padding_mask = None if attention_mask is None else ~attention_mask.bool()
        hidden = self.encoder(hidden, src_key_padding_mask=padding_mask)
        return DecoderOutput(logits=self.lm_head(self.norm(hidden)))


class QwenVisualPrefixDecoder(nn.Module):
    """Qwen causal LM conditioned on compressed visual tokens as soft prefix embeddings."""

    def __init__(
        self,
        backbone_name: str = "Qwen/Qwen2.5-0.5B",
        visual_hidden_size: int = 512,
        timesteps: int = 32,
        freeze_lm: bool = False,
        local_files_only: bool = False,
        trust_remote_code: bool = False,
    ) -> None:
        super().__init__()
        from transformers import AutoModelForCausalLM

        self.lm = AutoModelForCausalLM.from_pretrained(
            backbone_name,
            local_files_only=local_files_only,
            trust_remote_code=trust_remote_code,
        )
        hidden_size = int(self.lm.config.hidden_size)
        self.visual_proj = nn.Linear(visual_hidden_size, hidden_size)
        self.timestep_embed = nn.Embedding(timesteps, hidden_size)

        if freeze_lm:
            for parameter in self.lm.parameters():
                parameter.requires_grad = False

    def forward(
        self,
        input_ids: torch.Tensor,
        visual_tokens: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        timestep: torch.Tensor | None = None,
        mode: str = "ar",
    ) -> DecoderOutput:
        if mode not in {"ar", "diffusion"}:
            raise ValueError(f"Unsupported decoder mode: {mode}")
        if visual_tokens is None:
            raise ValueError("QwenVisualPrefixDecoder requires visual_tokens")

        text_embeds = self.lm.get_input_embeddings()(input_ids)
        if mode == "diffusion":
            if timestep is None:
                timestep = torch.zeros(input_ids.shape[0], dtype=torch.long, device=input_ids.device)
            if timestep.ndim == 0:
                timestep = timestep.expand(input_ids.shape[0])
            text_embeds = text_embeds + self.timestep_embed(timestep.long()).unsqueeze(1).to(
                dtype=text_embeds.dtype
            )
        visual_embeds = self.visual_proj(visual_tokens).to(dtype=text_embeds.dtype)
        inputs_embeds = torch.cat([visual_embeds, text_embeds], dim=1)

        prefix_mask = torch.ones(
            visual_embeds.shape[:2],
            dtype=torch.long,
            device=input_ids.device,
        )
        if attention_mask is None:
            text_mask = torch.ones_like(input_ids, dtype=torch.long)
        else:
            text_mask = attention_mask.to(dtype=torch.long, device=input_ids.device)
        full_attention_mask = torch.cat([prefix_mask, text_mask], dim=1)

        outputs = self.lm(
            inputs_embeds=inputs_embeds,
            attention_mask=full_attention_mask,
            use_cache=False,
        )

        prefix_len = visual_embeds.shape[1]
        text_logits = outputs.logits[:, prefix_len - 1 : -1, :]
        return DecoderOutput(logits=text_logits)


class OCRModel(nn.Module):
    def __init__(self, vision_encoder: nn.Module, decoder: nn.Module) -> None:
        super().__init__()
        self.vision_encoder = vision_encoder
        self.decoder = decoder

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        return self.vision_encoder(images)

    def decode_text(
        self,
        input_ids: torch.Tensor,
        visual_tokens: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        timestep: torch.Tensor | None = None,
        mode: str = "ar",
    ) -> DecoderOutput:
        return self.decoder(
            input_ids=input_ids,
            visual_tokens=visual_tokens,
            attention_mask=attention_mask,
            timestep=timestep,
            mode=mode,
        )

    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        timestep: torch.Tensor | None = None,
        mode: str = "ar",
    ) -> DecoderOutput:
        visual_tokens = self.encode_images(images)
        return self.decode_text(
            input_ids=input_ids,
            visual_tokens=visual_tokens,
            attention_mask=attention_mask,
            timestep=timestep,
            mode=mode,
        )

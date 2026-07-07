from __future__ import annotations

from typing import Any

import torch
from torch import nn

from docr.models.decoder import DecoderOutput


def build_unified_qwen_allowed_mask(
    mode: str,
    num_visual_tokens: int,
    text_length: int,
    num_canvases: int = 1,
    canvas_length: int | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Return a [visual + text, visual + text] bool mask where True means attend."""

    if mode not in {"ar", "diffusion"}:
        raise ValueError(f"Unsupported decoder mode: {mode}")
    total = num_visual_tokens + text_length
    mask = torch.zeros((total, total), dtype=torch.bool, device=device)
    mask[:num_visual_tokens, :num_visual_tokens] = True

    if mode == "ar":
        text_causal = torch.tril(torch.ones((text_length, text_length), dtype=torch.bool, device=device))
        mask[num_visual_tokens:, :num_visual_tokens] = True
        mask[num_visual_tokens:, num_visual_tokens:] = text_causal
        return mask

    if canvas_length is None:
        if text_length % num_canvases != 0:
            raise ValueError(
                f"text_length={text_length} is not divisible by num_canvases={num_canvases}"
            )
        canvas_length = text_length // num_canvases
    expected_text_length = num_canvases * canvas_length
    if expected_text_length != text_length:
        raise ValueError(
            f"num_canvases * canvas_length must equal text_length, got "
            f"{num_canvases} * {canvas_length} != {text_length}"
        )

    mask[num_visual_tokens:, :num_visual_tokens] = True
    for canvas_idx in range(num_canvases):
        start = num_visual_tokens + canvas_idx * canvas_length
        end = start + canvas_length
        mask[start:end, start:end] = True
    return mask


def build_linear_position_ids(
    batch_size: int,
    seq_len: int,
    device: torch.device | None = None,
) -> torch.Tensor:
    return torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0).expand(batch_size, -1)


def allowed_mask_to_additive_attention_mask(
    allowed_mask: torch.Tensor,
    batch_size: int,
    dtype: torch.dtype,
    key_padding_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if allowed_mask.ndim != 2:
        raise ValueError(f"allowed_mask must be 2D, got shape {tuple(allowed_mask.shape)}")
    expanded = allowed_mask.unsqueeze(0).expand(batch_size, -1, -1)
    if key_padding_mask is not None:
        expanded = expanded & key_padding_mask.bool().unsqueeze(1)
    additive = torch.zeros(expanded.shape, dtype=dtype, device=allowed_mask.device)
    additive = additive.masked_fill(~expanded, torch.finfo(dtype).min)
    return additive.unsqueeze(1)


class UnifiedQwenDecoder(nn.Module):
    """Qwen decoder with explicit AR and diffusion attention masks.

    This class reuses Hugging Face Qwen weights and decoder blocks, but owns the
    multimodal sequence assembly and mask construction. V1 intentionally keeps
    standard 1D RoPE through linear position ids.
    """

    def __init__(
        self,
        backbone_name: str = "Qwen/Qwen2.5-0.5B",
        visual_hidden_size: int = 512,
        timesteps: int = 32,
        num_canvases: int = 1,
        canvas_length: int | None = None,
        freeze_lm: bool = False,
        local_files_only: bool = False,
        trust_remote_code: bool = False,
        config: Any | None = None,
    ) -> None:
        super().__init__()
        if config is None:
            from transformers import AutoModelForCausalLM

            self.lm = AutoModelForCausalLM.from_pretrained(
                backbone_name,
                local_files_only=local_files_only,
                trust_remote_code=trust_remote_code,
            )
        else:
            from transformers import Qwen2ForCausalLM

            self.lm = Qwen2ForCausalLM(config)

        hidden_size = int(self.lm.config.hidden_size)
        self.visual_proj = nn.Linear(visual_hidden_size, hidden_size)
        self.timestep_embed = nn.Embedding(timesteps, hidden_size)
        self.num_canvases = num_canvases
        self.canvas_length = canvas_length

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
            raise ValueError("UnifiedQwenDecoder requires visual_tokens")

        text_embeds = self.lm.get_input_embeddings()(input_ids)
        if mode == "diffusion":
            timestep = self._normalize_timestep(timestep, input_ids)
            text_embeds = text_embeds + self.timestep_embed(timestep).unsqueeze(1).to(
                dtype=text_embeds.dtype
            )

        visual_embeds = self.visual_proj(visual_tokens).to(dtype=text_embeds.dtype)
        inputs_embeds = torch.cat([visual_embeds, text_embeds], dim=1)
        hidden_states = self._forward_model(
            inputs_embeds=inputs_embeds,
            text_attention_mask=attention_mask,
            num_visual_tokens=visual_embeds.shape[1],
            text_length=input_ids.shape[1],
            mode=mode,
        )

        prefix_len = visual_embeds.shape[1]
        if mode == "ar":
            logits = self.lm.lm_head(hidden_states[:, prefix_len - 1 : -1, :])
        else:
            logits = self.lm.lm_head(hidden_states[:, prefix_len:, :])
        return DecoderOutput(logits=logits)

    def _normalize_timestep(
        self,
        timestep: torch.Tensor | None,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        if timestep is None:
            timestep = torch.zeros(input_ids.shape[0], dtype=torch.long, device=input_ids.device)
        if timestep.ndim == 0:
            timestep = timestep.expand(input_ids.shape[0])
        return timestep.long()

    def _forward_model(
        self,
        inputs_embeds: torch.Tensor,
        text_attention_mask: torch.Tensor | None,
        num_visual_tokens: int,
        text_length: int,
        mode: str,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = inputs_embeds.shape
        model = self.lm.model
        position_ids = build_linear_position_ids(batch_size, seq_len, device=inputs_embeds.device)
        key_padding_mask = self._build_key_padding_mask(
            text_attention_mask=text_attention_mask,
            batch_size=batch_size,
            num_visual_tokens=num_visual_tokens,
            device=inputs_embeds.device,
        )
        allowed_mask = build_unified_qwen_allowed_mask(
            mode=mode,
            num_visual_tokens=num_visual_tokens,
            text_length=text_length,
            num_canvases=self.num_canvases,
            canvas_length=self._active_canvas_length(text_length) if mode == "diffusion" else None,
            device=inputs_embeds.device,
        )
        attention_mask = allowed_mask_to_additive_attention_mask(
            allowed_mask=allowed_mask,
            batch_size=batch_size,
            dtype=inputs_embeds.dtype,
            key_padding_mask=key_padding_mask,
        )

        hidden_states = inputs_embeds
        position_embeddings = model.rotary_emb(hidden_states, position_ids)
        layer_types = getattr(
            model.config,
            "layer_types",
            ["full_attention"] * int(model.config.num_hidden_layers),
        )
        mask_mapping = {layer_type: attention_mask for layer_type in set(layer_types)}

        for layer_idx, decoder_layer in enumerate(model.layers[: model.config.num_hidden_layers]):
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=mask_mapping[layer_types[layer_idx]],
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                past_key_values=None,
                use_cache=False,
            )

        return model.norm(hidden_states)

    def _active_canvas_length(self, text_length: int) -> int | None:
        if self.num_canvases == 1:
            return text_length
        if self.canvas_length is not None and self.num_canvases * self.canvas_length == text_length:
            return self.canvas_length
        if text_length % self.num_canvases == 0:
            return text_length // self.num_canvases
        return self.canvas_length

    def _build_key_padding_mask(
        self,
        text_attention_mask: torch.Tensor | None,
        batch_size: int,
        num_visual_tokens: int,
        device: torch.device,
    ) -> torch.Tensor | None:
        if text_attention_mask is None:
            return None
        visual_mask = torch.ones(
            (batch_size, num_visual_tokens),
            dtype=torch.bool,
            device=device,
        )
        text_mask = text_attention_mask.to(dtype=torch.bool, device=device)
        return torch.cat([visual_mask, text_mask], dim=1)

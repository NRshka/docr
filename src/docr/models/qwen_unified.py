from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import torch
from torch import nn

from docr.models.decoder import DecoderOutput, DualStreamDecoderOutput


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


def build_dual_stream_allowed_mask(
    num_visual_tokens: int,
    clean_attention_mask: torch.Tensor,
    block_starts: torch.Tensor,
    noisy_block_mask: torch.Tensor,
) -> torch.Tensor:
    """Build [batch, total, total] asymmetric clean/noisy attention permissions."""

    if clean_attention_mask.ndim != 2 or noisy_block_mask.ndim != 2:
        raise ValueError("clean and noisy masks must be [batch, sequence]")
    batch_size, clean_length = clean_attention_mask.shape
    if block_starts.shape != (batch_size,):
        raise ValueError("block_starts must have shape [batch]")
    if noisy_block_mask.shape[0] != batch_size:
        raise ValueError("noisy block batch size must match clean stream")
    block_length = noisy_block_mask.shape[1]
    total = num_visual_tokens + clean_length + block_length
    device = clean_attention_mask.device
    allowed = torch.zeros((batch_size, total, total), dtype=torch.bool, device=device)
    visual_end = num_visual_tokens
    clean_start = visual_end
    clean_end = clean_start + clean_length
    noisy_start = clean_end

    allowed[:, :visual_end, :visual_end] = True
    clean_valid = clean_attention_mask.bool()
    causal = torch.tril(torch.ones((clean_length, clean_length), dtype=torch.bool, device=device))
    allowed[:, clean_start:clean_end, :visual_end] = True
    allowed[:, clean_start:clean_end, clean_start:clean_end] = (
        causal.unsqueeze(0) & clean_valid.unsqueeze(1)
    )

    allowed[:, noisy_start:, :visual_end] = True
    clean_positions = torch.arange(clean_length, device=device).view(1, 1, clean_length)
    prefix_keys = clean_positions < block_starts.view(batch_size, 1, 1)
    prefix_keys &= clean_valid.view(batch_size, 1, clean_length)
    allowed[:, noisy_start:, clean_start:clean_end] = prefix_keys.expand(
        batch_size, block_length, clean_length
    )
    allowed[:, noisy_start:, noisy_start:] = noisy_block_mask.bool().unsqueeze(1).expand(
        batch_size, block_length, block_length
    )
    return allowed


def batched_allowed_mask_to_additive(
    allowed_mask: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    if allowed_mask.ndim != 3:
        raise ValueError("batched allowed mask must be [batch, query, key]")
    additive = torch.zeros(allowed_mask.shape, dtype=dtype, device=allowed_mask.device)
    return additive.masked_fill(~allowed_mask, torch.finfo(dtype).min).unsqueeze(1)


class GatedVisualCrossAttention(nn.Module):
    """Pre-norm visual cross-attention with an exactly closed initial residual gate."""

    def __init__(self, hidden_size: int, num_heads: int) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by cross-attention heads")
        self.query_norm = nn.LayerNorm(hidden_size)
        self.visual_norm = nn.LayerNorm(hidden_size)
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=0.0,
            batch_first=True,
        )
        self.gate = nn.Parameter(torch.zeros(()))

    def forward(self, hidden_states: torch.Tensor, visual_states: torch.Tensor) -> torch.Tensor:
        query = self.query_norm(hidden_states)
        memory = self.visual_norm(visual_states)
        attended, _ = self.attention(
            query=query,
            key=memory,
            value=memory,
            need_weights=False,
        )
        return hidden_states + torch.tanh(self.gate).to(dtype=attended.dtype) * attended


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
        sdpa_backend: str = "safe",
        cross_attention_layers: list[int] | None = None,
        cross_attention_heads: int = 8,
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
        self.sdpa_backend = sdpa_backend
        self.cross_attention_layer_indices = self._resolve_cross_attention_layers(
            cross_attention_layers or [], int(self.lm.config.num_hidden_layers)
        )
        self.cross_attention = nn.ModuleDict(
            {
                str(layer_idx): GatedVisualCrossAttention(hidden_size, cross_attention_heads)
                for layer_idx in self.cross_attention_layer_indices
            }
        )

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

        visual_embeds = self.visual_proj(
            visual_tokens.to(dtype=self.visual_proj.weight.dtype)
        ).to(dtype=text_embeds.dtype)
        inputs_embeds = torch.cat([visual_embeds, text_embeds], dim=1)
        hidden_states = self._forward_model(
            inputs_embeds=inputs_embeds,
            text_attention_mask=attention_mask,
            num_visual_tokens=visual_embeds.shape[1],
            text_length=input_ids.shape[1],
            mode=mode,
            visual_states=visual_embeds,
        )

        prefix_len = visual_embeds.shape[1]
        if mode == "ar":
            logits = self.lm.lm_head(hidden_states[:, prefix_len - 1 : -1, :])
        else:
            logits = self.lm.lm_head(hidden_states[:, prefix_len:, :])
        return DecoderOutput(logits=logits)

    def forward_dual_stream(
        self,
        clean_input_ids: torch.Tensor,
        noisy_block_ids: torch.Tensor,
        block_starts: torch.Tensor,
        visual_tokens: torch.Tensor,
        clean_attention_mask: torch.Tensor | None = None,
        noisy_block_mask: torch.Tensor | None = None,
        timestep: torch.Tensor | None = None,
    ) -> DualStreamDecoderOutput:
        if clean_input_ids.ndim != 2 or noisy_block_ids.ndim != 2:
            raise ValueError("clean and noisy ids must be [batch, sequence]")
        batch_size, clean_length = clean_input_ids.shape
        if noisy_block_ids.shape[0] != batch_size or visual_tokens.shape[0] != batch_size:
            raise ValueError("dual-stream batch dimensions must match")
        if clean_attention_mask is None:
            clean_attention_mask = torch.ones_like(clean_input_ids, dtype=torch.bool)
        else:
            clean_attention_mask = clean_attention_mask.to(
                device=clean_input_ids.device, dtype=torch.bool
            )
        if noisy_block_mask is None:
            noisy_block_mask = torch.ones_like(noisy_block_ids, dtype=torch.bool)
        else:
            noisy_block_mask = noisy_block_mask.to(
                device=noisy_block_ids.device, dtype=torch.bool
            )

        clean_embeds = self.lm.get_input_embeddings()(clean_input_ids)
        noisy_embeds = self.lm.get_input_embeddings()(noisy_block_ids)
        timestep = self._normalize_timestep(timestep, clean_input_ids)
        noisy_embeds = noisy_embeds + self.timestep_embed(timestep).unsqueeze(1).to(
            dtype=noisy_embeds.dtype
        )
        visual_embeds = self.visual_proj(
            visual_tokens.to(dtype=self.visual_proj.weight.dtype)
        ).to(dtype=clean_embeds.dtype)
        inputs_embeds = torch.cat([visual_embeds, clean_embeds, noisy_embeds], dim=1)
        num_visual_tokens = visual_embeds.shape[1]
        block_length = noisy_block_ids.shape[1]
        allowed_mask = build_dual_stream_allowed_mask(
            num_visual_tokens=num_visual_tokens,
            clean_attention_mask=clean_attention_mask,
            block_starts=block_starts.to(device=clean_input_ids.device, dtype=torch.long),
            noisy_block_mask=noisy_block_mask,
        )
        attention_mask = batched_allowed_mask_to_additive(allowed_mask, inputs_embeds.dtype)

        visual_positions = torch.arange(num_visual_tokens, device=clean_input_ids.device)
        clean_positions = num_visual_tokens + torch.arange(
            clean_length, device=clean_input_ids.device
        )
        noisy_offsets = torch.arange(block_length, device=clean_input_ids.device).unsqueeze(0)
        noisy_positions = num_visual_tokens + block_starts.long().unsqueeze(1) + noisy_offsets
        position_ids = torch.cat(
            [
                visual_positions.unsqueeze(0).expand(batch_size, -1),
                clean_positions.unsqueeze(0).expand(batch_size, -1),
                noisy_positions,
            ],
            dim=1,
        )
        hidden_states = self._forward_with_mask(
            inputs_embeds,
            attention_mask,
            position_ids,
            visual_states=visual_embeds,
            num_visual_tokens=num_visual_tokens,
        )
        clean_start = num_visual_tokens
        noisy_start = clean_start + clean_length
        ar_hidden = hidden_states[:, clean_start - 1 : noisy_start - 1, :]
        noisy_hidden = hidden_states[:, noisy_start:, :]
        return DualStreamDecoderOutput(
            ar_logits=self.lm.lm_head(ar_hidden),
            diffusion_logits=self.lm.lm_head(noisy_hidden),
        )

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
        visual_states: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = inputs_embeds.shape
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

        return self._forward_with_mask(
            inputs_embeds,
            attention_mask,
            position_ids,
            visual_states=visual_states,
            num_visual_tokens=num_visual_tokens,
        )

    def _forward_with_mask(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        visual_states: torch.Tensor,
        num_visual_tokens: int,
    ) -> torch.Tensor:
        model = self.lm.model
        hidden_states = inputs_embeds
        position_embeddings = model.rotary_emb(hidden_states, position_ids)
        layer_types = getattr(
            model.config,
            "layer_types",
            ["full_attention"] * int(model.config.num_hidden_layers),
        )
        mask_mapping = {layer_type: attention_mask for layer_type in set(layer_types)}

        with self._sdpa_kernel_context(inputs_embeds.device):
            for layer_idx, decoder_layer in enumerate(model.layers[: model.config.num_hidden_layers]):
                hidden_states = decoder_layer(
                    hidden_states,
                    attention_mask=mask_mapping[layer_types[layer_idx]],
                    position_ids=position_ids,
                    position_embeddings=position_embeddings,
                    past_key_values=None,
                    use_cache=False,
                )
                layer_key = str(layer_idx)
                if layer_key in self.cross_attention:
                    cross_attention = self.cross_attention[layer_key]
                    text_states = cross_attention(
                        hidden_states[:, num_visual_tokens:, :], visual_states
                    )
                    hidden_states = torch.cat(
                        [hidden_states[:, :num_visual_tokens, :], text_states], dim=1
                    )

        return model.norm(hidden_states)

    @staticmethod
    def _resolve_cross_attention_layers(
        configured_layers: list[int], num_hidden_layers: int
    ) -> tuple[int, ...]:
        resolved = []
        for layer_idx in configured_layers:
            normalized = layer_idx if layer_idx >= 0 else num_hidden_layers + layer_idx
            if not 0 <= normalized < num_hidden_layers:
                raise ValueError(
                    f"cross-attention layer {layer_idx} is outside a {num_hidden_layers}-layer model"
                )
            resolved.append(normalized)
        if len(set(resolved)) != len(resolved):
            raise ValueError("cross-attention layers must be unique")
        return tuple(sorted(resolved))

    def _sdpa_kernel_context(self, device: torch.device):
        if device.type != "cuda" or self.sdpa_backend == "auto":
            return nullcontext()
        from torch.nn.attention import SDPBackend, sdpa_kernel

        if self.sdpa_backend == "math":
            backends = [SDPBackend.MATH]
        elif self.sdpa_backend == "safe":
            # cuDNN SDPA backward has produced deterministic NaN query gradients with the
            # block/custom additive masks used here. Prefer fused kernels when supported,
            # retain math as a stable fallback, and deliberately exclude cuDNN attention.
            backends = [
                SDPBackend.FLASH_ATTENTION,
                SDPBackend.EFFICIENT_ATTENTION,
                SDPBackend.MATH,
            ]
        else:
            raise ValueError(f"Unknown SDPA backend policy: {self.sdpa_backend}")
        return sdpa_kernel(backends)

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

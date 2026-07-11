from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DecodeStats:
    cycles: int
    draft_forwards: int
    verify_forwards: int
    accepted_draft_tokens: int
    committed_tokens: int
    proposed_draft_tokens: int
    zero_acceptance_cycles: int
    accepted_by_offset: tuple[int, ...]
    proposed_by_offset: tuple[int, ...]

    @property
    def model_forwards(self) -> int:
        return self.draft_forwards + self.verify_forwards

    @property
    def mean_accepted_prefix(self) -> float:
        return self.accepted_draft_tokens / max(self.cycles, 1)

    @property
    def tokens_per_forward(self) -> float:
        return self.committed_tokens / max(self.model_forwards, 1)


@dataclass(frozen=True)
class DecodeResult:
    token_ids: torch.Tensor
    stats: DecodeStats


def _validate_inputs(images: torch.Tensor, max_new_tokens: int) -> None:
    if images.shape[0] != 1:
        raise ValueError("correctness-first decoding currently requires batch size 1")
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")


def _ar_logits(model: torch.nn.Module, visual_tokens: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    mask = torch.ones_like(ids, dtype=torch.bool)
    return model.decode_text(
        input_ids=ids,
        visual_tokens=visual_tokens,
        attention_mask=mask,
        mode="ar",
    ).logits


@torch.inference_mode()
def greedy_ar_decode(
    model: torch.nn.Module,
    images: torch.Tensor,
    *,
    query_token_id: int,
    max_new_tokens: int,
    eos_token_id: int | None = None,
) -> DecodeResult:
    """True greedy AR generation without a KV cache.

    A trailing query token creates the next-token logit. Its embedding cannot affect that logit
    because the AR output is shifted by one position.
    """

    _validate_inputs(images, max_new_tokens)
    visual_tokens = model.encode_images(images)
    generated = torch.empty((1, 0), dtype=torch.long, device=images.device)
    forwards = 0
    for _ in range(max_new_tokens):
        query = torch.full((1, 1), query_token_id, dtype=torch.long, device=images.device)
        logits = _ar_logits(model, visual_tokens, torch.cat([generated, query], dim=1))
        next_token = logits[:, generated.shape[1]].argmax(dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=1)
        forwards += 1
        if eos_token_id is not None and int(next_token.item()) == eos_token_id:
            break
    return DecodeResult(
        token_ids=generated,
        stats=DecodeStats(
            cycles=forwards,
            draft_forwards=0,
            verify_forwards=forwards,
            accepted_draft_tokens=0,
            committed_tokens=generated.shape[1],
            proposed_draft_tokens=0,
            zero_acceptance_cycles=0,
            accepted_by_offset=(),
            proposed_by_offset=(),
        ),
    )


@torch.inference_mode()
def linear_self_speculative_decode(
    model: torch.nn.Module,
    images: torch.Tensor,
    *,
    mask_token_id: int,
    query_token_id: int,
    diffusion_timestep: int,
    draft_width: int,
    max_new_tokens: int,
    eos_token_id: int | None = None,
) -> DecodeResult:
    """Uncached exact-greedy self-speculation using one all-mask draft block per cycle."""

    _validate_inputs(images, max_new_tokens)
    if draft_width <= 0:
        raise ValueError("draft_width must be positive")
    visual_tokens = model.encode_images(images)
    generated = torch.empty((1, 0), dtype=torch.long, device=images.device)
    accepted_by_offset = [0] * draft_width
    proposed_by_offset = [0] * draft_width
    accepted_total = 0
    proposed_total = 0
    zero_acceptance = 0
    cycles = 0

    while generated.shape[1] < max_new_tokens:
        width = min(draft_width, max_new_tokens - generated.shape[1])
        noisy = torch.full((1, width), mask_token_id, dtype=torch.long, device=images.device)
        clean_mask = torch.ones_like(generated, dtype=torch.bool)
        block_mask = torch.ones_like(noisy, dtype=torch.bool)
        draft_output = model.decode_dual_stream(
            clean_input_ids=generated,
            noisy_block_ids=noisy,
            block_starts=torch.tensor([generated.shape[1]], device=images.device),
            visual_tokens=visual_tokens,
            clean_attention_mask=clean_mask,
            noisy_block_mask=block_mask,
            timestep=torch.tensor([diffusion_timestep], device=images.device),
        )
        draft = draft_output.diffusion_logits.argmax(dim=-1)

        verify_ids = torch.cat([generated, draft], dim=1)
        verify_logits = _ar_logits(model, visual_tokens, verify_ids)
        verifier = verify_logits[:, generated.shape[1] : generated.shape[1] + width].argmax(dim=-1)
        matches = (draft == verifier)[0]
        mismatch = (~matches).nonzero(as_tuple=False)
        accepted = width if mismatch.numel() == 0 else int(mismatch[0, 0])
        if eos_token_id is not None and accepted > 0:
            accepted_eos = (draft[0, :accepted] == eos_token_id).nonzero(as_tuple=False)
            if accepted_eos.numel() > 0:
                accepted = int(accepted_eos[0, 0]) + 1

        proposed_total += width
        accepted_total += accepted
        for offset in range(width):
            proposed_by_offset[offset] += 1
        for offset in range(accepted):
            accepted_by_offset[offset] += 1
        if accepted == 0:
            zero_acceptance += 1

        committed = draft[:, :accepted]
        accepted_ends_sequence = (
            eos_token_id is not None
            and accepted > 0
            and int(committed[0, -1].item()) == eos_token_id
        )
        if accepted < width and not accepted_ends_sequence:
            committed = torch.cat([committed, verifier[:, accepted : accepted + 1]], dim=1)
        generated = torch.cat([generated, committed], dim=1)
        cycles += 1

        if eos_token_id is not None:
            eos_positions = (generated[0] == eos_token_id).nonzero(as_tuple=False)
            if eos_positions.numel() > 0:
                generated = generated[:, : int(eos_positions[0, 0]) + 1]
                break

    return DecodeResult(
        token_ids=generated,
        stats=DecodeStats(
            cycles=cycles,
            draft_forwards=cycles,
            verify_forwards=cycles,
            accepted_draft_tokens=accepted_total,
            committed_tokens=generated.shape[1],
            proposed_draft_tokens=proposed_total,
            zero_acceptance_cycles=zero_acceptance,
            accepted_by_offset=tuple(accepted_by_offset),
            proposed_by_offset=tuple(proposed_by_offset),
        ),
    )

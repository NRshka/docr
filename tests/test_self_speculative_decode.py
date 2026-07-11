from types import SimpleNamespace

import torch
from torch import nn

from docr.inference.self_speculative import greedy_ar_decode, linear_self_speculative_decode


class ScriptedModel(nn.Module):
    def __init__(self, target: list[int], drafts: list[list[int]]) -> None:
        super().__init__()
        self.target = target
        self.drafts = drafts
        self.draft_index = 0

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        return images[:, :1, :1, :1].flatten(1).unsqueeze(1)

    def decode_text(self, input_ids, visual_tokens, attention_mask=None, mode="ar"):
        del visual_tokens, attention_mask, mode
        logits = torch.full((1, input_ids.shape[1], 32), -10.0)
        for position in range(input_ids.shape[1]):
            logits[0, position, self.target[position]] = 10.0
        return SimpleNamespace(logits=logits)

    def decode_dual_stream(self, noisy_block_ids, **kwargs):
        del kwargs
        draft = self.drafts[self.draft_index][: noisy_block_ids.shape[1]]
        self.draft_index += 1
        logits = torch.full((1, len(draft), 32), -10.0)
        for position, token in enumerate(draft):
            logits[0, position, token] = 10.0
        return SimpleNamespace(diffusion_logits=logits)


def test_greedy_ar_decode_generates_until_eos():
    model = ScriptedModel([4, 5, 2, 9], [])
    result = greedy_ar_decode(
        model, torch.zeros(1, 3, 2, 2), query_token_id=0, max_new_tokens=4, eos_token_id=2
    )
    assert result.token_ids.tolist() == [[4, 5, 2]]
    assert result.stats.verify_forwards == 3


def test_self_speculation_accepts_prefix_and_commits_first_rejection():
    model = ScriptedModel([4, 5, 6, 7, 2, 9, 9, 9], [[4, 5, 9], [7, 2, 8]])
    result = linear_self_speculative_decode(
        model,
        torch.zeros(1, 3, 2, 2),
        mask_token_id=31,
        query_token_id=0,
        diffusion_timestep=3,
        draft_width=3,
        max_new_tokens=8,
        eos_token_id=2,
    )
    assert result.token_ids.tolist() == [[4, 5, 6, 7, 2]]
    assert result.stats.cycles == 2
    assert result.stats.accepted_draft_tokens == 4
    assert result.stats.accepted_by_offset == (2, 2, 0)
    assert result.stats.tokens_per_forward == 1.25


def test_self_speculation_handles_zero_acceptance():
    model = ScriptedModel([4, 5], [[9, 9], [8]])
    result = linear_self_speculative_decode(
        model,
        torch.zeros(1, 3, 2, 2),
        mask_token_id=31,
        query_token_id=0,
        diffusion_timestep=3,
        draft_width=2,
        max_new_tokens=2,
    )
    assert result.token_ids.tolist() == [[4, 5]]
    assert result.stats.zero_acceptance_cycles == 2

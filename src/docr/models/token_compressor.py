from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class MeanTokenCompressor(nn.Module):
    """Simple fixed-query compressor for scaffold experiments."""

    def __init__(self, input_dim: int, output_tokens: int) -> None:
        super().__init__()
        self.output_tokens = output_tokens
        self.proj = nn.Linear(input_dim, input_dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 3:
            raise ValueError("tokens must have shape [batch, seq, dim]")
        pooled = tokens.mean(dim=1, keepdim=True)
        compressed = pooled.expand(tokens.shape[0], self.output_tokens, tokens.shape[-1])
        return self.proj(compressed)


class FeedForward(nn.Module):
    def __init__(self, hidden_size: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        inner_size = int(hidden_size * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(hidden_size, inner_size),
            nn.GELU(),
            nn.Linear(inner_size, hidden_size),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.net(tokens)


class PerceiverResamplerLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(hidden_size)
        self.context_norm = nn.LayerNorm(hidden_size)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.self_attn_norm = nn.LayerNorm(hidden_size)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = FeedForward(hidden_size=hidden_size, mlp_ratio=mlp_ratio)

    def forward(self, queries: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        cross_out, _ = self.cross_attn(
            query=self.query_norm(queries),
            key=self.context_norm(context),
            value=self.context_norm(context),
            need_weights=False,
        )
        queries = queries + cross_out

        self_out, _ = self.self_attn(
            query=self.self_attn_norm(queries),
            key=self.self_attn_norm(queries),
            value=self.self_attn_norm(queries),
            need_weights=False,
        )
        queries = queries + self_out
        return queries + self.ffn(self.ffn_norm(queries))


class PerceiverTokenCompressor(nn.Module):
    """Learned-query resampler for compressing patch tokens into visual memory tokens."""

    def __init__(
        self,
        input_dim: int,
        output_tokens: int,
        num_layers: int = 2,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if input_dim % num_heads != 0:
            raise ValueError("input_dim must be divisible by num_heads")
        self.output_tokens = output_tokens
        self.query_tokens = nn.Parameter(torch.empty(output_tokens, input_dim))
        self.layers = nn.ModuleList(
            [
                PerceiverResamplerLayer(
                    hidden_size=input_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.output_norm = nn.LayerNorm(input_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.query_tokens, std=0.02)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 3:
            raise ValueError("tokens must have shape [batch, seq, dim]")
        queries = self.query_tokens.unsqueeze(0).expand(tokens.shape[0], -1, -1)
        for layer in self.layers:
            queries = layer(queries, tokens)
        return self.output_norm(queries)


class Learned2DPositionEmbedding(nn.Module):
    def __init__(self, hidden_size: int, max_grid_size: int = 256) -> None:
        super().__init__()
        self.row_embed = nn.Embedding(max_grid_size, hidden_size)
        self.col_embed = nn.Embedding(max_grid_size, hidden_size)

    def forward(self, grid_height: int, grid_width: int, device: torch.device) -> torch.Tensor:
        if grid_height > self.row_embed.num_embeddings or grid_width > self.col_embed.num_embeddings:
            raise ValueError(
                "Patch grid exceeds learned 2D position capacity: "
                f"{grid_height}x{grid_width} > {self.row_embed.num_embeddings}"
            )
        rows = torch.arange(grid_height, device=device)
        cols = torch.arange(grid_width, device=device)
        row_pos = self.row_embed(rows)[:, None, :]
        col_pos = self.col_embed(cols)[None, :, :]
        pos = row_pos + col_pos
        return pos.reshape(1, grid_height * grid_width, -1)


def resize_abs_pos_embedding(
    pos_embedding: torch.Tensor,
    grid_height: int,
    grid_width: int,
) -> torch.Tensor:
    """Resize [1, seq, dim] absolute position embeddings to a patch grid."""

    old_seq = pos_embedding.shape[1]
    old_size = int(old_seq**0.5)
    if old_size * old_size != old_seq:
        raise ValueError("Absolute position embedding length must be a square grid")
    pos = pos_embedding.reshape(1, old_size, old_size, -1).permute(0, 3, 1, 2)
    pos = F.interpolate(pos, size=(grid_height, grid_width), mode="bicubic", align_corners=False)
    return pos.permute(0, 2, 3, 1).reshape(1, grid_height * grid_width, -1)

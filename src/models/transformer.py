"""
Transformer blocks for cross-attention and self-attention in UNet.
Enables multi-modal conditioning and spatial reasoning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from einops import rearrange


class MultiHeadAttention(nn.Module):
    """Multi-head attention with support for self-attention and cross-attention."""

    def __init__(
        self,
        query_dim: int,
        num_heads: int = 8,
        head_dim: int = 64,
        dropout: float = 0.0,
        bias: bool = False,
        cross_attention_dim: Optional[int] = None,
    ):
        super().__init__()
        inner_dim = head_dim * num_heads
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5

        context_dim = cross_attention_dim if cross_attention_dim is not None else query_dim

        self.to_q = nn.Linear(query_dim, inner_dim, bias=bias)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=bias)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=bias)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: (B, N, query_dim) query tensor
            encoder_hidden_states: (B, M, cross_attention_dim) key/value tensor for cross-attention
            attention_mask: (B, N, M) attention mask
        Returns:
            (B, N, query_dim) output tensor
        """
        batch_size = hidden_states.shape[0]

        # Compute queries
        query = self.to_q(hidden_states)

        # Compute keys and values (cross-attention or self-attention)
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        key = self.to_k(encoder_hidden_states)
        value = self.to_v(encoder_hidden_states)

        # Reshape for multi-head attention
        query = rearrange(query, 'b n (h d) -> b h n d', h=self.num_heads)
        key = rearrange(key, 'b m (h d) -> b h m d', h=self.num_heads)
        value = rearrange(value, 'b m (h d) -> b h m d', h=self.num_heads)

        # Compute attention scores
        attention_scores = torch.matmul(query, key.transpose(-1, -2)) * self.scale

        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        attention_probs = F.softmax(attention_scores, dim=-1)

        # Apply attention to values
        hidden_states = torch.matmul(attention_probs, value)
        hidden_states = rearrange(hidden_states, 'b h n d -> b n (h d)')

        # Final projection
        hidden_states = self.to_out(hidden_states)

        return hidden_states


class FeedForward(nn.Module):
    """Position-wise feed-forward network with GEGLU activation."""

    def __init__(
        self,
        dim: int,
        dim_out: Optional[int] = None,
        mult: int = 4,
        dropout: float = 0.0,
        activation_fn: str = 'geglu',
    ):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim

        if activation_fn == 'geglu':
            self.project_in = GEGLU(dim, inner_dim)
        elif activation_fn == 'gelu':
            self.project_in = nn.Sequential(
                nn.Linear(dim, inner_dim),
                nn.GELU(),
            )
        else:
            raise ValueError(f'Unsupported activation: {activation_fn}')

        self.dropout = nn.Dropout(dropout)
        self.project_out = nn.Linear(inner_dim, dim_out)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.project_in(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.project_out(hidden_states)
        return hidden_states


class GEGLU(nn.Module):
    """Gated GLU activation function."""

    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class TransformerBlock(nn.Module):
    """
    Transformer block with self-attention, cross-attention, and feed-forward.

    This is the core building block for spatial reasoning and multi-modal fusion.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout: float = 0.0,
        cross_attention_dim: Optional[int] = None,
        activation_fn: str = 'geglu',
        only_cross_attention: bool = False,
    ):
        super().__init__()
        self.only_cross_attention = only_cross_attention

        # Self-attention (optional)
        if not only_cross_attention:
            self.attn1 = MultiHeadAttention(
                query_dim=dim,
                num_heads=num_attention_heads,
                head_dim=attention_head_dim,
                dropout=dropout,
            )
            self.norm1 = nn.LayerNorm(dim)

        # Cross-attention
        self.attn2 = MultiHeadAttention(
            query_dim=dim,
            num_heads=num_attention_heads,
            head_dim=attention_head_dim,
            dropout=dropout,
            cross_attention_dim=cross_attention_dim,
        )
        self.norm2 = nn.LayerNorm(dim)

        # Feed-forward
        self.ff = FeedForward(
            dim=dim,
            dropout=dropout,
            activation_fn=activation_fn,
        )
        self.norm3 = nn.LayerNorm(dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: (B, N, dim) input tensor
            encoder_hidden_states: (B, M, cross_attention_dim) conditioning tensor
            attention_mask: attention mask
        Returns:
            (B, N, dim) output tensor
        """
        # Self-attention
        if not self.only_cross_attention:
            norm_hidden_states = self.norm1(hidden_states)
            attn_output = self.attn1(norm_hidden_states)
            hidden_states = hidden_states + attn_output

        # Cross-attention
        norm_hidden_states = self.norm2(hidden_states)
        attn_output = self.attn2(
            norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
        )
        hidden_states = hidden_states + attn_output

        # Feed-forward
        norm_hidden_states = self.norm3(hidden_states)
        ff_output = self.ff(norm_hidden_states)
        hidden_states = hidden_states + ff_output

        return hidden_states


class SpatialTransformer(nn.Module):
    """
    Spatial transformer for 2D feature maps.

    Converts spatial features to sequence, applies transformer blocks,
    and converts back to spatial format.
    """

    def __init__(
        self,
        in_channels: int,
        num_attention_heads: int,
        attention_head_dim: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        cross_attention_dim: Optional[int] = None,
    ):
        super().__init__()
        inner_dim = num_attention_heads * attention_head_dim

        self.norm = nn.GroupNorm(32, in_channels, eps=1e-6)
        self.proj_in = nn.Conv2d(in_channels, inner_dim, kernel_size=1)

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                dim=inner_dim,
                num_attention_heads=num_attention_heads,
                attention_head_dim=attention_head_dim,
                dropout=dropout,
                cross_attention_dim=cross_attention_dim,
            )
            for _ in range(num_layers)
        ])

        self.proj_out = nn.Conv2d(inner_dim, in_channels, kernel_size=1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: (B, C, H, W) spatial features
            encoder_hidden_states: (B, seq_len, cross_attention_dim) conditioning
        Returns:
            (B, C, H, W) transformed features
        """
        batch, channel, height, weight = hidden_states.shape
        residual = hidden_states

        # Normalize and project
        hidden_states = self.norm(hidden_states)
        hidden_states = self.proj_in(hidden_states)

        # Reshape to sequence: (B, C, H, W) -> (B, H*W, C)
        hidden_states = rearrange(hidden_states, 'b c h w -> b (h w) c')

        # Apply transformer blocks
        for block in self.transformer_blocks:
            hidden_states = block(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
            )

        # Reshape back to spatial: (B, H*W, C) -> (B, C, H, W)
        hidden_states = rearrange(
            hidden_states, 'b (h w) c -> b c h w',
            h=height, w=weight
        )

        # Project out
        hidden_states = self.proj_out(hidden_states)

        return hidden_states + residual

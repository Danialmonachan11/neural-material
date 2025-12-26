"""
Cross-attention fusion for multi-modal conditioning.

Combines text, material, and spatial conditioning through attention mechanisms.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict
from einops import rearrange


class CrossAttentionFusion(nn.Module):
    """
    Multi-modal cross-attention fusion module.

    Fuses different conditioning modalities (text, material, spatial)
    through hierarchical cross-attention.
    """

    def __init__(
        self,
        query_dim: int = 768,
        num_heads: int = 8,
        head_dim: int = 64,
        dropout: float = 0.0,
        num_modalities: int = 3,  # text, material, spatial
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        self.num_modalities = num_modalities

        inner_dim = head_dim * num_heads

        # Separate Q, K, V projections for each modality
        self.modality_projections = nn.ModuleDict({
            'text': self._create_qkv_proj(query_dim, inner_dim),
            'material': self._create_qkv_proj(query_dim, inner_dim),
            'spatial': self._create_qkv_proj(query_dim, inner_dim),
        })

        # Modality-specific attention weights (learnable)
        self.modality_weights = nn.Parameter(torch.ones(num_modalities))

        # Output projection
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout),
        )

        # Gating mechanism for adaptive fusion
        self.fusion_gate = nn.Sequential(
            nn.Linear(query_dim * num_modalities, query_dim),
            nn.Sigmoid(),
        )

    def _create_qkv_proj(self, dim: int, inner_dim: int) -> nn.ModuleDict:
        """Create Q, K, V projections for a modality."""
        return nn.ModuleDict({
            'to_q': nn.Linear(dim, inner_dim, bias=False),
            'to_k': nn.Linear(dim, inner_dim, bias=False),
            'to_v': nn.Linear(dim, inner_dim, bias=False),
        })

    def forward(
        self,
        query: torch.Tensor,
        text_context: Optional[torch.Tensor] = None,
        material_context: Optional[torch.Tensor] = None,
        spatial_context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Multi-modal cross-attention fusion.

        Args:
            query: (B, N, dim) query features
            text_context: (B, M_text, dim) text conditioning
            material_context: (B, M_mat, dim) material conditioning
            spatial_context: (B, M_spatial, dim) spatial conditioning
        Returns:
            (B, N, dim) fused features
        """
        batch_size, seq_len, _ = query.shape

        contexts = {
            'text': text_context,
            'material': material_context,
            'spatial': spatial_context,
        }

        # Compute attention for each modality
        attended_features = []
        modality_outputs = []

        for i, (modality_name, context) in enumerate(contexts.items()):
            if context is not None:
                # Get projections for this modality
                proj = self.modality_projections[modality_name]

                # Compute Q, K, V
                q = proj['to_q'](query)
                k = proj['to_k'](context)
                v = proj['to_v'](context)

                # Reshape for multi-head attention
                q = rearrange(q, 'b n (h d) -> b h n d', h=self.num_heads)
                k = rearrange(k, 'b m (h d) -> b h m d', h=self.num_heads)
                v = rearrange(v, 'b m (h d) -> b h m d', h=self.num_heads)

                # Compute attention
                attn_scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale
                attn_probs = F.softmax(attn_scores, dim=-1)
                attn_out = torch.matmul(attn_probs, v)

                # Reshape back
                attn_out = rearrange(attn_out, 'b h n d -> b n (h d)')

                # Apply modality weight
                weight = torch.sigmoid(self.modality_weights[i])
                attended = attn_out * weight

                attended_features.append(attended)
                modality_outputs.append(attn_out)

        if len(attended_features) == 0:
            # No conditioning, return query
            return query

        # Combine attended features from all modalities
        if len(attended_features) == 1:
            combined = attended_features[0]
        else:
            # Adaptive fusion using gating
            concatenated = torch.cat(modality_outputs, dim=-1)
            gate = self.fusion_gate(concatenated)

            # Weighted sum with gating
            combined = sum(attended_features)
            combined = combined * gate

        # Final projection
        output = self.to_out(combined)

        return output


class HierarchicalFusion(nn.Module):
    """
    Hierarchical fusion of multi-modal conditions.

    First fuses related modalities, then combines them hierarchically.
    """

    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 8,
        num_fusion_layers: int = 2,
    ):
        super().__init__()

        # Level 1: Fuse text and material (semantic fusion)
        self.semantic_fusion = CrossAttentionFusion(
            query_dim=embed_dim,
            num_heads=num_heads,
            num_modalities=2,
        )

        # Level 2: Fuse semantic features with spatial (spatial fusion)
        self.spatial_fusion = CrossAttentionFusion(
            query_dim=embed_dim,
            num_heads=num_heads,
            num_modalities=2,
        )

        # Refinement layers
        self.refinement = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=embed_dim * 4,
                batch_first=True,
            )
            for _ in range(num_fusion_layers)
        ])

    def forward(
        self,
        features: torch.Tensor,
        text_context: Optional[torch.Tensor] = None,
        material_context: Optional[torch.Tensor] = None,
        spatial_context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Hierarchical fusion.

        Args:
            features: (B, N, dim) base features
            text_context: text conditioning
            material_context: material conditioning
            spatial_context: spatial conditioning
        Returns:
            (B, N, dim) hierarchically fused features
        """
        # Level 1: Semantic fusion (text + material)
        if text_context is not None or material_context is not None:
            semantic_features = self.semantic_fusion(
                query=features,
                text_context=text_context,
                material_context=material_context,
            )
        else:
            semantic_features = features

        # Level 2: Spatial fusion
        if spatial_context is not None:
            fused_features = self.spatial_fusion(
                query=semantic_features,
                spatial_context=spatial_context,
            )
        else:
            fused_features = semantic_features

        # Refinement
        refined = fused_features
        for layer in self.refinement:
            refined = layer(refined)

        return refined


class AdaptiveFusionGate(nn.Module):
    """
    Adaptive gating mechanism for controlling fusion strength.

    Learns to weight different modalities based on input.
    """

    def __init__(
        self,
        embed_dim: int = 768,
        num_modalities: int = 3,
    ):
        super().__init__()
        self.num_modalities = num_modalities

        # Gate network
        self.gate_network = nn.Sequential(
            nn.Linear(embed_dim * num_modalities, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, num_modalities),
            nn.Softmax(dim=-1),
        )

    def forward(
        self,
        modality_features: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute adaptive fusion weights.

        Args:
            modality_features: list of (B, N, dim) features per modality
        Returns:
            (B, N, dim) fused features
        """
        # Concatenate modality features
        concatenated = torch.cat(modality_features, dim=-1)  # (B, N, dim * num_modalities)

        # Compute fusion weights
        weights = self.gate_network(concatenated)  # (B, N, num_modalities)

        # Weighted fusion
        stacked = torch.stack(modality_features, dim=-1)  # (B, N, dim, num_modalities)
        weights = weights.unsqueeze(2)  # (B, N, 1, num_modalities)

        fused = (stacked * weights).sum(dim=-1)  # (B, N, dim)

        return fused


class ContextAwareAttention(nn.Module):
    """
    Context-aware attention that adapts based on conditioning strength.

    Dynamically adjusts attention patterns based on condition availability.
    """

    def __init__(
        self,
        dim: int = 768,
        num_heads: int = 8,
        context_dim: Optional[int] = None,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        context_dim = context_dim or dim

        # Condition strength predictor
        self.strength_predictor = nn.Sequential(
            nn.Linear(context_dim, dim // 4),
            nn.GELU(),
            nn.Linear(dim // 4, 1),
            nn.Sigmoid(),
        )

        # Standard attention
        self.attention = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            batch_first=True,
        )

        # Unconditional pathway
        self.unconditional_proj = nn.Linear(dim, dim)

    def forward(
        self,
        query: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Context-aware attention.

        Args:
            query: (B, N, dim) query
            context: (B, M, context_dim) context (optional)
        Returns:
            (B, N, dim) output
        """
        if context is None:
            # No context, use unconditional pathway
            return self.unconditional_proj(query)

        # Predict conditioning strength
        strength = self.strength_predictor(context.mean(dim=1, keepdim=True))  # (B, 1, 1)

        # Conditional attention
        conditional_out, _ = self.attention(query, context, context)

        # Unconditional pathway
        unconditional_out = self.unconditional_proj(query)

        # Blend based on strength
        output = strength * conditional_out + (1 - strength) * unconditional_out

        return output

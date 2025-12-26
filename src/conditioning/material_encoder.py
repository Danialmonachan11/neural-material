"""
Material and texture encoding for material-aware diffusion models.

Encodes material properties (BRDF parameters, texture descriptors) into
conditioning signals for the diffusion model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple
from einops import rearrange
import timm


class MaterialEncoder(nn.Module):
    """
    Encodes material properties and textures into embedding space.

    Supports both parameter-based (BRDF) and image-based (texture) materials.
    """

    def __init__(
        self,
        embed_dim: int = 768,
        num_material_params: int = 16,  # BRDF parameters
        use_texture_encoder: bool = True,
        pretrained_vision_model: str = 'resnet50',
    ):
        """
        Args:
            embed_dim: embedding dimension
            num_material_params: number of BRDF parameters
            use_texture_encoder: whether to encode texture images
            pretrained_vision_model: vision backbone for texture encoding
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.use_texture_encoder = use_texture_encoder

        # Parameter-based material encoder (for BRDF parameters)
        self.param_encoder = nn.Sequential(
            nn.Linear(num_material_params, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, embed_dim),
        )

        # Texture-based encoder (for texture images)
        if use_texture_encoder:
            # Use pretrained vision model as backbone
            self.texture_backbone = timm.create_model(
                pretrained_vision_model,
                pretrained=True,
                features_only=True,
                out_indices=[2, 3, 4],  # Multi-scale features
            )

            # Get feature dimensions from backbone
            backbone_channels = self.texture_backbone.feature_info.channels()

            # Multi-scale feature fusion
            self.feature_fusion = nn.ModuleList([
                nn.Conv2d(c, embed_dim // 4, kernel_size=1)
                for c in backbone_channels
            ])

            # Final projection
            self.texture_proj = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(embed_dim // 4 * 3, embed_dim),
            )

        # Material category embeddings (learned embeddings for material types)
        self.material_categories = ['metal', 'plastic', 'wood', 'fabric', 'glass', 'stone', 'other']
        self.category_embeddings = nn.Embedding(len(self.material_categories), embed_dim)

        # Fusion network for combining different material signals
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def encode_parameters(self, params: torch.Tensor) -> torch.Tensor:
        """
        Encode BRDF parameters.

        Args:
            params: (B, num_material_params) material parameters
                   e.g., [roughness, metallic, specular, diffuse_r, diffuse_g, diffuse_b, ...]
        Returns:
            (B, embed_dim) parameter embeddings
        """
        return self.param_encoder(params)

    def encode_texture(self, texture: torch.Tensor) -> torch.Tensor:
        """
        Encode texture image.

        Args:
            texture: (B, 3, H, W) texture image
        Returns:
            (B, embed_dim) texture embeddings
        """
        if not self.use_texture_encoder:
            raise ValueError("Texture encoder not enabled")

        # Extract multi-scale features
        features = self.texture_backbone(texture)

        # Fuse features from different scales
        fused_features = []
        for feat, fusion_layer in zip(features, self.feature_fusion):
            fused = fusion_layer(feat)
            fused_features.append(fused)

        # Concatenate and project
        combined = torch.cat(fused_features, dim=1)
        texture_embed = self.texture_proj(combined)

        return texture_embed

    def encode_category(self, categories: torch.Tensor) -> torch.Tensor:
        """
        Encode material category.

        Args:
            categories: (B,) material category indices
        Returns:
            (B, embed_dim) category embeddings
        """
        return self.category_embeddings(categories)

    def forward(
        self,
        params: Optional[torch.Tensor] = None,
        texture: Optional[torch.Tensor] = None,
        category: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode material from multiple modalities.

        Args:
            params: (B, num_params) BRDF parameters
            texture: (B, 3, H, W) texture image
            category: (B,) material category
        Returns:
            (B, embed_dim) material embedding
        """
        embeddings = []

        # Encode parameters
        if params is not None:
            param_embed = self.encode_parameters(params)
            embeddings.append(param_embed)

        # Encode texture
        if texture is not None:
            texture_embed = self.encode_texture(texture)
            embeddings.append(texture_embed)

        # Encode category
        if category is not None:
            category_embed = self.encode_category(category)
            embeddings.append(category_embed)

        if len(embeddings) == 0:
            raise ValueError("At least one material input must be provided")

        # Combine embeddings
        if len(embeddings) == 1:
            return embeddings[0]
        else:
            # Fuse multiple modalities
            combined = torch.cat(embeddings[:2], dim=1)  # Use first two
            fused = self.fusion(combined)

            # Add any additional embeddings as residuals
            for embed in embeddings[2:]:
                fused = fused + embed

            return fused


class MaterialConditioner(nn.Module):
    """
    Conditions the diffusion model on material properties.

    Integrates material embeddings into the UNet through cross-attention.
    """

    def __init__(
        self,
        material_embed_dim: int = 768,
        cross_attention_dim: int = 768,
        num_layers: int = 3,
        num_heads: int = 8,
    ):
        super().__init__()

        # Project material embeddings to cross-attention dimension
        self.material_proj = nn.Linear(material_embed_dim, cross_attention_dim)

        # Material-aware cross-attention layers
        self.cross_attention_layers = nn.ModuleList([
            MaterialCrossAttention(
                query_dim=cross_attention_dim,
                num_heads=num_heads,
            )
            for _ in range(num_layers)
        ])

        # Layer normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(cross_attention_dim)
            for _ in range(num_layers)
        ])

    def forward(
        self,
        hidden_states: torch.Tensor,
        material_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply material conditioning.

        Args:
            hidden_states: (B, seq_len, dim) feature sequence
            material_embeddings: (B, material_embed_dim) material embeddings
        Returns:
            (B, seq_len, dim) conditioned features
        """
        # Project material embeddings
        material_features = self.material_proj(material_embeddings)  # (B, dim)
        material_features = material_features.unsqueeze(1)  # (B, 1, dim)

        # Apply cross-attention layers
        output = hidden_states
        for cross_attn, norm in zip(self.cross_attention_layers, self.layer_norms):
            residual = output
            output = norm(output)
            output = cross_attn(output, material_features)
            output = output + residual

        return output


class MaterialCrossAttention(nn.Module):
    """Cross-attention layer for material conditioning."""

    def __init__(
        self,
        query_dim: int,
        num_heads: int = 8,
        head_dim: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()
        inner_dim = head_dim * num_heads
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(query_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        material_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: (B, N, dim) queries
            material_features: (B, M, dim) material keys/values
        Returns:
            (B, N, dim) attended features
        """
        batch_size = hidden_states.shape[0]

        # Compute Q, K, V
        q = self.to_q(hidden_states)
        k = self.to_k(material_features)
        v = self.to_v(material_features)

        # Reshape for multi-head attention
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.num_heads)
        k = rearrange(k, 'b m (h d) -> b h m d', h=self.num_heads)
        v = rearrange(v, 'b m (h d) -> b h m d', h=self.num_heads)

        # Attention
        attn_scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn_probs = F.softmax(attn_scores, dim=-1)

        # Apply attention to values
        out = torch.matmul(attn_probs, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        # Project output
        out = self.to_out(out)

        return out


class TextureSynthesizer(nn.Module):
    """
    Synthesizes texture details based on material properties.

    Adds fine-grained texture patterns to generated images.
    """

    def __init__(
        self,
        in_channels: int = 4,
        material_embed_dim: int = 768,
        texture_channels: int = 64,
    ):
        super().__init__()

        # Material-conditioned texture generator
        self.material_to_texture = nn.Sequential(
            nn.Linear(material_embed_dim, 256),
            nn.GELU(),
            nn.Linear(256, texture_channels * 4 * 4),
        )

        # Upsampling network for texture details
        self.texture_upsample = nn.Sequential(
            nn.Conv2d(texture_channels, texture_channels, 3, 1, 1),
            nn.GroupNorm(8, texture_channels),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode='nearest'),

            nn.Conv2d(texture_channels, texture_channels // 2, 3, 1, 1),
            nn.GroupNorm(8, texture_channels // 2),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode='nearest'),

            nn.Conv2d(texture_channels // 2, texture_channels // 4, 3, 1, 1),
            nn.GroupNorm(8, texture_channels // 4),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode='nearest'),

            nn.Conv2d(texture_channels // 4, in_channels, 3, 1, 1),
        )

    def forward(
        self,
        latents: torch.Tensor,
        material_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Add material-specific texture details to latents.

        Args:
            latents: (B, C, H, W) latent features
            material_embeddings: (B, material_embed_dim) material embeddings
        Returns:
            (B, C, H, W) textured latents
        """
        batch_size = latents.shape[0]

        # Generate texture seed from material embeddings
        texture_seed = self.material_to_texture(material_embeddings)
        texture_seed = texture_seed.view(batch_size, -1, 4, 4)

        # Upsample texture to match latent size
        texture = self.texture_upsample(texture_seed)

        # Resize to exact latent size if needed
        if texture.shape[-2:] != latents.shape[-2:]:
            texture = F.interpolate(
                texture,
                size=latents.shape[-2:],
                mode='bilinear',
                align_corners=False,
            )

        # Add texture details to latents
        enhanced = latents + texture * 0.1  # Small residual for subtle texture

        return enhanced


def compute_brdf_params(
    roughness: float,
    metallic: float,
    base_color: Tuple[float, float, float],
    specular: float = 0.5,
) -> torch.Tensor:
    """
    Compute BRDF parameter vector from high-level material properties.

    Args:
        roughness: surface roughness [0, 1]
        metallic: metallic property [0, 1]
        base_color: RGB base color [0, 1]^3
        specular: specular intensity [0, 1]
    Returns:
        (16,) BRDF parameter tensor
    """
    params = torch.zeros(16)

    # Basic parameters
    params[0] = roughness
    params[1] = metallic
    params[2:5] = torch.tensor(base_color)
    params[5] = specular

    # Derived parameters (simplified)
    params[6] = roughness ** 2  # Alpha (roughness squared)
    params[7] = 1.0 - metallic  # Dielectric fraction
    params[8:11] = torch.tensor(base_color) * (1 - metallic)  # Diffuse color
    params[11:14] = torch.ones(3) * metallic  # Metallic specular

    # Additional PBR parameters
    params[14] = 0.04  # Base reflectance for dielectrics
    params[15] = 1.0  # IOR approximation

    return params

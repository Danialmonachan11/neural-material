"""
Region-based control for spatial conditioning in diffusion models.

Enables precise control over different regions of the generated image
through spatial masks and region-specific prompts.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from einops import rearrange


@dataclass
class RegionMask:
    """
    Region mask definition with associated prompt and strength.

    Attributes:
        mask: (H, W) binary mask defining the region
        prompt: text description for this region
        strength: control strength (0.0 to 1.0)
        layer_weights: optional per-layer attention weights
    """
    mask: torch.Tensor
    prompt: str
    strength: float = 1.0
    layer_weights: Optional[List[float]] = None

    def __post_init__(self):
        if self.mask.dim() == 2:
            self.mask = self.mask.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)


class RegionController(nn.Module):
    """
    Region-based spatial controller for diffusion models.

    Modulates attention based on spatial regions to enable
    precise control over different areas of the generation.
    """

    def __init__(
        self,
        num_layers: int = 16,
        interpolation_mode: str = 'bilinear',
        blur_sigma: float = 3.0,
    ):
        """
        Args:
            num_layers: number of UNet layers to control
            interpolation_mode: interpolation for mask resizing
            blur_sigma: gaussian blur sigma for smooth transitions
        """
        super().__init__()
        self.num_layers = num_layers
        self.interpolation_mode = interpolation_mode
        self.blur_sigma = blur_sigma

        # Learnable layer-wise weights for region control
        self.layer_weights = nn.Parameter(torch.ones(num_layers))

    def create_gaussian_kernel(
        self,
        kernel_size: int,
        sigma: float,
        channels: int = 1,
        device: torch.device = None,
    ) -> torch.Tensor:
        """Create 2D Gaussian kernel for blurring masks."""
        # Create 1D gaussian
        x = torch.arange(kernel_size, dtype=torch.float32, device=device)
        x = x - (kernel_size - 1) / 2
        gauss = torch.exp(-x.pow(2) / (2 * sigma ** 2))
        gauss = gauss / gauss.sum()

        # Create 2D gaussian
        kernel = gauss[:, None] * gauss[None, :]
        kernel = kernel[None, None, :, :]
        kernel = kernel.repeat(channels, 1, 1, 1)

        return kernel

    def blur_mask(self, mask: torch.Tensor) -> torch.Tensor:
        """Apply Gaussian blur to mask for smooth transitions."""
        if self.blur_sigma <= 0:
            return mask

        kernel_size = int(2 * self.blur_sigma + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1

        kernel = self.create_gaussian_kernel(
            kernel_size, self.blur_sigma,
            channels=mask.shape[1],
            device=mask.device
        )

        padding = kernel_size // 2
        blurred = F.conv2d(mask, kernel, padding=padding, groups=mask.shape[1])

        return blurred

    def resize_mask(
        self,
        mask: torch.Tensor,
        target_size: Tuple[int, int],
    ) -> torch.Tensor:
        """Resize mask to target spatial resolution."""
        if mask.shape[-2:] == target_size:
            return mask

        resized = F.interpolate(
            mask,
            size=target_size,
            mode=self.interpolation_mode,
            align_corners=False if self.interpolation_mode != 'nearest' else None,
        )

        return resized

    def compute_region_attention_mask(
        self,
        regions: List[RegionMask],
        height: int,
        width: int,
        layer_idx: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Compute combined attention mask for all regions.

        Args:
            regions: list of region masks
            height: target height
            width: target width
            layer_idx: current layer index
            device: torch device
        Returns:
            (num_regions, 1, H, W) combined mask
        """
        if not regions:
            return torch.ones(1, 1, height, width, device=device)

        masks = []
        for region in regions:
            # Resize and blur mask
            mask = self.resize_mask(region.mask.to(device), (height, width))
            mask = self.blur_mask(mask)

            # Apply region strength
            mask = mask * region.strength

            # Apply layer-specific weight
            layer_weight = self.layer_weights[layer_idx]
            if region.layer_weights is not None and layer_idx < len(region.layer_weights):
                layer_weight = layer_weight * region.layer_weights[layer_idx]

            mask = mask * layer_weight

            masks.append(mask)

        # Stack all region masks
        combined_mask = torch.cat(masks, dim=0)  # (num_regions, 1, H, W)

        # Normalize so regions sum to 1
        combined_mask = combined_mask / (combined_mask.sum(dim=0, keepdim=True) + 1e-8)

        return combined_mask

    def forward(
        self,
        hidden_states: torch.Tensor,
        regions: List[RegionMask],
        layer_idx: int,
    ) -> torch.Tensor:
        """
        Apply region-based modulation to hidden states.

        Args:
            hidden_states: (B, C, H, W) features
            regions: list of region specifications
            layer_idx: current layer index
        Returns:
            (B, C, H, W) modulated features
        """
        B, C, H, W = hidden_states.shape

        # Compute region attention mask
        region_mask = self.compute_region_attention_mask(
            regions, H, W, layer_idx, hidden_states.device
        )  # (num_regions, 1, H, W)

        # Apply region mask (spatial modulation)
        modulated = hidden_states.unsqueeze(1) * region_mask.unsqueeze(0)  # (B, num_regions, C, H, W)
        modulated = modulated.sum(dim=1)  # (B, C, H, W)

        return modulated


class MultiRegionController(nn.Module):
    """
    Advanced multi-region controller with conflict resolution.

    Handles overlapping regions and ensures coherent generation
    across region boundaries.
    """

    def __init__(
        self,
        base_channels: int = 320,
        num_layers: int = 16,
        conflict_resolution: str = 'blend',  # 'blend', 'priority', 'max'
    ):
        super().__init__()
        self.base_channels = base_channels
        self.num_layers = num_layers
        self.conflict_resolution = conflict_resolution

        # Region-specific feature projections
        self.region_projections = nn.ModuleList([
            nn.Conv2d(base_channels, base_channels, kernel_size=1)
            for _ in range(num_layers)
        ])

        # Boundary harmonization network
        self.boundary_harmonizer = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.GroupNorm(32, base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
        )

    def resolve_conflicts(
        self,
        region_features: List[torch.Tensor],
        region_masks: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Resolve conflicts in overlapping regions.

        Args:
            region_features: list of (B, C, H, W) features per region
            region_masks: list of (B, 1, H, W) masks per region
        Returns:
            (B, C, H, W) resolved features
        """
        if len(region_features) == 1:
            return region_features[0]

        if self.conflict_resolution == 'blend':
            # Weighted blend based on mask strength
            total_weight = torch.stack(region_masks, dim=0).sum(dim=0) + 1e-8
            weighted_features = sum(
                feat * mask for feat, mask in zip(region_features, region_masks)
            )
            resolved = weighted_features / total_weight

        elif self.conflict_resolution == 'priority':
            # Priority based on order (last region wins in conflicts)
            resolved = region_features[0]
            for feat, mask in zip(region_features[1:], region_masks[1:]):
                resolved = resolved * (1 - mask) + feat * mask

        elif self.conflict_resolution == 'max':
            # Max pooling across regions
            stacked = torch.stack(region_features, dim=0)
            resolved, _ = stacked.max(dim=0)

        else:
            raise ValueError(f'Unknown conflict resolution: {self.conflict_resolution}')

        return resolved

    def forward(
        self,
        hidden_states: torch.Tensor,
        region_embeddings: List[torch.Tensor],
        region_masks: List[RegionMask],
        layer_idx: int,
    ) -> torch.Tensor:
        """
        Multi-region forward pass with conflict resolution.

        Args:
            hidden_states: (B, C, H, W) base features
            region_embeddings: list of region-specific embeddings
            region_masks: list of region masks
            layer_idx: current layer index
        Returns:
            (B, C, H, W) region-controlled features
        """
        B, C, H, W = hidden_states.shape

        region_features = []
        masks = []

        # Process each region
        for i, (embedding, region_mask) in enumerate(zip(region_embeddings, region_masks)):
            # Resize mask to current resolution
            mask = F.interpolate(
                region_mask.mask,
                size=(H, W),
                mode='bilinear',
                align_corners=False,
            )

            # Project features for this region
            projected = self.region_projections[min(layer_idx, len(self.region_projections) - 1)](
                hidden_states
            )

            # Apply region-specific modulation
            region_feat = projected * mask

            region_features.append(region_feat)
            masks.append(mask)

        # Resolve conflicts and combine regions
        combined = self.resolve_conflicts(region_features, masks)

        # Harmonize boundaries
        harmonized = self.boundary_harmonizer(combined)

        # Residual connection
        output = hidden_states + harmonized

        return output


def create_circular_mask(
    height: int,
    width: int,
    center: Tuple[float, float],
    radius: float,
    device: torch.device = None,
) -> torch.Tensor:
    """Create a circular region mask."""
    y, x = torch.meshgrid(
        torch.arange(height, device=device, dtype=torch.float32),
        torch.arange(width, device=device, dtype=torch.float32),
        indexing='ij'
    )

    cy, cx = center[0] * height, center[1] * width
    r = radius * min(height, width)

    mask = ((y - cy) ** 2 + (x - cx) ** 2) <= r ** 2
    return mask.float().unsqueeze(0).unsqueeze(0)


def create_rectangular_mask(
    height: int,
    width: int,
    bbox: Tuple[float, float, float, float],  # (x1, y1, x2, y2) in [0, 1]
    device: torch.device = None,
) -> torch.Tensor:
    """Create a rectangular region mask."""
    mask = torch.zeros(1, 1, height, width, device=device)

    x1 = int(bbox[0] * width)
    y1 = int(bbox[1] * height)
    x2 = int(bbox[2] * width)
    y2 = int(bbox[3] * height)

    mask[:, :, y1:y2, x1:x2] = 1.0

    return mask

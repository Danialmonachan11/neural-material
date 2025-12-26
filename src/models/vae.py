"""
Variational Autoencoder (VAE) for latent diffusion.
Compresses images to latent space for efficient training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class ResnetBlock2D(nn.Module):
    """2D ResNet block for VAE encoder/decoder."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)
        return h + self.shortcut(x)


class Downsample2D(nn.Module):
    """Downsampling layer with convolution."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample2D(nn.Module):
    """Upsampling layer with nearest neighbor + convolution."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv(x)
        return x


class Encoder(nn.Module):
    """VAE Encoder that compresses images to latent space."""

    def __init__(
        self,
        in_channels: int = 3,
        latent_channels: int = 4,
        block_out_channels: Tuple[int, ...] = (128, 256, 512, 512),
        layers_per_block: int = 2,
    ):
        super().__init__()

        # Input convolution
        self.conv_in = nn.Conv2d(in_channels, block_out_channels[0], kernel_size=3, padding=1)

        # Downsampling blocks
        down_blocks = []
        output_channel = block_out_channels[0]

        for i, block_out_channel in enumerate(block_out_channels):
            input_channel = output_channel
            output_channel = block_out_channel
            is_final_block = i == len(block_out_channels) - 1

            for _ in range(layers_per_block):
                down_blocks.append(ResnetBlock2D(input_channel, output_channel))
                input_channel = output_channel

            if not is_final_block:
                down_blocks.append(Downsample2D(output_channel))

        self.down_blocks = nn.ModuleList(down_blocks)

        # Middle block
        self.mid_block = nn.ModuleList([
            ResnetBlock2D(block_out_channels[-1], block_out_channels[-1]),
            ResnetBlock2D(block_out_channels[-1], block_out_channels[-1]),
        ])

        # Output layers for mean and logvar
        self.conv_norm_out = nn.GroupNorm(32, block_out_channels[-1])
        self.conv_out = nn.Conv2d(
            block_out_channels[-1],
            2 * latent_channels,  # mean and logvar
            kernel_size=3,
            padding=1,
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, in_channels, H, W) input image
        Returns:
            mean: (B, latent_channels, H/8, W/8)
            logvar: (B, latent_channels, H/8, W/8)
        """
        x = self.conv_in(x)

        for block in self.down_blocks:
            x = block(x)

        for block in self.mid_block:
            x = block(x)

        x = self.conv_norm_out(x)
        x = F.silu(x)
        x = self.conv_out(x)

        mean, logvar = torch.chunk(x, 2, dim=1)
        return mean, logvar


class Decoder(nn.Module):
    """VAE Decoder that reconstructs images from latent space."""

    def __init__(
        self,
        out_channels: int = 3,
        latent_channels: int = 4,
        block_out_channels: Tuple[int, ...] = (128, 256, 512, 512),
        layers_per_block: int = 2,
    ):
        super().__init__()

        # Input convolution
        self.conv_in = nn.Conv2d(
            latent_channels,
            block_out_channels[-1],
            kernel_size=3,
            padding=1,
        )

        # Middle block
        self.mid_block = nn.ModuleList([
            ResnetBlock2D(block_out_channels[-1], block_out_channels[-1]),
            ResnetBlock2D(block_out_channels[-1], block_out_channels[-1]),
        ])

        # Upsampling blocks
        up_blocks = []
        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]

        for i, block_out_channel in enumerate(reversed_block_out_channels):
            input_channel = output_channel
            output_channel = block_out_channel
            is_final_block = i == len(block_out_channels) - 1

            for _ in range(layers_per_block):
                up_blocks.append(ResnetBlock2D(input_channel, output_channel))
                input_channel = output_channel

            if not is_final_block:
                up_blocks.append(Upsample2D(output_channel))

        self.up_blocks = nn.ModuleList(up_blocks)

        # Output convolution
        self.conv_norm_out = nn.GroupNorm(32, block_out_channels[0])
        self.conv_out = nn.Conv2d(block_out_channels[0], out_channels, kernel_size=3, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (B, latent_channels, H, W) latent tensor
        Returns:
            (B, out_channels, 8*H, 8*W) reconstructed image
        """
        z = self.conv_in(z)

        for block in self.mid_block:
            z = block(z)

        for block in self.up_blocks:
            z = block(z)

        z = self.conv_norm_out(z)
        z = F.silu(z)
        z = self.conv_out(z)

        return z


class AutoencoderKL(nn.Module):
    """
    Variational Autoencoder with KL divergence for latent diffusion.

    Compresses images by 8x in spatial dimensions (64x64 -> 8x8 latent).
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        latent_channels: int = 4,
        block_out_channels: Tuple[int, ...] = (128, 256, 512, 512),
        layers_per_block: int = 2,
        scaling_factor: float = 0.18215,
    ):
        super().__init__()
        self.scaling_factor = scaling_factor

        self.encoder = Encoder(
            in_channels=in_channels,
            latent_channels=latent_channels,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
        )

        self.decoder = Decoder(
            out_channels=out_channels,
            latent_channels=latent_channels,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
        )

        # Learned scaling for latent distribution
        self.quant_conv = nn.Conv2d(2 * latent_channels, 2 * latent_channels, kernel_size=1)
        self.post_quant_conv = nn.Conv2d(latent_channels, latent_channels, kernel_size=1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode image to latent space.

        Args:
            x: (B, in_channels, H, W) input image in [-1, 1]
        Returns:
            (B, latent_channels, H/8, W/8) latent tensor
        """
        mean, logvar = self.encoder(x)
        moments = self.quant_conv(torch.cat([mean, logvar], dim=1))
        mean, logvar = torch.chunk(moments, 2, dim=1)

        # Sample from distribution
        std = torch.exp(0.5 * logvar)
        z = mean + std * torch.randn_like(std)

        # Scale latent
        z = z * self.scaling_factor

        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent to image.

        Args:
            z: (B, latent_channels, H, W) latent tensor
        Returns:
            (B, out_channels, 8*H, 8*W) reconstructed image
        """
        # Unscale latent
        z = z / self.scaling_factor

        z = self.post_quant_conv(z)
        x = self.decoder(z)
        return x

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full forward pass with reconstruction and KL divergence.

        Args:
            x: (B, in_channels, H, W) input image
        Returns:
            reconstructed image, mean, logvar
        """
        mean, logvar = self.encoder(x)
        moments = self.quant_conv(torch.cat([mean, logvar], dim=1))
        mean, logvar = torch.chunk(moments, 2, dim=1)

        # Sample
        std = torch.exp(0.5 * logvar)
        z = mean + std * torch.randn_like(std)

        # Decode
        z_scaled = self.post_quant_conv(z)
        reconstructed = self.decoder(z_scaled)

        return reconstructed, mean, logvar

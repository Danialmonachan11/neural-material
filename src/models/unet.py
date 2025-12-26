"""
UNet architecture with cross-attention for conditional diffusion models.
Supports multi-modal conditioning and material/texture control.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
from einops import rearrange
from .transformer import SpatialTransformer


class TimestepEmbedding(nn.Module):
    """Sinusoidal timestep embedding with learned linear projections."""

    def __init__(self, dim: int, max_period: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period

        self.linear_1 = nn.Linear(dim, dim * 4)
        self.act = nn.SiLU()
        self.linear_2 = nn.Linear(dim * 4, dim * 4)

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            timesteps: (batch_size,) tensor of timesteps
        Returns:
            (batch_size, dim * 4) tensor of embeddings
        """
        half_dim = self.dim // 2
        freqs = torch.exp(
            -torch.log(torch.tensor(self.max_period)) *
            torch.arange(0, half_dim, dtype=torch.float32, device=timesteps.device) / half_dim
        )
        args = timesteps[:, None].float() * freqs[None, :]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

        embedding = self.linear_1(embedding)
        embedding = self.act(embedding)
        embedding = self.linear_2(embedding)
        return embedding


class ResnetBlock(nn.Module):
    """Residual block with timestep and conditioning injection."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        groups: int = 32,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.norm1 = nn.GroupNorm(groups, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.temb_proj = nn.Linear(temb_channels, out_channels)

        self.norm2 = nn.GroupNorm(groups, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        self.shortcut = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) input tensor
            temb: (B, temb_channels) timestep embedding
        Returns:
            (B, out_channels, H, W) output tensor
        """
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)

        # Add timestep embedding
        h = h + self.temb_proj(F.silu(temb))[:, :, None, None]

        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)

        return h + self.shortcut(x)


class DownBlock(nn.Module):
    """Downsampling block with residual connections and optional attention."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        num_layers: int = 2,
        add_downsample: bool = True,
        attention_head_dim: int = 8,
        num_attention_heads: int = 8,
        cross_attention_dim: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()

        resnets = []
        attentions = []

        for i in range(num_layers):
            in_ch = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlock(
                    in_channels=in_ch,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    dropout=dropout,
                )
            )

            if cross_attention_dim is not None:
                attentions.append(
                    SpatialTransformer(
                        in_channels=out_channels,
                        num_attention_heads=num_attention_heads,
                        attention_head_dim=attention_head_dim,
                        cross_attention_dim=cross_attention_dim,
                    )
                )
            else:
                attentions.append(None)

        self.resnets = nn.ModuleList(resnets)
        self.attentions = nn.ModuleList(attentions)

        if add_downsample:
            self.downsample = nn.Conv2d(
                out_channels, out_channels, kernel_size=3, stride=2, padding=1
            )
        else:
            self.downsample = None

    def forward(
        self,
        x: torch.Tensor,
        temb: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            x: (B, C, H, W) input tensor
            temb: (B, temb_channels) timestep embedding
            encoder_hidden_states: (B, seq_len, cross_attention_dim) conditioning
        Returns:
            output tensor and list of skip connection tensors
        """
        output_states = []

        for resnet, attn in zip(self.resnets, self.attentions):
            x = resnet(x, temb)
            if attn is not None:
                x = attn(x, encoder_hidden_states)
            output_states.append(x)

        if self.downsample is not None:
            x = self.downsample(x)
            output_states.append(x)

        return x, output_states


class UpBlock(nn.Module):
    """Upsampling block with residual connections and skip connections."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        prev_output_channels: int,
        temb_channels: int,
        num_layers: int = 3,
        add_upsample: bool = True,
        attention_head_dim: int = 8,
        num_attention_heads: int = 8,
        cross_attention_dim: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()

        resnets = []
        attentions = []

        for i in range(num_layers):
            res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
            resnet_in_channels = prev_output_channels if i == 0 else out_channels

            resnets.append(
                ResnetBlock(
                    in_channels=resnet_in_channels + res_skip_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    dropout=dropout,
                )
            )

            if cross_attention_dim is not None:
                attentions.append(
                    SpatialTransformer(
                        in_channels=out_channels,
                        num_attention_heads=num_attention_heads,
                        attention_head_dim=attention_head_dim,
                        cross_attention_dim=cross_attention_dim,
                    )
                )
            else:
                attentions.append(None)

        self.resnets = nn.ModuleList(resnets)
        self.attentions = nn.ModuleList(attentions)

        if add_upsample:
            self.upsample = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            )
        else:
            self.upsample = None

    def forward(
        self,
        x: torch.Tensor,
        res_hidden_states_tuple: Tuple[torch.Tensor, ...],
        temb: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) input tensor
            res_hidden_states_tuple: skip connection tensors from encoder
            temb: (B, temb_channels) timestep embedding
            encoder_hidden_states: (B, seq_len, cross_attention_dim) conditioning
        Returns:
            (B, out_channels, H, W) output tensor
        """
        for resnet, attn in zip(self.resnets, self.attentions):
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]

            x = torch.cat([x, res_hidden_states], dim=1)
            x = resnet(x, temb)

            if attn is not None:
                x = attn(x, encoder_hidden_states)

        if self.upsample is not None:
            x = self.upsample(x)

        return x


class UNet2DConditionModel(nn.Module):
    """
    UNet model with cross-attention for conditional generation.

    Supports multi-modal conditioning including text, material properties,
    and spatial control signals.
    """

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 4,
        block_out_channels: Tuple[int, ...] = (320, 640, 1280, 1280),
        layers_per_block: int = 2,
        attention_head_dim: int = 8,
        num_attention_heads: Optional[Tuple[int, ...]] = None,
        cross_attention_dim: int = 768,
        dropout: float = 0.0,
    ):
        super().__init__()

        if num_attention_heads is None:
            num_attention_heads = tuple(dim // attention_head_dim for dim in block_out_channels)

        # Timestep embedding
        time_embed_dim = block_out_channels[0] * 4
        self.time_embedding = TimestepEmbedding(block_out_channels[0], max_period=10000)

        # Input convolution
        self.conv_in = nn.Conv2d(in_channels, block_out_channels[0], kernel_size=3, padding=1)

        # Encoder (downsampling)
        down_blocks = []
        output_channel = block_out_channels[0]

        for i, block_out_channel in enumerate(block_out_channels):
            input_channel = output_channel
            output_channel = block_out_channel
            is_final_block = i == len(block_out_channels) - 1

            down_blocks.append(
                DownBlock(
                    in_channels=input_channel,
                    out_channels=output_channel,
                    temb_channels=time_embed_dim,
                    num_layers=layers_per_block,
                    add_downsample=not is_final_block,
                    attention_head_dim=attention_head_dim,
                    num_attention_heads=num_attention_heads[i],
                    cross_attention_dim=cross_attention_dim,
                    dropout=dropout,
                )
            )

        self.down_blocks = nn.ModuleList(down_blocks)

        # Middle block
        self.mid_block = nn.ModuleList([
            ResnetBlock(
                in_channels=block_out_channels[-1],
                out_channels=block_out_channels[-1],
                temb_channels=time_embed_dim,
                dropout=dropout,
            ),
            SpatialTransformer(
                in_channels=block_out_channels[-1],
                num_attention_heads=num_attention_heads[-1],
                attention_head_dim=attention_head_dim,
                cross_attention_dim=cross_attention_dim,
            ),
            ResnetBlock(
                in_channels=block_out_channels[-1],
                out_channels=block_out_channels[-1],
                temb_channels=time_embed_dim,
                dropout=dropout,
            ),
        ])

        # Decoder (upsampling)
        up_blocks = []
        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]

        for i, block_out_channel in enumerate(reversed_block_out_channels):
            prev_output_channel = output_channel
            output_channel = block_out_channel
            input_channel = reversed_block_out_channels[min(i + 1, len(block_out_channels) - 1)]
            is_final_block = i == len(block_out_channels) - 1

            up_blocks.append(
                UpBlock(
                    in_channels=input_channel,
                    out_channels=output_channel,
                    prev_output_channels=prev_output_channel,
                    temb_channels=time_embed_dim,
                    num_layers=layers_per_block + 1,
                    add_upsample=not is_final_block,
                    attention_head_dim=attention_head_dim,
                    num_attention_heads=num_attention_heads[len(block_out_channels) - 1 - i],
                    cross_attention_dim=cross_attention_dim,
                    dropout=dropout,
                )
            )

        self.up_blocks = nn.ModuleList(up_blocks)

        # Output convolution
        self.conv_norm_out = nn.GroupNorm(32, block_out_channels[0])
        self.conv_out = nn.Conv2d(block_out_channels[0], out_channels, kernel_size=3, padding=1)

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        return_dict: bool = True,
    ) -> torch.Tensor:
        """
        Args:
            sample: (B, in_channels, H, W) noisy latent tensor
            timestep: (B,) diffusion timestep
            encoder_hidden_states: (B, seq_len, cross_attention_dim) conditioning
            return_dict: whether to return dict (not used, for compatibility)
        Returns:
            (B, out_channels, H, W) predicted noise
        """
        # Timestep embedding
        temb = self.time_embedding(timestep)

        # Input convolution
        sample = self.conv_in(sample)

        # Encoder
        down_block_res_samples = (sample,)
        for down_block in self.down_blocks:
            sample, res_samples = down_block(sample, temb, encoder_hidden_states)
            down_block_res_samples += tuple(res_samples)

        # Middle
        sample = self.mid_block[0](sample, temb)
        sample = self.mid_block[1](sample, encoder_hidden_states)
        sample = self.mid_block[2](sample, temb)

        # Decoder
        for up_block in self.up_blocks:
            res_samples = down_block_res_samples[-len(up_block.resnets):]
            down_block_res_samples = down_block_res_samples[:-len(up_block.resnets)]
            sample = up_block(sample, res_samples, temb, encoder_hidden_states)

        # Output
        sample = self.conv_norm_out(sample)
        sample = F.silu(sample)
        sample = self.conv_out(sample)

        return sample

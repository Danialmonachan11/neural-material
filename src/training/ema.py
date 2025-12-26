"""
Exponential Moving Average (EMA) for model parameters.

EMA improves generation quality by maintaining a running average of model weights.
"""

import torch
import torch.nn as nn
from copy import deepcopy
from typing import Optional


class EMAModel:
    """
    Exponential Moving Average of model parameters.

    Maintains shadow parameters that are updated as:
    shadow = decay * shadow + (1 - decay) * param
    """

    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.9999,
        device: Optional[torch.device] = None,
    ):
        """
        Args:
            model: model to track
            decay: EMA decay rate (higher = more smoothing)
            device: device for shadow parameters
        """
        self.decay = decay
        self.device = device or next(model.parameters()).device

        # Create shadow parameters
        self.shadow_params = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow_params[name] = param.data.clone().to(self.device)

        self.num_updates = 0

    @torch.no_grad()
    def step(self, model: nn.Module):
        """
        Update EMA parameters.

        Args:
            model: current model
        """
        # Adjust decay based on number of updates
        decay = min(self.decay, (1 + self.num_updates) / (10 + self.num_updates))

        # Update shadow parameters
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow_params:
                self.shadow_params[name].mul_(decay).add_(
                    param.data.to(self.device), alpha=1 - decay
                )

        self.num_updates += 1

    @torch.no_grad()
    def copy_to(self, model: nn.Module):
        """
        Copy EMA parameters to model.

        Args:
            model: target model
        """
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow_params:
                param.data.copy_(self.shadow_params[name])

    def state_dict(self):
        """Get state dict for saving."""
        return {
            "decay": self.decay,
            "num_updates": self.num_updates,
            "shadow_params": self.shadow_params,
        }

    def load_state_dict(self, state_dict):
        """Load state dict."""
        self.decay = state_dict["decay"]
        self.num_updates = state_dict["num_updates"]
        self.shadow_params = state_dict["shadow_params"]

    def __repr__(self):
        return f"EMAModel(decay={self.decay}, num_updates={self.num_updates})"

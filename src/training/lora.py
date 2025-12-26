"""
LoRA (Low-Rank Adaptation) implementation for parameter-efficient fine-tuning.

LoRA injects trainable low-rank matrices into frozen pretrained weights,
dramatically reducing the number of trainable parameters while maintaining
or improving performance.

Paper: https://arxiv.org/abs/2106.09685
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Union
import math


class LoRALayer(nn.Module):
    """
    LoRA layer that wraps a linear layer with low-rank adaptation.

    The adaptation is: h = W_0 x + (B A) x * scaling
    where W_0 is frozen, A and B are trainable low-rank matrices.
    """

    def __init__(
        self,
        original_layer: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
        merge_weights: bool = False,
    ):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.merge_weights = merge_weights
        self.merged = False

        # Store original layer (frozen)
        self.original_layer = original_layer
        for param in self.original_layer.parameters():
            param.requires_grad = False

        # LoRA matrices
        in_features = original_layer.in_features
        out_features = original_layer.out_features

        # Initialize A with kaiming uniform, B with zeros
        # This ensures at initialization, the LoRA path outputs zeros
        self.lora_A = nn.Parameter(torch.zeros(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))

        # Initialize A
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with LoRA adaptation.

        Args:
            x: (B, ..., in_features) input tensor
        Returns:
            (B, ..., out_features) output tensor
        """
        # Original forward pass
        result = self.original_layer(x)

        if not self.merged:
            # Add LoRA adaptation
            lora_result = self.dropout(x) @ self.lora_A @ self.lora_B * self.scaling
            result = result + lora_result

        return result

    def merge(self):
        """Merge LoRA weights into the original layer for inference."""
        if not self.merged:
            # W = W_0 + scaling * B @ A
            delta_w = (self.lora_B @ self.lora_A.T) * self.scaling
            self.original_layer.weight.data += delta_w.T
            self.merged = True

    def unmerge(self):
        """Unmerge LoRA weights from the original layer."""
        if self.merged:
            delta_w = (self.lora_B @ self.lora_A.T) * self.scaling
            self.original_layer.weight.data -= delta_w.T
            self.merged = False

    def extra_repr(self) -> str:
        return f'rank={self.rank}, alpha={self.alpha}, scaling={self.scaling}'


def inject_lora_into_model(
    model: nn.Module,
    rank: int = 8,
    alpha: float = 16.0,
    target_modules: Optional[List[str]] = None,
    dropout: float = 0.0,
) -> nn.Module:
    """
    Inject LoRA layers into a model.

    Args:
        model: the model to inject LoRA into
        rank: rank of LoRA matrices
        alpha: LoRA alpha parameter for scaling
        target_modules: list of module names to target (e.g., ['to_q', 'to_k', 'to_v'])
                       If None, targets all linear layers in attention
        dropout: dropout probability for LoRA
    Returns:
        model with LoRA injected
    """
    if target_modules is None:
        # Default: target query, key, value projections in attention
        target_modules = ['to_q', 'to_k', 'to_v', 'to_out.0']

    def _inject_lora_recursive(module: nn.Module, name: str = ''):
        """Recursively inject LoRA into matching modules."""
        for child_name, child_module in module.named_children():
            full_name = f'{name}.{child_name}' if name else child_name

            # Check if this module should be replaced
            if any(target in full_name for target in target_modules):
                if isinstance(child_module, nn.Linear):
                    # Replace with LoRA layer
                    lora_layer = LoRALayer(
                        child_module,
                        rank=rank,
                        alpha=alpha,
                        dropout=dropout,
                    )
                    setattr(module, child_name, lora_layer)
                    print(f'Injected LoRA into: {full_name}')

            # Recurse into children
            _inject_lora_recursive(child_module, full_name)

    _inject_lora_recursive(model)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'\nTotal parameters: {total_params:,}')
    print(f'Trainable parameters: {trainable_params:,}')
    print(f'Trainable %: {100 * trainable_params / total_params:.2f}%')

    return model


def extract_lora_weights(model: nn.Module) -> Dict[str, torch.Tensor]:
    """
    Extract only LoRA weights from a model.

    Args:
        model: model with LoRA layers
    Returns:
        dictionary of LoRA weights
    """
    lora_state_dict = {}

    for name, module in model.named_modules():
        if isinstance(module, LoRALayer):
            lora_state_dict[f'{name}.lora_A'] = module.lora_A.data
            lora_state_dict[f'{name}.lora_B'] = module.lora_B.data

    return lora_state_dict


def merge_lora_weights(model: nn.Module, merge: bool = True) -> nn.Module:
    """
    Merge or unmerge all LoRA weights in a model.

    Args:
        model: model with LoRA layers
        merge: whether to merge (True) or unmerge (False)
    Returns:
        model with merged/unmerged weights
    """
    for module in model.modules():
        if isinstance(module, LoRALayer):
            if merge:
                module.merge()
            else:
                module.unmerge()

    return model


class LoRAConfig:
    """Configuration for LoRA fine-tuning."""

    def __init__(
        self,
        rank: int = 8,
        alpha: float = 16.0,
        target_modules: Optional[List[str]] = None,
        dropout: float = 0.0,
        bias: str = 'none',  # 'none', 'all', 'lora_only'
    ):
        self.rank = rank
        self.alpha = alpha
        self.target_modules = target_modules or ['to_q', 'to_k', 'to_v', 'to_out.0']
        self.dropout = dropout
        self.bias = bias

    def __repr__(self):
        return (
            f'LoRAConfig(rank={self.rank}, alpha={self.alpha}, '
            f'target_modules={self.target_modules}, dropout={self.dropout})'
        )


class LoRALinear(nn.Module):
    """
    Alternative LoRA implementation that directly subclasses Linear.
    More memory efficient for very large models.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # Original linear layer
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        # LoRA matrices
        if rank > 0:
            self.lora_A = nn.Parameter(torch.zeros(in_features, rank))
            self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

            self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        else:
            self.lora_A = None
            self.lora_B = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.linear(x)

        if self.lora_A is not None and self.training:
            lora_result = self.dropout(x) @ self.lora_A @ self.lora_B * self.scaling
            result = result + lora_result

        return result


def print_lora_summary(model: nn.Module):
    """Print a summary of LoRA layers in the model."""
    lora_layers = []

    for name, module in model.named_modules():
        if isinstance(module, (LoRALayer, LoRALinear)):
            lora_layers.append({
                'name': name,
                'rank': module.rank,
                'alpha': module.alpha,
                'params': sum(p.numel() for p in [module.lora_A, module.lora_B]),
            })

    if lora_layers:
        print('\n' + '=' * 80)
        print('LoRA Summary')
        print('=' * 80)
        print(f'{"Layer Name":<50} {"Rank":<8} {"Alpha":<8} {"Params":<12}')
        print('-' * 80)

        total_lora_params = 0
        for layer in lora_layers:
            print(f'{layer["name"]:<50} {layer["rank"]:<8} {layer["alpha"]:<8.1f} {layer["params"]:<12,}')
            total_lora_params += layer['params']

        print('-' * 80)
        print(f'Total LoRA layers: {len(lora_layers)}')
        print(f'Total LoRA parameters: {total_lora_params:,}')
        print('=' * 80 + '\n')
    else:
        print('No LoRA layers found in model.')


# Utility for saving and loading LoRA weights
def save_lora_weights(model: nn.Module, path: str):
    """Save only LoRA weights to disk."""
    lora_weights = extract_lora_weights(model)
    torch.save(lora_weights, path)
    print(f'Saved LoRA weights to {path}')


def load_lora_weights(model: nn.Module, path: str, strict: bool = True):
    """Load LoRA weights from disk."""
    lora_weights = torch.load(path)

    # Load into model
    missing, unexpected = [], []
    for name, param in lora_weights.items():
        try:
            # Navigate to the parameter
            parts = name.split('.')
            module = model
            for part in parts[:-1]:
                module = getattr(module, part)

            # Set the parameter
            param_name = parts[-1]
            if hasattr(module, param_name):
                getattr(module, param_name).data.copy_(param)
            else:
                unexpected.append(name)
        except AttributeError:
            missing.append(name)

    if strict and (missing or unexpected):
        raise RuntimeError(
            f'Error loading LoRA weights:\n'
            f'Missing keys: {missing}\n'
            f'Unexpected keys: {unexpected}'
        )

    print(f'Loaded LoRA weights from {path}')
    return model

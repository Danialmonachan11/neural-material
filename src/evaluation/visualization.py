"""
Visualization utilities for diffusion model outputs.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional, Dict
from pathlib import Path
from PIL import Image
import torchvision.transforms as T


def tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert tensor to numpy image.

    Args:
        tensor: (C, H, W) or (H, W, C) tensor in [-1, 1] or [0, 1]
    Returns:
        (H, W, C) numpy array in [0, 255]
    """
    if tensor.dim() == 3 and tensor.shape[0] in [1, 3, 4]:
        tensor = tensor.permute(1, 2, 0)

    # Normalize to [0, 1]
    if tensor.min() < 0:
        tensor = (tensor + 1) / 2

    tensor = torch.clamp(tensor, 0, 1)

    # Convert to numpy and scale to [0, 255]
    image = (tensor.cpu().numpy() * 255).astype(np.uint8)

    return image


def visualize_generation(
    generated_images: torch.Tensor,
    prompts: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    num_images: int = 16,
    nrow: int = 4,
    figsize: tuple = (12, 12),
):
    """
    Visualize generated images in a grid.

    Args:
        generated_images: (N, C, H, W) generated images
        prompts: optional list of prompts
        save_path: path to save figure
        num_images: number of images to display
        nrow: number of images per row
        figsize: figure size
    """
    num_images = min(num_images, generated_images.shape[0])
    ncol = nrow
    nrow = (num_images + ncol - 1) // ncol

    fig, axes = plt.subplots(nrow, ncol, figsize=figsize)
    if nrow == 1:
        axes = axes.reshape(1, -1)

    for idx in range(num_images):
        row = idx // ncol
        col = idx % ncol

        image = tensor_to_image(generated_images[idx])
        axes[row, col].imshow(image)
        axes[row, col].axis('off')

        if prompts is not None and idx < len(prompts):
            # Truncate long prompts
            prompt = prompts[idx]
            if len(prompt) > 40:
                prompt = prompt[:37] + "..."
            axes[row, col].set_title(prompt, fontsize=8)

    # Hide unused subplots
    for idx in range(num_images, nrow * ncol):
        row = idx // ncol
        col = idx % ncol
        axes[row, col].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")

    plt.show()


def create_comparison_grid(
    real_images: torch.Tensor,
    generated_images: torch.Tensor,
    prompts: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    num_pairs: int = 8,
    figsize: tuple = (16, 8),
):
    """
    Create side-by-side comparison of real and generated images.

    Args:
        real_images: (N, C, H, W) real images
        generated_images: (N, C, H, W) generated images
        prompts: optional prompts
        save_path: save path
        num_pairs: number of image pairs to show
        figsize: figure size
    """
    num_pairs = min(num_pairs, real_images.shape[0], generated_images.shape[0])

    fig, axes = plt.subplots(num_pairs, 2, figsize=figsize)
    if num_pairs == 1:
        axes = axes.reshape(1, -1)

    for idx in range(num_pairs):
        # Real image
        real_img = tensor_to_image(real_images[idx])
        axes[idx, 0].imshow(real_img)
        axes[idx, 0].axis('off')
        if idx == 0:
            axes[idx, 0].set_title("Real", fontsize=12, fontweight='bold')

        # Generated image
        gen_img = tensor_to_image(generated_images[idx])
        axes[idx, 1].imshow(gen_img)
        axes[idx, 1].axis('off')
        if idx == 0:
            axes[idx, 1].set_title("Generated", fontsize=12, fontweight='bold')

        # Add prompt as ylabel
        if prompts is not None and idx < len(prompts):
            prompt = prompts[idx]
            if len(prompt) > 30:
                prompt = prompt[:27] + "..."
            axes[idx, 0].set_ylabel(prompt, fontsize=9, rotation=0, ha='right', va='center')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved comparison to {save_path}")

    plt.show()


def plot_training_curves(
    metrics_history: Dict[str, List[float]],
    save_path: Optional[str] = None,
    figsize: tuple = (14, 6),
):
    """
    Plot training curves.

    Args:
        metrics_history: dict of metric names to lists of values
        save_path: save path
        figsize: figure size
    """
    num_metrics = len(metrics_history)
    ncols = min(3, num_metrics)
    nrows = (num_metrics + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    if num_metrics == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, (metric_name, values) in enumerate(metrics_history.items()):
        axes[idx].plot(values, linewidth=2)
        axes[idx].set_title(metric_name.replace('_', ' ').title(), fontsize=12)
        axes[idx].set_xlabel('Step', fontsize=10)
        axes[idx].set_ylabel('Value', fontsize=10)
        axes[idx].grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(num_metrics, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved training curves to {save_path}")

    plt.show()


def visualize_attention_maps(
    image: torch.Tensor,
    attention_maps: torch.Tensor,
    save_path: Optional[str] = None,
    num_heads: int = 8,
    figsize: tuple = (16, 8),
):
    """
    Visualize attention maps overlaid on image.

    Args:
        image: (C, H, W) image
        attention_maps: (num_heads, H, W) attention weights
        save_path: save path
        num_heads: number of attention heads to visualize
        figsize: figure size
    """
    image_np = tensor_to_image(image)
    num_heads = min(num_heads, attention_maps.shape[0])

    fig, axes = plt.subplots(2, num_heads // 2, figsize=figsize)
    axes = axes.flatten()

    for idx in range(num_heads):
        attn_map = attention_maps[idx].cpu().numpy()

        # Resize attention map to image size
        from scipy.ndimage import zoom
        zoom_factor = image_np.shape[0] / attn_map.shape[0]
        attn_map_resized = zoom(attn_map, zoom_factor, order=1)

        # Overlay attention on image
        axes[idx].imshow(image_np)
        axes[idx].imshow(attn_map_resized, alpha=0.5, cmap='jet')
        axes[idx].set_title(f'Head {idx + 1}', fontsize=10)
        axes[idx].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved attention visualization to {save_path}")

    plt.show()


def visualize_region_control(
    generated_image: torch.Tensor,
    region_masks: List[torch.Tensor],
    region_prompts: List[str],
    save_path: Optional[str] = None,
    figsize: tuple = (16, 4),
):
    """
    Visualize region-based control.

    Args:
        generated_image: (C, H, W) generated image
        region_masks: list of (H, W) region masks
        region_prompts: list of region prompts
        save_path: save path
        figsize: figure size
    """
    num_regions = len(region_masks)

    fig, axes = plt.subplots(1, num_regions + 1, figsize=figsize)

    # Show generated image
    image_np = tensor_to_image(generated_image)
    axes[0].imshow(image_np)
    axes[0].set_title('Generated Image', fontsize=12, fontweight='bold')
    axes[0].axis('off')

    # Show each region
    for idx, (mask, prompt) in enumerate(zip(region_masks, region_prompts)):
        mask_np = mask.cpu().numpy()

        # Apply mask to image
        masked_image = image_np.copy()
        mask_3d = np.stack([mask_np] * 3, axis=-1)
        masked_image = (masked_image * mask_3d).astype(np.uint8)

        axes[idx + 1].imshow(masked_image)
        axes[idx + 1].set_title(f'Region {idx + 1}\n{prompt[:20]}...', fontsize=9)
        axes[idx + 1].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved region control visualization to {save_path}")

    plt.show()


def create_progressive_generation_gif(
    intermediate_images: List[torch.Tensor],
    save_path: str,
    duration: int = 100,
):
    """
    Create GIF showing progressive generation.

    Args:
        intermediate_images: list of (C, H, W) images at different timesteps
        save_path: path to save GIF
        duration: duration per frame in ms
    """
    pil_images = []

    for img_tensor in intermediate_images:
        img_np = tensor_to_image(img_tensor)
        pil_img = Image.fromarray(img_np)
        pil_images.append(pil_img)

    # Save as GIF
    pil_images[0].save(
        save_path,
        save_all=True,
        append_images=pil_images[1:],
        duration=duration,
        loop=0,
    )

    print(f"Saved progressive generation GIF to {save_path}")


def plot_material_distribution(
    material_predictions: np.ndarray,
    material_labels: List[str],
    save_path: Optional[str] = None,
    figsize: tuple = (10, 6),
):
    """
    Plot distribution of predicted materials.

    Args:
        material_predictions: (N,) array of material class predictions
        material_labels: list of material class names
        save_path: save path
        figsize: figure size
    """
    # Count occurrences
    unique, counts = np.unique(material_predictions, return_counts=True)

    # Create bar plot
    fig, ax = plt.subplots(figsize=figsize)

    colors = sns.color_palette("husl", len(unique))
    ax.bar(unique, counts, color=colors)

    ax.set_xlabel('Material Class', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Material Distribution in Generated Images', fontsize=14, fontweight='bold')

    # Set x-tick labels
    ax.set_xticks(unique)
    ax.set_xticklabels([material_labels[i] for i in unique], rotation=45, ha='right')

    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved material distribution to {save_path}")

    plt.show()

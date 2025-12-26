# ControlDiff Architecture

This document provides a detailed overview of the ControlDiff architecture and implementation.

## System Overview

ControlDiff is a modular diffusion model framework designed for material-aware, controllable image generation with advanced fine-tuning capabilities.

```
┌─────────────────────────────────────────────────────────┐
│                    ControlDiff System                    │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌────────────┐  ┌──────────────┐  ┌─────────────────┐ │
│  │    VAE     │  │     UNet     │  │   Scheduler     │ │
│  │  Encoder/  │  │  Denoising   │  │   (DDPM/DDIM)   │ │
│  │  Decoder   │  │   Network    │  │                 │ │
│  └────────────┘  └──────────────┘  └─────────────────┘ │
│                                                          │
│  ┌─────────────────────────────────────────────────────┐ │
│  │           Conditioning Mechanisms                   │ │
│  ├─────────────┬─────────────┬─────────────────────────┤ │
│  │   Text      │  Material   │  Region Control         │ │
│  │ (CLIP/T5)   │  Encoder    │  (Spatial Masks)        │ │
│  └─────────────┴─────────────┴─────────────────────────┘ │
│                                                          │
│  ┌─────────────────────────────────────────────────────┐ │
│  │          Fine-tuning Methods                        │ │
│  ├─────────────┬─────────────┬─────────────────────────┤ │
│  │    LoRA     │ DreamBooth  │  Transfer Learning      │ │
│  └─────────────┴─────────────┴─────────────────────────┘ │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

## Core Components

### 1. UNet Architecture

The UNet is a U-shaped neural network with encoder-decoder structure:

**Key Features:**
- **Encoder (Downsampling):** Progressive feature extraction at multiple scales
- **Middle Block:** Bottleneck with self-attention for global reasoning
- **Decoder (Upsampling):** Progressive reconstruction with skip connections
- **Cross-Attention:** Multi-modal conditioning at each resolution

**Architecture Details:**
```
Input: (B, 4, 64, 64) latents

Encoder:
├─ Block 1: (4, 64, 64) → (320, 64, 64)
├─ Block 2: (320, 64, 64) → (640, 32, 32)
├─ Block 3: (640, 32, 32) → (1280, 16, 16)
└─ Block 4: (1280, 16, 16) → (1280, 8, 8)

Middle:
└─ ResNet + Attention + ResNet: (1280, 8, 8)

Decoder:
├─ Block 4: (1280, 8, 8) → (1280, 16, 16)
├─ Block 3: (1280, 16, 16) → (640, 32, 32)
├─ Block 2: (640, 32, 32) → (320, 64, 64)
└─ Block 1: (320, 64, 64) → (4, 64, 64)

Output: (B, 4, 64, 64) predicted noise
```

### 2. Transformer Blocks

Spatial transformer for multi-modal fusion:

```python
class TransformerBlock:
    1. Self-Attention: Spatial reasoning
    2. Cross-Attention: Condition fusion (text, material, etc.)
    3. Feed-Forward: Non-linear transformation
```

**Multi-Head Attention:**
- Query: from image features
- Key/Value: from conditioning (text, material, region)
- Heads: 8 (each head learns different relationships)
- Dimension per head: 64

### 3. VAE (Variational Autoencoder)

Compresses images to latent space:

```
Image Space:        Latent Space:
(3, 512, 512)  →   (4, 64, 64)
   8x spatial compression
```

**Purpose:**
- Reduces computational cost by 64x
- Enables training on 512x512 images with 64x64 latents
- Semantic compression (preserves important features)

**Training:**
- Reconstruction loss: MSE between input and decoded output
- KL divergence: Regularizes latent distribution
- Perceptual loss: Ensures semantic similarity

### 4. Diffusion Process

Forward diffusion (training):
```
x_0 → x_1 → x_2 → ... → x_T (noise)
    ↑ Gradually add Gaussian noise
```

Reverse diffusion (inference):
```
x_T → x_{T-1} → ... → x_1 → x_0 (image)
    ↑ Predicted by UNet, guided by conditions
```

**Noise Schedule:**
- Linear: β_t increases linearly from 0.0001 to 0.02
- Cosine: Smoother schedule, better for high-resolution
- Scaled Linear: Used by Stable Diffusion

## Fine-Tuning Methods

### LoRA (Low-Rank Adaptation)

Injects trainable low-rank matrices into attention layers:

```
Original: h = W_0 · x
LoRA:     h = W_0 · x + (B · A) · x

where:
  W_0: frozen pretrained weights
  A, B: trainable low-rank matrices (rank r << d)

Parameters:
  Original: d × d
  LoRA:     d × r + r × d ≈ 2dr (r=8: ~1-2% of original)
```

**Benefits:**
- 100x fewer parameters than full fine-tuning
- Fast training (minutes vs hours)
- Multiple LoRA adapters can be merged
- Minimal memory overhead

### DreamBooth

Subject-specific personalization with prior preservation:

```
Loss = L_instance + λ · L_prior

L_instance: Loss on subject images
L_prior:    Loss on class images (prevents overfitting)
```

**Process:**
1. Train on 3-5 images of subject with unique identifier (e.g., "sks")
2. Generate class images for regularization
3. Fine-tune with combined loss
4. Result: Model learns subject while preserving diversity

## Conditioning Mechanisms

### 1. Material Conditioning

**Input Modalities:**
- BRDF Parameters: (roughness, metallic, specular, color, ...)
- Texture Images: Reference textures encoded by CNN
- Category Labels: Learned embeddings for material types

**Encoding:**
```
Material Properties → Material Encoder → Embedding (768-dim)
                                              ↓
                                    Cross-Attention in UNet
```

### 2. Region Control

Spatial control through region masks:

```python
Region 1: Left half  → "metallic surface"
Region 2: Right half → "wooden texture"

Combined via spatial attention modulation
```

**Implementation:**
- Each region has: mask, prompt, strength
- Masks blurred for smooth transitions
- Multi-region fusion with conflict resolution

### 3. Cross-Attention Fusion

Hierarchical fusion of multiple conditions:

```
Level 1: Text + Material → Semantic Features
Level 2: Semantic + Spatial → Final Conditioning
Level 3: Refinement → Output Embedding
```

## Training Pipeline

### Distributed Training

**Data Parallel (DDP):**
```
GPU 0: Batch 0 → Gradients
GPU 1: Batch 1 → Gradients
...
GPU N: Batch N → Gradients
         ↓
    Synchronize & Average
         ↓
    Update Shared Model
```

**Features:**
- Automatic gradient synchronization
- Efficient communication (NCCL backend)
- Gradient accumulation for large effective batch sizes

### Mixed Precision Training

```python
with autocast():  # FP16 forward pass
    loss = model(input)

scaler.scale(loss).backward()  # FP16 backward
scaler.step(optimizer)         # FP32 update
```

**Benefits:**
- 2x faster training
- 2x less memory
- Minimal accuracy loss

### EMA (Exponential Moving Average)

```python
θ_ema = decay · θ_ema + (1 - decay) · θ_train

decay = 0.9999  # Very slow update
```

**Purpose:**
- Smooths training dynamics
- Better generation quality
- Reduces artifacts

## Evaluation Metrics

### FID (Frechet Inception Distance)

Measures distribution similarity:
```
FID = ||μ_real - μ_gen||² + Tr(Σ_real + Σ_gen - 2√(Σ_real·Σ_gen))

Lower is better (< 10 is excellent)
```

### CLIP Score

Text-image alignment:
```
CLIP Score = similarity(CLIP_image, CLIP_text)

Higher is better (> 0.3 is good)
```

### Inception Score

Quality and diversity:
```
IS = exp(E[KL(p(y|x) || p(y))])

Higher is better (> 5 is good)
```

## Performance Characteristics

### Computational Requirements

**Training (512x512 images):**
- Base model: 8x A100 GPUs, ~48 hours
- LoRA fine-tune: 1x A100 GPU, ~4 hours
- DreamBooth: 1x A100 GPU, ~2 hours

**Inference:**
- Single image: ~2 seconds (50 steps, A100)
- Batch of 4: ~5 seconds (50 steps, A100)

### Memory Usage

```
Component          Memory (GB)
─────────────────────────────
UNet              ~3.5
VAE               ~0.5
Text Encoder      ~0.5
Activations       ~8.0 (BS=4)
Optimizer States  ~7.0
─────────────────────────────
Total             ~20 GB
```

**Optimizations:**
- Gradient checkpointing: -50% memory, +30% time
- Mixed precision: -50% memory
- LoRA: -98% trainable parameters

## Design Decisions

### Why Latent Diffusion?

- **Efficiency:** 64x less computation than pixel space
- **Quality:** Semantic compression preserves important features
- **Scalability:** Enables high-resolution generation

### Why Cross-Attention for Conditioning?

- **Flexibility:** Handles variable-length inputs
- **Effectiveness:** Strong empirical results
- **Interpretability:** Attention maps show what model focuses on

### Why LoRA over Full Fine-tuning?

- **Speed:** 100x faster training
- **Memory:** Fits on consumer GPUs
- **Modularity:** Multiple adapters can coexist
- **Preservation:** Minimal degradation of base capabilities

## Extension Points

The architecture is designed for easy extension:

1. **New Conditioning Modalities:** Add encoders in `src/conditioning/`
2. **Custom Schedulers:** Implement in `src/models/diffusion.py`
3. **Novel Architectures:** Replace UNet while maintaining interfaces
4. **Evaluation Metrics:** Add to `src/evaluation/metrics.py`

## References

- Ho et al. (2020): Denoising Diffusion Probabilistic Models
- Rombach et al. (2022): High-Resolution Image Synthesis with Latent Diffusion Models
- Hu et al. (2021): LoRA: Low-Rank Adaptation of Large Language Models
- Ruiz et al. (2022): DreamBooth: Fine Tuning Text-to-Image Diffusion Models

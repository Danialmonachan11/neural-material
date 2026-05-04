# ControlDiff: Advanced Material-Conditioned Diffusion Model *https://danialmonachan11.github.io/demo-llm.html*

A comprehensive diffusion model framework implementing modern fine-tuning techniques (LoRA, DreamBooth) and controllable generation for material-aware image synthesis.

## Key Features

### 🎨 Advanced Architecture
- **Hybrid UNet-Transformer** architecture for high-quality generation
- **Latent Diffusion** for efficient training and inference
- **Multi-scale feature extraction** with cross-attention mechanisms

### 🔧 Fine-tuning Methods
- **LoRA (Low-Rank Adaptation)**: Efficient parameter-efficient fine-tuning
- **DreamBooth**: Subject-specific personalization
- **Domain Transfer Learning**: Quick adaptation to new domains

### 🎯 Controllable Generation
- **Region-based Control**: Spatial conditioning for precise layout control
- **Material Conditioning**: Texture and material-aware synthesis
- **Multi-modal Conditioning**: Text, image, and structure guidance

### 🚀 Production-Ready Pipeline
- **Distributed Training**: Multi-GPU and cluster support
- **Experiment Tracking**: WandB/TensorBoard integration
- **Automated Evaluation**: Comprehensive metrics (FID, CLIP score, material accuracy)
- **Modular Design**: Easy to extend and customize

## Project Structure

```
controldiff/
├── src/
│   ├── models/
│   │   ├── unet.py              # UNet architecture with attention
│   │   ├── transformer.py       # Transformer components
│   │   ├── vae.py              # Variational autoencoder
│   │   └── diffusion.py        # Diffusion process
│   ├── training/
│   │   ├── lora.py             # LoRA implementation
│   │   ├── dreambooth.py       # DreamBooth training
│   │   ├── trainer.py          # Main training loop
│   │   └── distributed.py      # Multi-GPU training
│   ├── conditioning/
│   │   ├── region_control.py   # Spatial conditioning
│   │   ├── material_encoder.py # Material feature extraction
│   │   └── cross_attention.py  # Multi-modal fusion
│   ├── data/
│   │   ├── dataset.py          # Dataset implementations
│   │   └── augmentation.py     # Data augmentation
│   └── evaluation/
│       ├── metrics.py          # Evaluation metrics
│       └── visualization.py    # Result visualization
├── configs/
│   ├── base_config.yaml        # Base configuration
│   ├── lora_finetune.yaml      # LoRA fine-tuning config
│   └── dreambooth.yaml         # DreamBooth config
├── scripts/
│   ├── train.py                # Training script
│   ├── finetune_lora.py        # LoRA fine-tuning
│   ├── finetune_dreambooth.py  # DreamBooth fine-tuning
│   └── inference.py            # Inference script
├── notebooks/
│   ├── 01_quickstart.ipynb
│   ├── 02_lora_finetuning.ipynb
│   └── 03_controllable_generation.ipynb
├── tests/
├── requirements.txt
└── setup.py
```

## Installation

```bash
# Clone repository
git clone https://github.com/Danialmonachan11/neural-material.git
cd neural-material

# Install dependencies
pip install -e .

# For development
pip install -e ".[dev]"
```

## Quick Start

### Training from Scratch
```bash
python scripts/train.py --config configs/base_config.yaml
```

### LoRA Fine-tuning
```bash
python scripts/finetune_lora.py \
    --base_model checkpoints/base_model.pt \
    --data_dir data/custom_dataset \
    --lora_rank 8 \
    --output_dir outputs/lora_finetune
```

### DreamBooth Personalization
```bash
python scripts/finetune_dreambooth.py \
    --instance_images data/dreambooth/subject \
    --class_images data/dreambooth/class \
    --instance_prompt "a photo of sks material"
```

## Research Areas Covered

### 1. Generative Model Development
- Latent diffusion with VAE compression
- Noise scheduling strategies (cosine, linear, custom)
- UNet with multi-head self-attention and cross-attention
- Transformer-based denoising blocks

### 2. Fine-tuning & Adaptation
- **LoRA**: Inject low-rank matrices into attention layers
- **DreamBooth**: Prior preservation + subject-specific fine-tuning
- **Textual Inversion**: Learn new embeddings for concepts
- **Custom Schedulers**: Adaptive learning rate and sampling strategies

### 3. Controllable Generation
- **Spatial Control**: Region masks, layout conditioning
- **Material Encoding**: BRDF parameters, texture embeddings
- **Structure Guidance**: Edge maps, depth maps, normal maps
- **Multi-condition Fusion**: Cross-attention based conditioning

### 4. Scalable Infrastructure
- PyTorch DDP for multi-GPU training
- Mixed precision training (FP16/BF16)
- Gradient accumulation and checkpointing
- Experiment management with Hydra + WandB
- Automated hyperparameter tuning

## Evaluation Metrics

The framework includes comprehensive evaluation tools:

- **FID (Fréchet Inception Distance)**: Measures image quality and diversity
- **CLIP Score**: Evaluates text-image alignment
- **Inception Score**: Assesses generation quality
- **Material Accuracy**: Custom metric for material classification

See `src/evaluation/metrics.py` for implementation details.

## Acknowledgments

This project implements techniques from several research papers:

- **Denoising Diffusion Probabilistic Models** (Ho et al., 2020)
- **High-Resolution Image Synthesis with Latent Diffusion Models** (Rombach et al., 2022)
- **LoRA: Low-Rank Adaptation of Large Language Models** (Hu et al., 2021)
- **DreamBooth: Fine Tuning Text-to-Image Diffusion Models** (Ruiz et al., 2022)

## License

MIT License

# Getting Started with ControlDiff

This guide will help you get started with ControlDiff for advanced diffusion model development and fine-tuning.

## Installation

### 1. Clone and Install

```bash
cd controldiff
pip install -e .
```

### 2. Install Development Dependencies

```bash
pip install -e ".[dev]"
```

### 3. Verify Installation

```bash
python -c "import sys; sys.path.append('src'); from models import UNet2DConditionModel; print('Installation successful!')"
```

## Quick Start

### Basic Image Generation

```python
import torch
from models import UNet2DConditionModel, AutoencoderKL, DDIMScheduler
from conditioning import MaterialEncoder, compute_brdf_params

# Initialize models
device = 'cuda'
unet = UNet2DConditionModel().to(device)
vae = AutoencoderKL().to(device)
scheduler = DDIMScheduler()

# Define material properties
brdf_params = compute_brdf_params(
    roughness=0.3,
    metallic=0.9,
    base_color=(0.8, 0.8, 0.8),
)

# Encode material
material_encoder = MaterialEncoder().to(device)
material_embedding = material_encoder.encode_parameters(brdf_params.to(device))

# Generate image
scheduler.set_timesteps(50)
latents = torch.randn(1, 4, 64, 64, device=device)

for t in scheduler.timesteps:
    noise_pred = unet(latents, t, material_embedding.unsqueeze(1))
    latents = scheduler.step(noise_pred, t, latents)

image = vae.decode(latents)
```

## Training from Scratch

### 1. Prepare Your Dataset

Organize your data as:
```
data/
├── train/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── metadata.json  # Contains prompts/materials
└── val/
    ├── image1.jpg
    └── metadata.json
```

### 2. Configure Training

Edit `configs/base_config.yaml`:

```yaml
training:
  batch_size: 4
  num_epochs: 100
  learning_rate: 1.0e-4

data:
  train_data_dir: 'data/train'
  val_data_dir: 'data/val'
  image_size: 512
```

### 3. Start Training

```bash
python scripts/train.py --config configs/base_config.yaml
```

### 4. Monitor Training

The training script automatically logs to WandB. View your runs at:
```
https://wandb.ai/your-username/controldiff
```

## LoRA Fine-tuning

### Quick LoRA Fine-tuning

```python
from models import UNet2DConditionModel
from training.lora import inject_lora_into_model, save_lora_weights

# Load pretrained model
unet = UNet2DConditionModel()
unet.load_state_dict(torch.load('checkpoints/base_model.pt'))

# Inject LoRA
unet = inject_lora_into_model(
    unet,
    rank=8,        # Lower rank = fewer parameters
    alpha=16.0,    # Scaling factor
    target_modules=['to_q', 'to_k', 'to_v'],
)

# Train only LoRA parameters (1-2% of total parameters!)
lora_params = [p for p in unet.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(lora_params, lr=5e-5)

# ... training loop ...

# Save only LoRA weights (very small file!)
save_lora_weights(unet, 'lora_weights.pt')
```

### Using Script

```bash
python scripts/finetune_lora.py \
    --config configs/lora_finetune.yaml \
    --base_model checkpoints/base_model.pt \
    --data_dir data/custom_dataset
```

## DreamBooth Personalization

### 1. Prepare Instance Images

Place 3-5 images of your subject in `data/dreambooth/instance/`:
```
data/dreambooth/instance/
├── subject_001.jpg
├── subject_002.jpg
└── subject_003.jpg
```

### 2. Generate Class Images (Optional)

```python
from training.dreambooth import generate_class_images

generate_class_images(
    model=unet,
    vae=vae,
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    class_prompt="a photo of material",
    num_images=200,
    output_dir="data/dreambooth/class",
)
```

### 3. Fine-tune

```bash
python scripts/finetune_dreambooth.py \
    --config configs/dreambooth.yaml \
    --instance_prompt "a photo of sks material" \
    --class_prompt "a photo of material"
```

## Controllable Generation

### Region-Based Control

```python
from conditioning import RegionMask, RegionController, create_rectangular_mask

# Define regions
region1 = RegionMask(
    mask=create_rectangular_mask(512, 512, bbox=(0.0, 0.0, 0.5, 1.0)),
    prompt="metallic surface",
    strength=1.0,
)

region2 = RegionMask(
    mask=create_rectangular_mask(512, 512, bbox=(0.5, 0.0, 1.0, 1.0)),
    prompt="wooden texture",
    strength=1.0,
)

# Apply region control during generation
region_controller = RegionController()
# ... integrate into generation loop ...
```

### Material Conditioning

```python
from conditioning import MaterialEncoder

# From texture image
material_encoder = MaterialEncoder(use_texture_encoder=True)
texture_image = load_image("texture.jpg")
material_embedding = material_encoder.encode_texture(texture_image)

# From BRDF parameters
brdf_params = compute_brdf_params(
    roughness=0.4,
    metallic=0.8,
    base_color=(0.7, 0.7, 0.7),
)
material_embedding = material_encoder.encode_parameters(brdf_params)

# From category
category = torch.tensor([0])  # 0 = metal
material_embedding = material_encoder.encode_category(category)
```

## Evaluation

### Compute Metrics

```python
from evaluation import compute_all_metrics

metrics = compute_all_metrics(
    real_images=real_images,
    generated_images=generated_images,
    prompts=prompts,
)

print(f"FID: {metrics['fid']:.2f}")
print(f"CLIP Score: {metrics['clip_score']:.3f}")
print(f"Inception Score: {metrics['inception_score_mean']:.2f}")
```

### Visualization

```python
from evaluation import visualize_generation, create_comparison_grid

# Visualize generated images
visualize_generation(
    generated_images,
    prompts=prompts,
    save_path='outputs/generated.png',
)

# Side-by-side comparison
create_comparison_grid(
    real_images,
    generated_images,
    prompts=prompts,
    save_path='outputs/comparison.png',
)
```

## Distributed Training

### Single-Node Multi-GPU

```bash
torchrun --nproc_per_node=8 scripts/train.py --config configs/base_config.yaml
```

### Multi-Node Training

```bash
# Node 0
torchrun \
    --nproc_per_node=8 \
    --nnodes=4 \
    --node_rank=0 \
    --master_addr=192.168.1.1 \
    --master_port=29500 \
    scripts/train.py --config configs/base_config.yaml

# Node 1, 2, 3 (change node_rank)
torchrun \
    --nproc_per_node=8 \
    --nnodes=4 \
    --node_rank=1 \
    --master_addr=192.168.1.1 \
    --master_port=29500 \
    scripts/train.py --config configs/base_config.yaml
```

## Best Practices

### Memory Optimization

1. **Use Gradient Checkpointing**
```python
unet.enable_gradient_checkpointing()
```

2. **Mixed Precision Training**
```yaml
training:
  mixed_precision: 'fp16'  # or 'bf16'
```

3. **Gradient Accumulation**
```yaml
training:
  gradient_accumulation_steps: 4
```

### Improving Quality

1. **Use EMA for Better Generations**
```yaml
training:
  use_ema: true
  ema_decay: 0.9999
```

2. **Longer Training**
- Base model: 100-500K steps
- Fine-tuning: 5-50K steps
- DreamBooth: 500-2000 steps

3. **Optimal Hyperparameters**
- Learning rate: 1e-4 (base), 5e-5 (fine-tune), 2e-6 (DreamBooth)
- Batch size: As large as GPU allows
- Gradient clipping: 1.0

## Troubleshooting

### Out of Memory

- Reduce batch size
- Enable gradient checkpointing
- Use gradient accumulation
- Reduce image resolution

### Poor Generation Quality

- Train longer
- Use EMA
- Increase model capacity
- Check data quality

### Slow Training

- Use mixed precision
- Enable torch.compile (PyTorch 2.0+)
- Use faster data loading (more workers)
- Distributed training

## Next Steps

1. Explore the [example notebooks](notebooks/)
2. Read the [architecture documentation](docs/architecture.md)
3. Check out [advanced techniques](docs/advanced.md)
4. Join our community discussions

## Resources

- Paper: [Understanding Diffusion Models](https://arxiv.org/abs/2006.11239)
- LoRA: [Low-Rank Adaptation](https://arxiv.org/abs/2106.09685)
- DreamBooth: [Fine-Tuning Text-to-Image Diffusion](https://arxiv.org/abs/2208.12242)

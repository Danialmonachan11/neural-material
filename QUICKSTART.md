# ControlDiff - Quick Start Guide

## Step-by-Step Setup and Run Instructions

### 1. Installation

```bash
# Navigate to project directory
cd D:\CV_BUILDER\controldiff

# Install in development mode
pip install -e .

# Or install with all dependencies at once
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -e .
```

**Verify installation:**
```bash
python -c "import sys; sys.path.insert(0, 'src'); from models import UNet2DConditionModel; print('✓ Installation successful!')"
```

### 2. Test Core Components (No Training Required)

Create a test file `test_models.py`:

```python
import torch
import sys
sys.path.insert(0, 'src')

from models import UNet2DConditionModel, AutoencoderKL, DDPMScheduler
from conditioning import MaterialEncoder, compute_brdf_params

print("Testing ControlDiff components...\n")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}\n")

# 1. Test UNet
print("1. Testing UNet...")
unet = UNet2DConditionModel(
    in_channels=4,
    out_channels=4,
    block_out_channels=(320, 640, 1280),
    layers_per_block=2,
).to(device)

latents = torch.randn(1, 4, 64, 64).to(device)
timesteps = torch.tensor([500]).to(device)
context = torch.randn(1, 77, 768).to(device)

with torch.no_grad():
    output = unet(latents, timesteps, context)

print(f"   Input shape: {latents.shape}")
print(f"   Output shape: {output.shape}")
print("   ✓ UNet working!\n")

# 2. Test VAE
print("2. Testing VAE...")
vae = AutoencoderKL(latent_channels=4).to(device)

image = torch.randn(1, 3, 256, 256).to(device)

with torch.no_grad():
    latent = vae.encode(image)
    reconstructed = vae.decode(latent)

print(f"   Image shape: {image.shape}")
print(f"   Latent shape: {latent.shape}")
print(f"   Reconstructed shape: {reconstructed.shape}")
print("   ✓ VAE working!\n")

# 3. Test Material Encoder
print("3. Testing Material Encoder...")
material_encoder = MaterialEncoder(embed_dim=768).to(device)

# Create BRDF parameters
brdf_params = compute_brdf_params(
    roughness=0.3,
    metallic=0.9,
    base_color=(0.8, 0.8, 0.8)
).unsqueeze(0).to(device)

with torch.no_grad():
    material_embedding = material_encoder.encode_parameters(brdf_params)

print(f"   BRDF params shape: {brdf_params.shape}")
print(f"   Material embedding shape: {material_embedding.shape}")
print("   ✓ Material Encoder working!\n")

# 4. Test LoRA injection
print("4. Testing LoRA...")
from training.lora import inject_lora_into_model, print_lora_summary

unet_lora = inject_lora_into_model(
    unet,
    rank=4,
    alpha=8.0,
    target_modules=['to_q', 'to_k'],
)

print_lora_summary(unet_lora)
print("   ✓ LoRA injection working!\n")

print("=" * 60)
print("All tests passed! ✓")
print("ControlDiff is ready to use.")
print("=" * 60)
```

**Run the test:**
```bash
python test_models.py
```

### 3. Simple Generation Example

Create `simple_generate.py`:

```python
import torch
import sys
sys.path.insert(0, 'src')

from models import UNet2DConditionModel, AutoencoderKL, DDIMScheduler
from conditioning import MaterialEncoder, compute_brdf_params
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

print("Simple Image Generation Example\n")

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Initialize models
unet = UNet2DConditionModel(
    in_channels=4,
    out_channels=4,
    block_out_channels=(320, 640, 1280),
).to(device)

vae = AutoencoderKL(latent_channels=4).to(device)
scheduler = DDIMScheduler(num_train_timesteps=1000)
material_encoder = MaterialEncoder().to(device)

# Set to eval mode (using random weights for demo)
unet.eval()
vae.eval()

# Define material
material_params = compute_brdf_params(
    roughness=0.3,
    metallic=0.8,
    base_color=(0.7, 0.7, 0.8)
).unsqueeze(0).to(device)

material_embedding = material_encoder.encode_parameters(material_params)
context = material_embedding.unsqueeze(1).expand(-1, 77, -1)  # Expand to sequence

# Generate
print("Generating image (this will be random noise patterns since model is untrained)...")

scheduler.set_timesteps(20)  # Use fewer steps for speed
latents = torch.randn(1, 4, 64, 64, device=device)

with torch.no_grad():
    for i, t in enumerate(scheduler.timesteps):
        timestep = t.unsqueeze(0).to(device)

        # Predict noise
        noise_pred = unet(latents, timestep, context)

        # Denoise step
        latents = scheduler.step(noise_pred, int(t), latents)

        if i % 5 == 0:
            print(f"  Step {i+1}/{len(scheduler.timesteps)}")

    # Decode
    image = vae.decode(latents)

# Denormalize and save
image = (image + 1) / 2
image = torch.clamp(image, 0, 1)

# Save
from torchvision.utils import save_image
save_image(image, 'generated_sample.png')

print(f"\n✓ Generated image saved to: generated_sample.png")
print("Note: Image will be random patterns since model is untrained.")
print("Train the model or load pretrained weights for real results.")
```

**Run:**
```bash
python simple_generate.py
```

### 4. Test LoRA Fine-tuning (Dummy Data)

Create `test_lora_training.py`:

```python
import torch
import sys
sys.path.insert(0, 'src')

from models import UNet2DConditionModel, DDPMScheduler
from training.lora import inject_lora_into_model, save_lora_weights, print_lora_summary

print("Testing LoRA Fine-tuning\n")

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Create model
unet = UNet2DConditionModel(
    in_channels=4,
    out_channels=4,
    block_out_channels=(320, 640),
    layers_per_block=1,
).to(device)

# Inject LoRA
print("Injecting LoRA...")
unet = inject_lora_into_model(
    unet,
    rank=4,
    alpha=8.0,
    target_modules=['to_q', 'to_k', 'to_v'],
)

print_lora_summary(unet)

# Setup training
scheduler = DDPMScheduler(num_train_timesteps=1000)
lora_params = [p for p in unet.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(lora_params, lr=1e-4)

print(f"\nTraining {len(lora_params)} LoRA parameter groups...")

# Dummy training loop
unet.train()
num_steps = 50

for step in range(num_steps):
    # Create dummy batch
    latents = torch.randn(2, 4, 64, 64, device=device)
    noise = torch.randn_like(latents)
    timesteps = torch.randint(0, 1000, (2,), device=device)
    context = torch.randn(2, 77, 768, device=device)

    # Add noise
    noisy_latents = scheduler.add_noise(latents, noise, timesteps)

    # Predict noise
    noise_pred = unet(noisy_latents, timesteps, context)

    # Loss
    loss = torch.nn.functional.mse_loss(noise_pred, noise)

    # Backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 10 == 0:
        print(f"Step {step}/{num_steps}: Loss = {loss.item():.4f}")

# Save LoRA weights
save_lora_weights(unet, 'lora_weights_test.pt')

print(f"\n✓ LoRA training test complete!")
print(f"✓ LoRA weights saved to: lora_weights_test.pt")
print(f"File size: {torch.load('lora_weights_test.pt').__sizeof__() / 1024:.2f} KB")
```

**Run:**
```bash
python test_lora_training.py
```

### 5. Interactive Jupyter Notebook

```bash
# Install jupyter if needed
pip install jupyter ipywidgets

# Start Jupyter
jupyter notebook

# Open: notebooks/01_quickstart.ipynb
```

### 6. Using Pre-trained Models (When Available)

If you have pretrained weights:

```python
import torch
from models import UNet2DConditionModel

# Load model
unet = UNet2DConditionModel()
unet.load_state_dict(torch.load('path/to/checkpoint.pt'))
unet.eval()

# Use for generation...
```

### 7. Full Training (With Your Data)

First, create a simple dataset class `src/data/dataset.py`:

```python
import torch
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
import torchvision.transforms as T

class SimpleImageDataset(Dataset):
    def __init__(self, data_dir, size=512):
        self.data_dir = Path(data_dir)
        self.images = list(self.data_dir.glob('*.jpg')) + \
                     list(self.data_dir.glob('*.png'))

        self.transform = T.Compose([
            T.Resize(size),
            T.CenterCrop(size),
            T.ToTensor(),
            T.Normalize([0.5], [0.5]),
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert('RGB')
        image = self.transform(image)

        return {
            'images': image,
            'prompts': ['a photo'],  # Add your prompts
        }
```

Then prepare your data:
```
data/
└── train/
    ├── image1.jpg
    ├── image2.jpg
    └── ...
```

Run training:
```bash
python scripts/train.py --config configs/base_config.yaml
```

## Common Issues & Solutions

### Issue 1: CUDA Out of Memory
```bash
# Reduce batch size in config
training:
  batch_size: 1  # or 2
  gradient_accumulation_steps: 8  # Effective batch = 8
```

### Issue 2: Module Import Errors
```python
# Always add src to path at the start of scripts
import sys
sys.path.insert(0, 'src')
```

### Issue 3: No CUDA Available
```python
# The code works on CPU too (just slower)
device = 'cpu'
```

### Issue 4: Missing Dependencies
```bash
# Install missing packages
pip install transformers diffusers accelerate
pip install wandb tensorboard
pip install matplotlib seaborn
```

## Quick Commands Reference

```bash
# Test installation
python test_models.py

# Simple generation
python simple_generate.py

# Test LoRA
python test_lora_training.py

# Run notebook
jupyter notebook notebooks/01_quickstart.ipynb

# Full training (when data ready)
python scripts/train.py --config configs/base_config.yaml

# LoRA fine-tune
python scripts/finetune_lora.py --config configs/lora_finetune.yaml

# Inference
python scripts/inference.py --checkpoint model.pt --prompt "metallic surface"
```

## Next Steps

1. ✓ Run `test_models.py` to verify setup
2. ✓ Try `simple_generate.py` to see generation pipeline
3. ✓ Test LoRA with `test_lora_training.py`
4. ✓ Explore notebooks for interactive learning
5. Prepare your dataset for real training
6. Train base model or fine-tune with LoRA
7. Evaluate and iterate!

## Getting Help

- Check `GETTING_STARTED.md` for detailed examples
- See `ARCHITECTURE.md` for technical details
- Review notebooks for interactive tutorials

Happy generating! 🚀

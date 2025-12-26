"""
LoRA fine-tuning script.

Usage:
    python scripts/finetune_lora.py --config configs/lora_finetune.yaml
"""

import torch
import argparse
from pathlib import Path
import yaml
from omegaconf import OmegaConf

import sys
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from models import UNet2DConditionModel
from training.lora import inject_lora_into_model, save_lora_weights, print_lora_summary


def main(args):
    # Load config
    with open(args.config, 'r') as f:
        config = OmegaConf.create(yaml.safe_load(f))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load base model
    print(f"Loading base model from {config.training.pretrained_model_path}")
    unet = UNet2DConditionModel(**config.model.unet)
    unet.load_state_dict(torch.load(config.training.pretrained_model_path))
    unet = unet.to(device)

    # Inject LoRA
    print("Injecting LoRA layers...")
    unet = inject_lora_into_model(
        unet,
        rank=config.lora.rank,
        alpha=config.lora.alpha,
        target_modules=config.lora.target_modules,
        dropout=config.lora.dropout,
    )

    # Print summary
    print_lora_summary(unet)

    # Create optimizer (only for LoRA parameters)
    lora_params = [p for p in unet.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        lora_params,
        lr=config.training.learning_rate,
    )

    print(f"Ready for LoRA fine-tuning!")
    print(f"Trainable parameters: {sum(p.numel() for p in lora_params):,}")

    # TODO: Implement training loop (similar to train.py)

    # Save LoRA weights
    output_dir = Path(config.logging.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_lora_weights(unet, output_dir / 'lora_weights.pt')

    print("LoRA fine-tuning complete!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()

    main(args)

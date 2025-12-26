"""
Main training script for ControlDiff.

Usage:
    python scripts/train.py --config configs/base_config.yaml
"""

import torch
import argparse
from pathlib import Path
import yaml
from omegaconf import OmegaConf

import sys
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from models import UNet2DConditionModel, AutoencoderKL, DiffusionModel, DDPMScheduler
from training import DiffusionTrainer, setup_distributed, cleanup_distributed
from data.dataset import DiffusionDataset


def load_config(config_path: str):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return OmegaConf.create(config)


def create_model(config):
    """Create model from config."""
    # UNet
    unet = UNet2DConditionModel(
        **config.model.unet
    )

    # VAE
    vae = AutoencoderKL(
        **config.model.vae
    )

    # Diffusion scheduler
    if config.model.diffusion.scheduler_type == 'ddpm':
        scheduler = DDPMScheduler(
            num_train_timesteps=config.model.diffusion.num_train_timesteps,
            beta_start=config.model.diffusion.beta_start,
            beta_end=config.model.diffusion.beta_end,
            beta_schedule=config.model.diffusion.beta_schedule,
        )
    else:
        raise ValueError(f"Unknown scheduler: {config.model.diffusion.scheduler_type}")

    return unet, vae, scheduler


def main(args):
    # Load config
    config = load_config(args.config)

    # Set seed
    torch.manual_seed(config.seed)

    # Setup distributed training
    rank, world_size, local_rank = setup_distributed()
    device = f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu'

    print(f"Starting training on device: {device}")

    # Create models
    unet, vae, scheduler = create_model(config)

    # Load pretrained weights if specified
    if hasattr(config.training, 'pretrained_model_path'):
        print(f"Loading pretrained model from {config.training.pretrained_model_path}")
        unet.load_state_dict(torch.load(config.training.pretrained_model_path))

    # Create text encoder (simplified - in practice use CLIP or T5)
    from transformers import CLIPTextModel, CLIPTokenizer
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

    # Create optimizer
    optimizer = torch.optim.AdamW(
        unet.parameters(),
        lr=config.training.learning_rate,
        betas=(config.training.adam_beta1, config.training.adam_beta2),
        weight_decay=config.training.adam_weight_decay,
        eps=config.training.adam_epsilon,
    )

    # Create trainer
    trainer = DiffusionTrainer(
        model=unet,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        scheduler=scheduler,
        optimizer=optimizer,
        device=device,
        use_ema=config.training.use_ema,
        ema_decay=config.training.ema_decay,
        mixed_precision=config.training.mixed_precision,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        max_grad_norm=config.training.max_grad_norm,
        log_every_n_steps=config.training.log_every_n_steps,
        save_every_n_steps=config.training.save_every_n_steps,
        validate_every_n_steps=config.training.validate_every_n_steps,
        output_dir=config.logging.output_dir,
        use_wandb=config.logging.use_wandb,
        project_name=config.logging.wandb_project,
    )

    # Create datasets
    # NOTE: You'll need to implement DiffusionDataset
    # train_dataset = DiffusionDataset(config.data.train_data_dir, config.data.image_size)
    # train_loader = torch.utils.data.DataLoader(
    #     train_dataset,
    #     batch_size=config.training.batch_size,
    #     num_workers=config.data.num_workers,
    #     shuffle=True,
    # )

    # For now, use placeholder
    print("NOTE: Dataset implementation required. This is a template script.")

    # Train
    # trainer.train(train_loader, config.training.num_epochs)

    # Cleanup
    cleanup_distributed()

    print("Training complete!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    args = parser.parse_args()

    main(args)

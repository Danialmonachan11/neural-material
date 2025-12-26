"""
Main training loop for diffusion models with full experiment tracking.

Supports distributed training, mixed precision, gradient accumulation,
and comprehensive logging.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from typing import Optional, Dict, Any, Callable
from pathlib import Path
import wandb
from tqdm import tqdm
import time


class DiffusionTrainer:
    """
    Comprehensive trainer for diffusion models.

    Features:
    - Mixed precision training (FP16/BF16)
    - Gradient accumulation
    - EMA (Exponential Moving Average)
    - Checkpoint management
    - WandB/TensorBoard logging
    - Learning rate scheduling
    """

    def __init__(
        self,
        model: nn.Module,
        vae: nn.Module,
        text_encoder: nn.Module,
        tokenizer: Any,
        scheduler: Any,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: Optional[Any] = None,
        device: str = "cuda",
        use_ema: bool = True,
        ema_decay: float = 0.9999,
        mixed_precision: str = "fp16",  # 'no', 'fp16', 'bf16'
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        log_every_n_steps: int = 10,
        save_every_n_steps: int = 1000,
        validate_every_n_steps: int = 500,
        output_dir: str = "outputs",
        use_wandb: bool = True,
        project_name: str = "controldiff",
    ):
        self.model = model.to(device)
        self.vae = vae.to(device)
        self.text_encoder = text_encoder.to(device)
        self.tokenizer = tokenizer
        self.scheduler = scheduler
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.device = device

        # Freeze VAE and text encoder
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)

        # EMA
        self.use_ema = use_ema
        if use_ema:
            from .ema import EMAModel
            self.ema_model = EMAModel(model, decay=ema_decay)

        # Mixed precision
        self.mixed_precision = mixed_precision
        self.scaler = GradScaler() if mixed_precision == "fp16" else None

        # Training config
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.log_every_n_steps = log_every_n_steps
        self.save_every_n_steps = save_every_n_steps
        self.validate_every_n_steps = validate_every_n_steps

        # Output directory
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Logging
        self.use_wandb = use_wandb
        if use_wandb:
            wandb.init(project=project_name, config={
                "mixed_precision": mixed_precision,
                "gradient_accumulation": gradient_accumulation_steps,
                "ema_decay": ema_decay if use_ema else None,
            })

        # Metrics
        self.global_step = 0
        self.epoch = 0

    def encode_prompt(self, prompts: list) -> torch.Tensor:
        """Encode text prompts to conditioning vectors."""
        text_inputs = self.tokenizer(
            prompts,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(self.device)

        with torch.no_grad():
            encoder_hidden_states = self.text_encoder(text_input_ids)[0]

        return encoder_hidden_states

    def training_step(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        """
        Single training step.

        Args:
            batch: batch with 'images' and 'prompts'
        Returns:
            metrics dictionary
        """
        images = batch["images"].to(self.device)
        prompts = batch["prompts"]

        # Encode images to latents
        with torch.no_grad():
            latents = self.vae.encode(images)

        # Encode prompts
        encoder_hidden_states = self.encode_prompt(prompts)

        # Sample noise and timesteps
        noise = torch.randn_like(latents)
        batch_size = latents.shape[0]
        timesteps = torch.randint(
            0, self.scheduler.num_train_timesteps,
            (batch_size,),
            device=self.device
        ).long()

        # Add noise to latents
        noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)

        # Forward pass with mixed precision
        with autocast(enabled=(self.mixed_precision == "fp16")):
            # Predict noise
            noise_pred = self.model(
                noisy_latents,
                timesteps,
                encoder_hidden_states=encoder_hidden_states,
            )

            # Compute loss
            loss = nn.functional.mse_loss(noise_pred, noise)

            # Scale loss for gradient accumulation
            loss = loss / self.gradient_accumulation_steps

        # Backward pass
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        metrics = {
            "loss": loss.item() * self.gradient_accumulation_steps,
            "lr": self.optimizer.param_groups[0]["lr"],
        }

        return metrics

    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int,
    ):
        """Train for one epoch."""
        self.model.train()
        self.epoch = epoch

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")

        for step, batch in enumerate(progress_bar):
            # Training step
            metrics = self.training_step(batch)

            # Gradient accumulation
            if (step + 1) % self.gradient_accumulation_steps == 0:
                # Clip gradients
                if self.max_grad_norm > 0:
                    if self.scaler is not None:
                        self.scaler.unscale_(self.optimizer)

                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.max_grad_norm
                    )

                # Optimizer step
                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()

                self.optimizer.zero_grad()

                # Update EMA
                if self.use_ema:
                    self.ema_model.step(self.model)

                self.global_step += 1

                # Logging
                if self.global_step % self.log_every_n_steps == 0:
                    progress_bar.set_postfix(metrics)

                    if self.use_wandb:
                        wandb.log(metrics, step=self.global_step)

                # Checkpointing
                if self.global_step % self.save_every_n_steps == 0:
                    self.save_checkpoint()

    def save_checkpoint(self, name: Optional[str] = None):
        """Save training checkpoint."""
        if name is None:
            name = f"checkpoint-{self.global_step}"

        checkpoint_path = self.output_dir / name
        checkpoint_path.mkdir(exist_ok=True)

        # Save model
        torch.save(
            self.model.state_dict(),
            checkpoint_path / "model.pt"
        )

        # Save EMA
        if self.use_ema:
            torch.save(
                self.ema_model.state_dict(),
                checkpoint_path / "model_ema.pt"
            )

        # Save optimizer
        torch.save(
            self.optimizer.state_dict(),
            checkpoint_path / "optimizer.pt"
        )

        # Save training state
        torch.save({
            "global_step": self.global_step,
            "epoch": self.epoch,
        }, checkpoint_path / "training_state.pt")

        print(f"Saved checkpoint to {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint."""
        checkpoint_path = Path(checkpoint_path)

        # Load model
        self.model.load_state_dict(
            torch.load(checkpoint_path / "model.pt")
        )

        # Load EMA
        if self.use_ema and (checkpoint_path / "model_ema.pt").exists():
            self.ema_model.load_state_dict(
                torch.load(checkpoint_path / "model_ema.pt")
            )

        # Load optimizer
        if (checkpoint_path / "optimizer.pt").exists():
            self.optimizer.load_state_dict(
                torch.load(checkpoint_path / "optimizer.pt")
            )

        # Load training state
        if (checkpoint_path / "training_state.pt").exists():
            state = torch.load(checkpoint_path / "training_state.pt")
            self.global_step = state["global_step"]
            self.epoch = state["epoch"]

        print(f"Loaded checkpoint from {checkpoint_path}")

    def train(
        self,
        train_loader: DataLoader,
        num_epochs: int,
        val_loader: Optional[DataLoader] = None,
    ):
        """Full training loop."""
        print("Starting training...")
        print(f"Total epochs: {num_epochs}")
        print(f"Steps per epoch: {len(train_loader)}")

        for epoch in range(self.epoch, num_epochs):
            self.train_epoch(train_loader, epoch)

            # Validation
            if val_loader is not None:
                self.validate(val_loader)

        # Save final checkpoint
        self.save_checkpoint("final")

        if self.use_wandb:
            wandb.finish()

        print("Training complete!")

    @torch.no_grad()
    def validate(self, val_loader: DataLoader):
        """Run validation."""
        self.model.eval()

        total_loss = 0
        num_batches = 0

        for batch in tqdm(val_loader, desc="Validation"):
            images = batch["images"].to(self.device)
            prompts = batch["prompts"]

            # Encode
            latents = self.vae.encode(images)
            encoder_hidden_states = self.encode_prompt(prompts)

            # Sample noise
            noise = torch.randn_like(latents)
            batch_size = latents.shape[0]
            timesteps = torch.randint(
                0, self.scheduler.num_train_timesteps,
                (batch_size,),
                device=self.device
            ).long()

            # Add noise
            noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)

            # Predict
            noise_pred = self.model(
                noisy_latents,
                timesteps,
                encoder_hidden_states=encoder_hidden_states,
            )

            # Loss
            loss = nn.functional.mse_loss(noise_pred, noise)
            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches

        print(f"Validation loss: {avg_loss:.4f}")

        if self.use_wandb:
            wandb.log({"val_loss": avg_loss}, step=self.global_step)

        self.model.train()

"""
DreamBooth implementation for subject-specific personalization.

DreamBooth fine-tunes diffusion models on a few images of a specific subject
while preserving the model's ability to generate diverse outputs.

Paper: https://arxiv.org/abs/2208.12242
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Optional, List, Dict, Tuple
from pathlib import Path
import random
from PIL import Image
from torchvision import transforms


class DreamBoothDataset(Dataset):
    """
    Dataset for DreamBooth training with prior preservation.

    Combines instance images (few-shot subject examples) with class images
    (generated images of the same class) to prevent overfitting.
    """

    def __init__(
        self,
        instance_images_path: str,
        instance_prompt: str,
        class_images_path: Optional[str] = None,
        class_prompt: Optional[str] = None,
        size: int = 512,
        center_crop: bool = True,
        use_prior_preservation: bool = True,
    ):
        """
        Args:
            instance_images_path: path to folder with instance images
            instance_prompt: prompt template with unique identifier (e.g., "a photo of sks dog")
            class_images_path: path to folder with class images for prior preservation
            class_prompt: class prompt (e.g., "a photo of a dog")
            size: image size
            center_crop: whether to center crop
            use_prior_preservation: whether to use prior preservation loss
        """
        self.size = size
        self.center_crop = center_crop
        self.instance_prompt = instance_prompt
        self.class_prompt = class_prompt
        self.use_prior_preservation = use_prior_preservation

        # Load instance images
        self.instance_images_path = Path(instance_images_path)
        if not self.instance_images_path.exists():
            raise ValueError(f"Instance images path doesn't exist: {instance_images_path}")

        self.instance_images = list(self.instance_images_path.glob("*.jpg")) + \
                              list(self.instance_images_path.glob("*.png"))

        if len(self.instance_images) == 0:
            raise ValueError(f"No images found in {instance_images_path}")

        print(f"Found {len(self.instance_images)} instance images")

        # Load class images for prior preservation
        self.class_images = []
        if use_prior_preservation and class_images_path:
            self.class_images_path = Path(class_images_path)
            if self.class_images_path.exists():
                self.class_images = list(self.class_images_path.glob("*.jpg")) + \
                                   list(self.class_images_path.glob("*.png"))
                print(f"Found {len(self.class_images)} class images")
            else:
                print(f"Warning: Class images path doesn't exist: {class_images_path}")

        # Image transformations
        self.image_transforms = transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def __len__(self):
        return len(self.instance_images)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns a dictionary with:
        - instance_images: instance image tensor
        - instance_prompt: instance prompt string
        - class_images: class image tensor (if using prior preservation)
        - class_prompt: class prompt string (if using prior preservation)
        """
        # Load instance image
        instance_image = Image.open(self.instance_images[idx % len(self.instance_images)])
        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")
        instance_image = self.image_transforms(instance_image)

        example = {
            "instance_images": instance_image,
            "instance_prompt": self.instance_prompt,
        }

        # Load class image for prior preservation
        if self.use_prior_preservation and len(self.class_images) > 0:
            class_image = Image.open(self.class_images[idx % len(self.class_images)])
            if not class_image.mode == "RGB":
                class_image = class_image.convert("RGB")
            class_image = self.image_transforms(class_image)

            example["class_images"] = class_image
            example["class_prompt"] = self.class_prompt

        return example


class DreamBoothTrainer:
    """
    Trainer for DreamBooth fine-tuning.

    Implements prior preservation loss and handles the training loop.
    """

    def __init__(
        self,
        model: nn.Module,
        vae: nn.Module,
        text_encoder: nn.Module,
        tokenizer,
        device: str = "cuda",
        prior_loss_weight: float = 1.0,
        learning_rate: float = 5e-6,
        use_lora: bool = False,
        lora_rank: int = 4,
    ):
        """
        Args:
            model: diffusion model (UNet)
            vae: variational autoencoder
            text_encoder: text encoder for prompts
            tokenizer: tokenizer for text encoder
            device: device to train on
            prior_loss_weight: weight for prior preservation loss
            learning_rate: learning rate
            use_lora: whether to use LoRA for efficient fine-tuning
            lora_rank: rank for LoRA matrices
        """
        self.device = device
        self.prior_loss_weight = prior_loss_weight
        self.use_lora = use_lora

        # Move models to device
        self.model = model.to(device)
        self.vae = vae.to(device)
        self.text_encoder = text_encoder.to(device)
        self.tokenizer = tokenizer

        # Freeze VAE and text encoder
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)

        # Optionally inject LoRA
        if use_lora:
            from .lora import inject_lora_into_model
            self.model = inject_lora_into_model(
                self.model,
                rank=lora_rank,
                target_modules=['to_q', 'to_k', 'to_v', 'to_out.0'],
            )

        # Optimizer
        params_to_optimize = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(params_to_optimize, lr=learning_rate)

    def encode_prompt(self, prompt: str) -> torch.Tensor:
        """
        Encode text prompt to conditioning vector.

        Args:
            prompt: text prompt
        Returns:
            (1, seq_len, hidden_dim) conditioning tensor
        """
        text_inputs = self.tokenizer(
            prompt,
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
        scheduler,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Single training step with optional prior preservation.

        Args:
            batch: batch from DreamBoothDataset
            scheduler: noise scheduler
        Returns:
            loss and metrics dict
        """
        # Encode instance images
        instance_images = batch["instance_images"].to(self.device)
        with torch.no_grad():
            instance_latents = self.vae.encode(instance_images).sample()

        # Encode instance prompts
        instance_prompt_embeds = self.encode_prompt(batch["instance_prompt"][0])
        instance_prompt_embeds = instance_prompt_embeds.repeat(instance_images.shape[0], 1, 1)

        # Sample noise and timesteps
        noise = torch.randn_like(instance_latents)
        batch_size = instance_latents.shape[0]
        timesteps = torch.randint(
            0, scheduler.num_train_timesteps, (batch_size,),
            device=self.device
        ).long()

        # Add noise to latents
        noisy_latents = scheduler.add_noise(instance_latents, noise, timesteps)

        # Predict noise
        noise_pred = self.model(
            noisy_latents,
            timesteps,
            encoder_hidden_states=instance_prompt_embeds,
        )

        # Compute instance loss
        instance_loss = F.mse_loss(noise_pred, noise, reduction="mean")

        # Prior preservation loss
        prior_loss = torch.tensor(0.0, device=self.device)
        if "class_images" in batch:
            class_images = batch["class_images"].to(self.device)

            with torch.no_grad():
                class_latents = self.vae.encode(class_images).sample()

            class_prompt_embeds = self.encode_prompt(batch["class_prompt"][0])
            class_prompt_embeds = class_prompt_embeds.repeat(class_images.shape[0], 1, 1)

            # Sample noise and timesteps for class images
            class_noise = torch.randn_like(class_latents)
            class_timesteps = torch.randint(
                0, scheduler.num_train_timesteps, (class_latents.shape[0],),
                device=self.device
            ).long()

            noisy_class_latents = scheduler.add_noise(class_latents, class_noise, class_timesteps)

            # Predict noise for class images
            class_noise_pred = self.model(
                noisy_class_latents,
                class_timesteps,
                encoder_hidden_states=class_prompt_embeds,
            )

            prior_loss = F.mse_loss(class_noise_pred, class_noise, reduction="mean")

        # Total loss
        loss = instance_loss + self.prior_loss_weight * prior_loss

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        metrics = {
            "loss": loss.item(),
            "instance_loss": instance_loss.item(),
            "prior_loss": prior_loss.item(),
        }

        return loss, metrics

    def train(
        self,
        dataloader: DataLoader,
        scheduler,
        num_epochs: int = 100,
        save_steps: int = 100,
        output_dir: str = "outputs",
    ):
        """
        Full training loop.

        Args:
            dataloader: DreamBooth dataloader
            scheduler: noise scheduler
            num_epochs: number of training epochs
            save_steps: save checkpoint every N steps
            output_dir: output directory for checkpoints
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        global_step = 0
        self.model.train()

        for epoch in range(num_epochs):
            for batch in dataloader:
                loss, metrics = self.training_step(batch, scheduler)

                if global_step % 10 == 0:
                    print(
                        f"Epoch {epoch}, Step {global_step}: "
                        f"Loss={metrics['loss']:.4f}, "
                        f"Instance={metrics['instance_loss']:.4f}, "
                        f"Prior={metrics['prior_loss']:.4f}"
                    )

                if global_step % save_steps == 0:
                    checkpoint_path = output_path / f"checkpoint-{global_step}.pt"
                    if self.use_lora:
                        from .lora import extract_lora_weights
                        lora_weights = extract_lora_weights(self.model)
                        torch.save(lora_weights, checkpoint_path)
                    else:
                        torch.save(self.model.state_dict(), checkpoint_path)
                    print(f"Saved checkpoint to {checkpoint_path}")

                global_step += 1

        # Save final model
        final_path = output_path / "final_model.pt"
        if self.use_lora:
            from .lora import extract_lora_weights
            lora_weights = extract_lora_weights(self.model)
            torch.save(lora_weights, final_path)
        else:
            torch.save(self.model.state_dict(), final_path)
        print(f"Training complete! Saved final model to {final_path}")


def generate_class_images(
    model,
    vae,
    text_encoder,
    tokenizer,
    class_prompt: str,
    num_images: int = 200,
    output_dir: str = "class_images",
    device: str = "cuda",
):
    """
    Generate class images for prior preservation.

    Args:
        model: diffusion model
        vae: VAE
        text_encoder: text encoder
        tokenizer: tokenizer
        class_prompt: prompt for class images (e.g., "a photo of a dog")
        num_images: number of class images to generate
        output_dir: output directory
        device: device
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    model.eval()
    vae.eval()

    print(f"Generating {num_images} class images with prompt: '{class_prompt}'")

    for i in range(num_images):
        with torch.no_grad():
            # Encode prompt
            text_inputs = tokenizer(
                class_prompt,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids.to(device)
            encoder_hidden_states = text_encoder(text_input_ids)[0]

            # Generate latent
            latents = model.sample(
                batch_size=1,
                encoder_hidden_states=encoder_hidden_states,
                height=64,
                width=64,
                num_inference_steps=50,
            )

            # Decode to image
            images = vae.decode(latents)

            # Save image
            image = (images[0] * 0.5 + 0.5).clamp(0, 1)
            image = transforms.ToPILImage()(image.cpu())
            image.save(output_path / f"class_image_{i:04d}.png")

        if (i + 1) % 10 == 0:
            print(f"Generated {i + 1}/{num_images} images")

    print(f"Class image generation complete! Saved to {output_path}")

"""
Inference script for generating images with ControlDiff.

Usage:
    python scripts/inference.py \
        --checkpoint checkpoints/model.pt \
        --prompt "a photo of metallic material" \
        --output outputs/generated.png
"""

import torch
import argparse
from pathlib import Path
from PIL import Image
import sys

sys.path.append(str(Path(__file__).parent.parent / 'src'))

from models import UNet2DConditionModel, AutoencoderKL, DDIMScheduler
from transformers import CLIPTextModel, CLIPTokenizer


@torch.no_grad()
def generate_image(
    unet,
    vae,
    text_encoder,
    tokenizer,
    scheduler,
    prompt: str,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    device: str = 'cuda',
):
    """Generate image from text prompt."""

    # Encode prompt
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids.to(device)
    encoder_hidden_states = text_encoder(text_input_ids)[0]

    # Set timesteps
    scheduler.set_timesteps(num_inference_steps)

    # Start from random noise
    latents = torch.randn(1, 4, 64, 64, device=device)

    # Denoising loop
    for t in scheduler.timesteps:
        # Predict noise
        latent_model_input = latents
        timestep = t.to(device).unsqueeze(0)

        noise_pred = unet(
            latent_model_input,
            timestep,
            encoder_hidden_states=encoder_hidden_states,
        )

        # Classifier-free guidance
        if guidance_scale > 1.0:
            noise_pred_uncond = unet(
                latent_model_input,
                timestep,
                encoder_hidden_states=torch.zeros_like(encoder_hidden_states),
            )
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred - noise_pred_uncond)

        # Denoise
        latents = scheduler.step(noise_pred, int(t), latents)

    # Decode latents
    image = vae.decode(latents)

    # Convert to PIL
    image = (image[0] / 2 + 0.5).clamp(0, 1)
    image = image.permute(1, 2, 0).cpu().numpy()
    image = (image * 255).astype('uint8')
    image = Image.fromarray(image)

    return image


def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load models
    print("Loading models...")

    unet = UNet2DConditionModel()
    unet.load_state_dict(torch.load(args.checkpoint, map_location=device))
    unet = unet.to(device)
    unet.eval()

    vae = AutoencoderKL()
    vae = vae.to(device)
    vae.eval()

    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
    text_encoder = text_encoder.to(device)
    text_encoder.eval()

    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

    scheduler = DDIMScheduler(num_train_timesteps=1000)

    # Generate image
    print(f"Generating image for prompt: '{args.prompt}'")

    image = generate_image(
        unet=unet,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        scheduler=scheduler,
        prompt=args.prompt,
        num_inference_steps=args.num_steps,
        guidance_scale=args.guidance_scale,
        device=device,
    )

    # Save image
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)

    print(f"Saved generated image to {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--prompt', type=str, required=True, help='Text prompt')
    parser.add_argument('--output', type=str, default='output.png', help='Output image path')
    parser.add_argument('--num-steps', type=int, default=50, help='Number of inference steps')
    parser.add_argument('--guidance-scale', type=float, default=7.5, help='Guidance scale')

    args = parser.parse_args()
    main(args)

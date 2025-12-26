"""
Evaluation metrics for diffusion models.

Includes FID, CLIP score, Inception Score, and custom material accuracy metrics.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Optional
from scipy import linalg
import clip
from torchvision.models import inception_v3
from pytorch_fid import fid_score


class FIDScore:
    """
    Frechet Inception Distance (FID) for measuring image quality.

    Lower FID = better quality and diversity.
    """

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.inception = inception_v3(pretrained=True, transform_input=False).to(device)
        self.inception.eval()

    @torch.no_grad()
    def extract_features(self, images: torch.Tensor) -> np.ndarray:
        """
        Extract Inception features from images.

        Args:
            images: (N, 3, H, W) images in [0, 1]
        Returns:
            (N, 2048) feature array
        """
        # Resize to 299x299 for Inception
        if images.shape[-2:] != (299, 299):
            images = nn.functional.interpolate(
                images, size=(299, 299), mode='bilinear', align_corners=False
            )

        # Normalize for Inception
        images = images * 2 - 1  # [0, 1] -> [-1, 1]

        # Extract features
        features = self.inception(images.to(self.device))

        return features.cpu().numpy()

    def compute_statistics(self, features: np.ndarray) -> tuple:
        """Compute mean and covariance of features."""
        mu = np.mean(features, axis=0)
        sigma = np.cov(features, rowvar=False)
        return mu, sigma

    def calculate_fid(
        self,
        real_images: torch.Tensor,
        generated_images: torch.Tensor,
    ) -> float:
        """
        Calculate FID between real and generated images.

        Args:
            real_images: (N, 3, H, W) real images
            generated_images: (M, 3, H, W) generated images
        Returns:
            FID score
        """
        # Extract features
        real_features = self.extract_features(real_images)
        gen_features = self.extract_features(generated_images)

        # Compute statistics
        mu_real, sigma_real = self.compute_statistics(real_features)
        mu_gen, sigma_gen = self.compute_statistics(gen_features)

        # Compute FID
        fid = self._compute_fid(mu_real, sigma_real, mu_gen, sigma_gen)

        return fid

    def _compute_fid(
        self,
        mu1: np.ndarray,
        sigma1: np.ndarray,
        mu2: np.ndarray,
        sigma2: np.ndarray,
        eps: float = 1e-6,
    ) -> float:
        """Compute FID from statistics."""
        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)

        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)

        diff = mu1 - mu2

        # Product might be almost singular
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        # Numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            covmean = covmean.real

        fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)

        return float(fid)


class CLIPScore:
    """
    CLIP score for text-image alignment.

    Higher score = better alignment between text and image.
    """

    def __init__(self, device: str = "cuda", model_name: str = "ViT-B/32"):
        self.device = device
        self.model, self.preprocess = clip.load(model_name, device=device)
        self.model.eval()

    @torch.no_grad()
    def compute_score(
        self,
        images: torch.Tensor,
        texts: List[str],
    ) -> float:
        """
        Compute CLIP score between images and texts.

        Args:
            images: (N, 3, H, W) images in [0, 1]
            texts: list of N text prompts
        Returns:
            average CLIP score
        """
        # Preprocess images
        # CLIP expects specific preprocessing, so convert properly
        from torchvision import transforms
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711]
            )
        ])

        processed_images = []
        for img in images:
            processed_images.append(preprocess(img))
        image_inputs = torch.stack(processed_images).to(self.device)

        # Tokenize texts
        text_inputs = clip.tokenize(texts).to(self.device)

        # Compute features
        image_features = self.model.encode_image(image_inputs)
        text_features = self.model.encode_text(text_inputs)

        # Normalize features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # Compute similarity
        similarity = (image_features * text_features).sum(dim=-1)

        return similarity.mean().item()


class InceptionScore:
    """
    Inception Score for measuring quality and diversity.

    Higher score = better quality and diversity.
    """

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.inception = inception_v3(pretrained=True, transform_input=False).to(device)
        self.inception.eval()

    @torch.no_grad()
    def compute_score(
        self,
        images: torch.Tensor,
        splits: int = 10,
    ) -> tuple:
        """
        Compute Inception Score.

        Args:
            images: (N, 3, H, W) images in [0, 1]
            splits: number of splits for computing mean/std
        Returns:
            (mean_score, std_score)
        """
        N = images.shape[0]

        # Get predictions
        preds = []
        for i in range(0, N, 32):  # Batch size 32
            batch = images[i:i+32]

            # Resize and normalize
            if batch.shape[-2:] != (299, 299):
                batch = nn.functional.interpolate(
                    batch, size=(299, 299), mode='bilinear', align_corners=False
                )
            batch = batch * 2 - 1

            # Get predictions
            pred = nn.functional.softmax(self.inception(batch.to(self.device)), dim=1)
            preds.append(pred.cpu().numpy())

        preds = np.concatenate(preds, axis=0)

        # Compute score for each split
        split_scores = []
        for k in range(splits):
            part = preds[k * (N // splits):(k + 1) * (N // splits), :]
            py = np.mean(part, axis=0)
            scores = []
            for i in range(part.shape[0]):
                pyx = part[i, :]
                scores.append(np.sum(pyx * np.log(pyx / py + 1e-10)))
            split_scores.append(np.exp(np.mean(scores)))

        return np.mean(split_scores), np.std(split_scores)


class MaterialAccuracy:
    """
    Custom metric for material classification accuracy.

    Measures how well the generated material matches the target.
    """

    def __init__(self, material_encoder: nn.Module, device: str = "cuda"):
        self.material_encoder = material_encoder.to(device)
        self.material_encoder.eval()
        self.device = device

    @torch.no_grad()
    def compute_accuracy(
        self,
        generated_images: torch.Tensor,
        target_materials: torch.Tensor,
    ) -> float:
        """
        Compute material classification accuracy.

        Args:
            generated_images: (N, 3, H, W) generated images
            target_materials: (N,) target material category indices
        Returns:
            accuracy score
        """
        # Extract material features from images
        material_features = self.material_encoder.encode_texture(
            generated_images.to(self.device)
        )

        # Classify materials (simplified - assumes encoder has classification head)
        # In practice, you'd have a separate classifier
        pred_materials = material_features.argmax(dim=-1)

        # Compute accuracy
        accuracy = (pred_materials == target_materials.to(self.device)).float().mean()

        return accuracy.item()


def compute_all_metrics(
    real_images: torch.Tensor,
    generated_images: torch.Tensor,
    prompts: List[str],
    device: str = "cuda",
) -> Dict[str, float]:
    """
    Compute all evaluation metrics.

    Args:
        real_images: (N, 3, H, W) real images
        generated_images: (N, 3, H, W) generated images
        prompts: list of prompts
        device: device
    Returns:
        dictionary of metrics
    """
    metrics = {}

    # FID
    fid_metric = FIDScore(device=device)
    metrics["fid"] = fid_metric.calculate_fid(real_images, generated_images)

    # CLIP Score
    clip_metric = CLIPScore(device=device)
    metrics["clip_score"] = clip_metric.compute_score(generated_images, prompts)

    # Inception Score
    is_metric = InceptionScore(device=device)
    mean_is, std_is = is_metric.compute_score(generated_images)
    metrics["inception_score_mean"] = mean_is
    metrics["inception_score_std"] = std_is

    return metrics


def evaluate_diversity(
    generated_images: torch.Tensor,
    num_samples: int = 1000,
) -> Dict[str, float]:
    """
    Evaluate diversity of generated images.

    Args:
        generated_images: (N, 3, H, W) generated images
        num_samples: number of samples for diversity computation
    Returns:
        diversity metrics
    """
    N = min(generated_images.shape[0], num_samples)
    images = generated_images[:N]

    # Flatten images
    images_flat = images.reshape(N, -1).cpu().numpy()

    # Compute pairwise distances
    from sklearn.metrics.pairwise import cosine_distances

    distances = cosine_distances(images_flat)

    # Average distance (higher = more diverse)
    avg_distance = distances.sum() / (N * (N - 1))

    # Min distance (measure of mode collapse)
    np.fill_diagonal(distances, np.inf)
    min_distance = distances.min(axis=1).mean()

    return {
        "diversity_avg": avg_distance,
        "diversity_min": min_distance,
    }

from __future__ import annotations

import csv
import json
import os
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models





OMNIGLOT_NORMALIZE_MEAN = 0.92206 * 256.0
OMNIGLOT_NORMALIZE_STD = 0.08426 * 256.0 * 256.0


def omniglot_model_to_unit_interval(images: torch.Tensor) -> torch.Tensor:
    pixels_255 = images * OMNIGLOT_NORMALIZE_STD + OMNIGLOT_NORMALIZE_MEAN
    return (pixels_255 / 255.0).clamp(0.0, 1.0)


def model_to_unit_interval(images: torch.Tensor, dataset: str) -> torch.Tensor:
    if dataset.lower() == "cifar100":
        return ((images + 1.0) / 2.0).clamp(0.0, 1.0)
    if "omni" in dataset.lower():
        return omniglot_model_to_unit_interval(images)
    raise ValueError(f"Unsupported fidelity dataset: {dataset}")


class OmniglotReferenceClassifier(nn.Module):
    """Independent CNN used only to evaluate real/reconstructed images."""

    def __init__(self, num_classes: int = 963, feature_dim: int = 256, dropout: float = 0.0):
        super().__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.projection = nn.Linear(128 * 4 * 4, feature_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, images: torch.Tensor, return_features: bool = False):
        hidden = self.features(images).flatten(start_dim=1)
        features = F.relu(self.projection(hidden), inplace=False)
        logits = self.classifier(self.dropout(features))
        if return_features:
            return logits, features
        return logits


class CIFAR100ReferenceClassifier(nn.Module):
    """CIFAR-adapted ResNet-18 with an exposed frozen feature vector."""

    def __init__(self, num_classes: int = 100, dropout: float = 0.0):
        super().__init__()
        backbone = models.resnet18(weights=None)
        backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        backbone.maxpool = nn.Identity()
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        self.feature_dim = 512
        self.num_classes = num_classes
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.feature_dim, num_classes)
        self.register_buffer("input_mean", torch.tensor((0.5071, 0.4867, 0.4408)).view(1, 3, 1, 1))
        self.register_buffer("input_std", torch.tensor((0.2675, 0.2565, 0.2761)).view(1, 3, 1, 1))

    def forward(self, images: torch.Tensor, return_features: bool = False):
        normalized = (images - self.input_mean) / self.input_std
        features = self.features(normalized).flatten(start_dim=1)
        logits = self.classifier(self.dropout(features))
        if return_features:
            return logits, features
        return logits


def load_reference_classifier(path: str, device: torch.device) -> nn.Module:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
        raise ValueError("Reference classifier checkpoint must contain a state_dict and config.")
    config = checkpoint.get("config", {})
    dataset = str(config.get("dataset", "omniglot")).lower()
    if dataset == "cifar100":
        model = CIFAR100ReferenceClassifier(
            num_classes=int(config.get("num_classes", 100)),
            dropout=float(config.get("dropout", 0.0)),
        )
    else:
        model = OmniglotReferenceClassifier(
            num_classes=int(config.get("num_classes", 963)),
            feature_dim=int(config.get("feature_dim", 256)),
            dropout=float(config.get("dropout", 0.0)),
        )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _gaussian_window(channels: int, size: int, sigma: float, device, dtype) -> torch.Tensor:
    coordinates = torch.arange(size, device=device, dtype=dtype) - (size - 1) / 2
    kernel_1d = torch.exp(-(coordinates.pow(2)) / (2 * sigma * sigma))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    return kernel_2d.expand(channels, 1, size, size).contiguous()


def ssim_per_sample(
    real: torch.Tensor,
    pseudo: torch.Tensor,
    data_range: float = 1.0,
    window_size: int = 7,
    sigma: float = 1.5,
) -> torch.Tensor:
    """Channel-averaged SSIM for small 28x28 images, returned per sample."""
    if real.shape != pseudo.shape:
        raise ValueError(f"SSIM inputs must have equal shapes, got {real.shape} and {pseudo.shape}.")
    channels = real.size(1)
    window = _gaussian_window(channels, window_size, sigma, real.device, real.dtype)
    padding = window_size // 2
    mu_real = F.conv2d(real, window, padding=padding, groups=channels)
    mu_pseudo = F.conv2d(pseudo, window, padding=padding, groups=channels)
    mu_real_sq = mu_real.pow(2)
    mu_pseudo_sq = mu_pseudo.pow(2)
    mu_cross = mu_real * mu_pseudo
    sigma_real = F.conv2d(real * real, window, padding=padding, groups=channels) - mu_real_sq
    sigma_pseudo = F.conv2d(pseudo * pseudo, window, padding=padding, groups=channels) - mu_pseudo_sq
    sigma_cross = F.conv2d(real * pseudo, window, padding=padding, groups=channels) - mu_cross
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    numerator = (2 * mu_cross + c1) * (2 * sigma_cross + c2)
    denominator = (mu_real_sq + mu_pseudo_sq + c1) * (sigma_real + sigma_pseudo + c2)
    return (numerator / denominator.clamp_min(1e-12)).flatten(start_dim=1).mean(dim=1)


def _safe_std(values: torch.Tensor) -> float:
    return values.std(unbiased=True).item() if values.numel() > 1 else 0.0


@dataclass
class FidelitySummary:
    sample_count: int
    pixel_mse_mean: float
    pixel_mse_std: float
    ssim_mean: float
    ssim_std: float
    frozen_feature_cosine_mean: float
    frozen_feature_cosine_std: float
    independent_real_accuracy: float
    independent_pseudo_accuracy: float
    prediction_agreement_rate: float
    label_preservation_rate: float


def compute_fidelity_metrics(
    stored_images_model_space: torch.Tensor,
    pseudo_images_model_space: torch.Tensor,
    labels: torch.Tensor,
    reference_classifier: nn.Module,
    dataset: str = "omniglot",
) -> Tuple[FidelitySummary, Dict[str, torch.Tensor]]:
    real = model_to_unit_interval(stored_images_model_space, dataset)
    pseudo = model_to_unit_interval(pseudo_images_model_space, dataset)
    labels = labels.view(-1).long()

    pixel_mse = (real - pseudo).pow(2).flatten(start_dim=1).mean(dim=1)
    ssim = ssim_per_sample(real, pseudo)
    real_logits, real_features = reference_classifier(real, return_features=True)
    pseudo_logits, pseudo_features = reference_classifier(pseudo, return_features=True)
    feature_cosine = F.cosine_similarity(real_features, pseudo_features, dim=1)
    real_predictions = real_logits.argmax(dim=1)
    pseudo_predictions = pseudo_logits.argmax(dim=1)
    real_correct = real_predictions.eq(labels)
    pseudo_correct = pseudo_predictions.eq(labels)
    prediction_agreement = real_predictions.eq(pseudo_predictions)
    preserved = real_correct & pseudo_correct
    denominator = real_correct.sum().item()
    lpr = preserved.sum().item() / denominator if denominator else float("nan")

    summary = FidelitySummary(
        sample_count=len(labels),
        pixel_mse_mean=pixel_mse.mean().item(),
        pixel_mse_std=_safe_std(pixel_mse),
        ssim_mean=ssim.mean().item(),
        ssim_std=_safe_std(ssim),
        frozen_feature_cosine_mean=feature_cosine.mean().item(),
        frozen_feature_cosine_std=_safe_std(feature_cosine),
        independent_real_accuracy=real_correct.float().mean().item(),
        independent_pseudo_accuracy=pseudo_correct.float().mean().item(),
        prediction_agreement_rate=prediction_agreement.float().mean().item(),
        label_preservation_rate=lpr,
    )
    per_sample = {
        "pixel_mse": pixel_mse,
        "ssim": ssim,
        "frozen_feature_cosine": feature_cosine,
        "real_prediction": real_predictions,
        "pseudo_prediction": pseudo_predictions,
        "real_correct": real_correct,
        "pseudo_correct": pseudo_correct,
        "prediction_agreement": prediction_agreement,
        "label_preserved": preserved,
    }
    return summary, per_sample


def compute_fidelity_metrics_batched(
    stored_images_model_space: torch.Tensor,
    pseudo_images_model_space: torch.Tensor,
    labels: torch.Tensor,
    reference_classifier: nn.Module,
    dataset: str,
    batch_size: int = 128,
) -> Tuple[FidelitySummary, Dict[str, torch.Tensor]]:
    device = next(reference_classifier.parameters()).device
    collected = {}
    for start in range(0, len(labels), batch_size):
        stop = min(start + batch_size, len(labels))
        _, per_sample = compute_fidelity_metrics(
            stored_images_model_space[start:stop].to(device),
            pseudo_images_model_space[start:stop].to(device),
            labels[start:stop].to(device),
            reference_classifier,
            dataset=dataset,
        )
        for key, values in per_sample.items():
            collected.setdefault(key, []).append(values.detach().cpu())
    combined = {key: torch.cat(parts) for key, parts in collected.items()}
    real_correct = combined["real_correct"]
    pseudo_correct = combined["pseudo_correct"]
    preserved = combined["label_preserved"]
    denominator = real_correct.sum().item()
    summary = FidelitySummary(
        sample_count=len(labels),
        pixel_mse_mean=combined["pixel_mse"].mean().item(),
        pixel_mse_std=_safe_std(combined["pixel_mse"]),
        ssim_mean=combined["ssim"].mean().item(),
        ssim_std=_safe_std(combined["ssim"]),
        frozen_feature_cosine_mean=combined["frozen_feature_cosine"].mean().item(),
        frozen_feature_cosine_std=_safe_std(combined["frozen_feature_cosine"]),
        independent_real_accuracy=real_correct.float().mean().item(),
        independent_pseudo_accuracy=pseudo_correct.float().mean().item(),
        prediction_agreement_rate=combined["prediction_agreement"].float().mean().item(),
        label_preservation_rate=preserved.sum().item() / denominator if denominator else float("nan"),
    )
    return summary, combined


def append_event_samples_csv(
    path: str,
    run_id: str,
    event_index: int,
    meta_iteration: int,
    buffer_indices: Iterable[int],
    labels: torch.Tensor,
    per_sample: Dict[str, torch.Tensor],
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = [
        "run_id", "event_index", "meta_iteration", "buffer_index", "label",
        "pixel_mse", "ssim", "frozen_feature_cosine", "real_prediction",
        "pseudo_prediction", "real_correct", "pseudo_correct", "prediction_agreement",
        "label_preserved",
    ]
    write_header = not os.path.exists(path)
    cpu = {key: value.detach().cpu() for key, value in per_sample.items()}
    with open(path, "a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for row_index, (buffer_index, label) in enumerate(zip(buffer_indices, labels.detach().cpu().view(-1))):
            writer.writerow({
                "run_id": run_id,
                "event_index": event_index,
                "meta_iteration": meta_iteration,
                "buffer_index": int(buffer_index),
                "label": int(label.item()),
                "pixel_mse": float(cpu["pixel_mse"][row_index]),
                "ssim": float(cpu["ssim"][row_index]),
                "frozen_feature_cosine": float(cpu["frozen_feature_cosine"][row_index]),
                "real_prediction": int(cpu["real_prediction"][row_index]),
                "pseudo_prediction": int(cpu["pseudo_prediction"][row_index]),
                "real_correct": int(cpu["real_correct"][row_index]),
                "pseudo_correct": int(cpu["pseudo_correct"][row_index]),
                "prediction_agreement": int(cpu["prediction_agreement"][row_index]),
                "label_preserved": int(cpu["label_preserved"][row_index]),
            })


def write_summary_json(path: str, summary: FidelitySummary, extra: Optional[Dict] = None) -> None:
    payload = asdict(summary)
    if extra:
        payload.update(extra)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def write_summary_csv(path: str, summary: FidelitySummary, extra: Optional[Dict] = None) -> None:
    payload = asdict(summary)
    if extra:
        payload.update(extra)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(payload))
        writer.writeheader()
        writer.writerow(payload)

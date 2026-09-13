from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class FSSAEConfig:
    dataset: str = "omniglot"
    in_channels: int = 3
    image_size: int = 28
    encoder_channels: int = 32
    latent_dim: int = 256
    adapter_dim: int = 512
    cpn_hidden_dim: int = 2304
    normalization: str = "batch"
    num_classes: int = 1000
    sparsity_target: float = 0.05
    sparsity_weight: float = 1.0
    reconstruction_weight: float = 1.0
    fisher_reconstruction_weight: float = 0.1
    weight_decay_weight: float = 0.0
    fda_weight: float = 1.0


class FSSAE(nn.Module):
    """
    Fisher-guided stacked sparse autoencoder block used by the SRN.

    This module keeps SRN responsibilities explicit:
    - encode images into sparse latent representations
    - decode latent codes back to image space
    - compute FSSAE-specific losses from latent/reconstruction statistics
    """

    def __init__(self, config: FSSAEConfig):
        super().__init__()
        self.config = config
        self.dataset = config.dataset.lower()


        self.omniglot_mean = 0.92206 * 256.0
        self.omniglot_std = 0.08426 * 256.0 * 256.0

        hidden_channels = config.encoder_channels
        mid_channels = hidden_channels * 2
        bottleneck_channels = hidden_channels * 4

        def norm(channels: int) -> nn.Module:
            normalization = getattr(config, "normalization", "batch").lower()
            if normalization == "batch":
                return nn.BatchNorm2d(channels)
            if normalization == "group":
                groups = min(8, channels)
                while channels % groups != 0:
                    groups -= 1
                return nn.GroupNorm(groups, channels)
            raise ValueError("normalization must be 'batch' or 'group'.")

        self.encoder = nn.Sequential(
            nn.Conv2d(config.in_channels, hidden_channels, kernel_size=3, stride=1, padding=1),
            norm(hidden_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(hidden_channels, mid_channels, kernel_size=3, stride=1, padding=1),
            norm(mid_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(mid_channels, bottleneck_channels, kernel_size=3, stride=1, padding=1),
            norm(bottleneck_channels),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, config.in_channels, config.image_size, config.image_size)
            encoded = self.encoder(dummy)
            self.feature_shape = tuple(encoded.shape[1:])
            self.flattened_dim = int(encoded.numel())

        self.to_latent = nn.Linear(self.flattened_dim, config.latent_dim)
        self.from_latent = nn.Linear(config.latent_dim, self.flattened_dim)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                self.feature_shape[0],
                mid_channels,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            norm(mid_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                mid_channels,
                hidden_channels,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            norm(hidden_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                hidden_channels,
                config.in_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.Tanh(),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        return self.to_latent(features.flatten(start_dim=1))

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        features = self.from_latent(latent).view(latent.size(0), *self.feature_shape)
        reconstruction = self.decoder(features)
        if reconstruction.shape[-2:] != (self.config.image_size, self.config.image_size):
            reconstruction = F.interpolate(
                reconstruction,
                size=(self.config.image_size, self.config.image_size),
                mode="bilinear",
                align_corners=False,
            )
        if "omni" not in self.dataset:
            return reconstruction


        if reconstruction.size(1) == 3:
            grayscale = reconstruction.mean(dim=1, keepdim=True)
            reconstruction = grayscale.expand(-1, 3, -1, -1)
        unit_interval = (reconstruction + 1.0) * 0.5
        return (unit_interval * 255.0 - self.omniglot_mean) / self.omniglot_std

    def to_unit_interval(self, images: torch.Tensor) -> torch.Tensor:
        if "omni" in self.dataset:
            return ((images * self.omniglot_std + self.omniglot_mean) / 255.0).clamp(0.0, 1.0)
        return ((images + 1.0) * 0.5).clamp(0.0, 1.0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        latent = self.encode(x)
        reconstruction = self.decode(latent)
        return latent, reconstruction

    def sparse_kl_loss(self, latent: torch.Tensor) -> torch.Tensor:
        rho = self.config.sparsity_target
        rho_hat = torch.sigmoid(latent).mean(dim=0).clamp(1e-6, 1 - 1e-6)
        rho_tensor = torch.full_like(rho_hat, rho)
        kl = rho_tensor * torch.log(rho_tensor / rho_hat)
        kl = kl + (1 - rho_tensor) * torch.log((1 - rho_tensor) / (1 - rho_hat))
        return kl.sum()

    def fda_feature_weights(self, latent: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        labels = labels.view(-1)
        flat_dim = latent.size(1)
        global_mean = latent.mean(dim=0)

        between = torch.zeros(flat_dim, device=latent.device, dtype=latent.dtype)
        within = torch.zeros(flat_dim, device=latent.device, dtype=latent.dtype)

        for class_id in labels.unique(sorted=False):
            mask = labels == class_id
            class_latent = latent[mask]
            if class_latent.numel() == 0:
                continue
            class_mean = class_latent.mean(dim=0)
            between = between + class_latent.size(0) * (class_mean - global_mean).pow(2)
            within = within + (class_latent - class_mean).pow(2).sum(dim=0)

        fisher_score = between / (within + 1e-6)
        fisher_score = fisher_score.abs()
        if torch.allclose(fisher_score.sum(), torch.tensor(0.0, device=latent.device, dtype=latent.dtype)):
            return torch.ones_like(fisher_score) / fisher_score.numel()
        return fisher_score / fisher_score.sum()

    def reconstruction_loss(
        self,
        x: torch.Tensor,
        reconstruction: torch.Tensor,
        latent: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if "omni" in self.dataset:
            reconstruction_target = self.to_unit_interval(x)
            reconstruction_value = self.to_unit_interval(reconstruction)
        else:



            reconstruction_target = x
            reconstruction_value = reconstruction
        pixel_loss = (reconstruction_value - reconstruction_target).pow(2).mean(dim=(1, 2, 3))
        if labels is None:
            return pixel_loss.mean()




        feature_weights = self.fda_feature_weights(latent, labels)
        weighted_latent = latent * feature_weights.unsqueeze(0)
        weighted_reconstruction = self.decode(weighted_latent)
        weighted_reconstruction_value = (
            self.to_unit_interval(weighted_reconstruction)
            if "omni" in self.dataset
            else weighted_reconstruction
        )
        weighted_pixel_loss = (
            weighted_reconstruction_value - reconstruction_target
        ).pow(2).mean(dim=(1, 2, 3))

        reconstructed_latent = self.encode(weighted_reconstruction)
        latent_consistency = ((reconstructed_latent - latent).pow(2) * feature_weights.unsqueeze(0)).sum(dim=1)






        fisher_auxiliary = 0.5 * (weighted_pixel_loss.mean() + latent_consistency.mean())
        return pixel_loss.mean() + self.config.fisher_reconstruction_weight * fisher_auxiliary

    def regularization_loss(self) -> torch.Tensor:
        if self.config.weight_decay_weight <= 0:
            return torch.zeros((), device=self.to_latent.weight.device)
        penalty = torch.zeros((), device=self.to_latent.weight.device)
        for parameter in self.parameters():
            penalty = penalty + parameter.pow(2).sum()
        return 0.5 * penalty

    def loss_terms(
        self,
        x: torch.Tensor,
        reconstruction: torch.Tensor,
        latent: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        reconstruction_loss = self.reconstruction_loss(x, reconstruction, latent, labels)
        sparse_loss = self.sparse_kl_loss(latent)
        regularization_loss = self.regularization_loss()
        total = (
            self.config.reconstruction_weight * reconstruction_loss
            + self.config.sparsity_weight * sparse_loss
            + self.config.weight_decay_weight * regularization_loss
        )
        return {
            "reconstruction": reconstruction_loss,
            "sparse": sparse_loss,
            "regularization": regularization_loss,
            "total": total,
        }


class SRN(nn.Module):
    """
    Sparse Representation Network.

    Owns the FSSAE module and is the only place where sparse representations
    and reconstructions are produced.
    """

    def __init__(self, config: FSSAEConfig):
        super().__init__()
        self.fssae = FSSAE(config)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        latent, reconstruction = self.fssae(x)
        return {
            "latent": latent,
            "reconstruction": reconstruction,
        }

    def loss_terms(
        self,
        x: torch.Tensor,
        reconstruction: torch.Tensor,
        latent: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        return self.fssae.loss_terms(x, reconstruction, latent, labels)


class CPN(nn.Module):
    """
    Classification Prediction Network.

    This module only consumes SRN latent features and produces logits.
    """

    def __init__(self, config: FSSAEConfig):
        super().__init__()
        self.adapter = nn.Linear(config.latent_dim, config.adapter_dim)
        hidden_dim = getattr(config, "cpn_hidden_dim", 2304)
        self.hidden = nn.Linear(config.adapter_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, config.num_classes)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        latent = F.relu(self.adapter(latent), inplace=True)
        hidden = F.relu(self.hidden(latent), inplace=True)
        return self.output(hidden)

    def forward_with_weights(self, latent: torch.Tensor, weights: Iterable[torch.Tensor]) -> torch.Tensor:
        adapter_weight, adapter_bias, hidden_weight, hidden_bias, output_weight, output_bias = weights
        latent = F.linear(latent, adapter_weight, adapter_bias)
        latent = F.relu(latent, inplace=False)
        hidden = F.linear(latent, hidden_weight, hidden_bias)
        hidden = F.relu(hidden, inplace=False)
        return F.linear(hidden, output_weight, output_bias)


class FSEMLModel(nn.Module):
    """
    Clear SRN + CPN decomposition for the paper-aligned implementation.

    This is intentionally separate from the legacy AE branch so that we can
    evolve the paper implementation without destabilizing the old codepath.
    """

    def __init__(self, config: FSSAEConfig | None = None):
        super().__init__()
        self.config = config or FSSAEConfig()
        self.srn = SRN(self.config)
        self.cpn = CPN(self.config)

    def forward_srn(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.srn(x)

    def forward_cpn(
        self,
        latent: torch.Tensor,
        fast_weights: Iterable[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if fast_weights is None:
            return self.cpn(latent)
        return self.cpn.forward_with_weights(latent, fast_weights)

    def forward_features(
        self,
        x: torch.Tensor,
        fast_weights: Iterable[torch.Tensor] | None = None,
    ) -> Dict[str, torch.Tensor]:
        srn_outputs = self.forward_srn(x)
        logits = self.forward_cpn(srn_outputs["latent"], fast_weights=fast_weights)
        return {
            **srn_outputs,
            "logits": logits,
        }

    def forward(
        self,
        x: torch.Tensor,
        vars: Iterable[torch.Tensor] | None = None,
        bn_training: bool = False,
        feature: bool = False,
        return_dict: bool = False,
    ):
        del bn_training

        outputs = self.forward_features(x, fast_weights=vars)
        if return_dict or feature:
            return outputs
        return outputs["reconstruction"], outputs["logits"]

    def cpn_named_parameters(self) -> Iterable[Tuple[str, nn.Parameter]]:
        return self.cpn.named_parameters()

    def srn_named_parameters(self) -> Iterable[Tuple[str, nn.Parameter]]:
        return self.srn.named_parameters()

    def inner_loop_named_parameters(self) -> Iterable[Tuple[str, nn.Parameter]]:
        return self.cpn_named_parameters()

    def outer_loop_named_parameters(self) -> Iterable[Tuple[str, nn.Parameter]]:
        return self.named_parameters()

    def classification_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits, labels)

    def total_loss(self, x: torch.Tensor, labels: torch.Tensor) -> Dict[str, torch.Tensor]:
        outputs = self.forward(x, return_dict=True)
        srn_losses = self.srn.loss_terms(
            x=x,
            reconstruction=outputs["reconstruction"],
            latent=outputs["latent"],
            labels=labels,
        )
        classification = self.classification_loss(outputs["logits"], labels)
        total = classification + self.config.fda_weight * srn_losses["total"]
        return {
            "classification": classification,
            "reconstruction": srn_losses["reconstruction"],
            "sparse": srn_losses["sparse"],
            "regularization": srn_losses["regularization"],
            "fssae_total": srn_losses["total"],
            "total": total,
        }

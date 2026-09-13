from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class VectorFSSAEConfig:
    input_dim: int = 384
    hidden_dim: int = 256
    latent_dim: int = 128
    classifier_hidden: int = 512
    num_classes: int = 150
    sparsity_target: float = 0.05
    sparsity_weight: float = 5e-3
    reconstruction_weight: float = 0.1
    fisher_weight: float = 0.1


class VectorFSSAE(nn.Module):
    def __init__(self, config: VectorFSSAEConfig):
        super().__init__()
        self.config = config
        self.encoder = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(config.latent_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.input_dim),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def sparse_kl(self, z: torch.Tensor) -> torch.Tensor:
        rho = self.config.sparsity_target
        rho_hat = torch.sigmoid(z).mean(0).clamp(1e-6, 1 - 1e-6)
        return (rho * torch.log(rho / rho_hat) + (1 - rho) * torch.log((1 - rho) / (1 - rho_hat))).mean()

    def fisher_weights(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        global_mean = z.mean(0)
        between = torch.zeros_like(global_mean)
        within = torch.zeros_like(global_mean)
        for class_id in labels.unique():
            values = z[labels == class_id]
            mean = values.mean(0)
            between += len(values) * (mean - global_mean).pow(2)
            within += (values - mean).pow(2).sum(0)
        score = between / (within + 1e-6)
        if score.sum() <= 1e-12:
            return torch.full_like(score, 1.0 / score.numel())
        return score / score.sum()

    def loss(self, x: torch.Tensor, z: torch.Tensor, reconstructed: torch.Tensor, labels: torch.Tensor):
        mse = F.mse_loss(reconstructed, x)
        weights = self.fisher_weights(z, labels)
        weighted_reconstruction = self.decode(z * weights.unsqueeze(0))
        consistency = ((self.encode(weighted_reconstruction) - z).pow(2) * weights.unsqueeze(0)).sum(1).mean()
        fisher = 0.5 * (F.mse_loss(weighted_reconstruction, x) + consistency)
        sparse = self.sparse_kl(z)
        total = self.config.reconstruction_weight * mse + self.config.sparsity_weight * sparse
        total = total + self.config.fisher_weight * fisher
        return {"total": total, "reconstruction": mse, "sparse": sparse, "fisher": fisher}


class VectorCPN(nn.Module):
    def __init__(self, config: VectorFSSAEConfig):
        super().__init__()
        self.adapter = nn.Linear(config.latent_dim, config.hidden_dim)
        self.hidden = nn.Linear(config.hidden_dim, config.classifier_hidden)
        self.output = nn.Linear(config.classifier_hidden, config.num_classes)

    def forward(self, z: torch.Tensor, weights: Iterable[torch.Tensor] | None = None):
        if weights is None:
            return self.output(F.relu(self.hidden(F.relu(self.adapter(z)))))
        aw, ab, hw, hb, ow, ob = weights
        value = F.relu(F.linear(z, aw, ab))
        value = F.relu(F.linear(value, hw, hb))
        return F.linear(value, ow, ob)


class VectorFSEML(nn.Module):
    def __init__(self, config: VectorFSSAEConfig):
        super().__init__()
        self.config = config
        self.srn = VectorFSSAE(config)
        self.cpn = VectorCPN(config)

    def cpn_parameters(self):
        return list(self.cpn.parameters())

    @torch.no_grad()
    def reset_classifier_rows(self, class_ids: Iterable[int], optimizer=None, initialization: str = "zero"):
        rows = torch.as_tensor(sorted(set(class_ids)), dtype=torch.long, device=self.cpn.output.weight.device)
        if rows.numel() == 0:
            return
        if initialization == "zero":
            self.cpn.output.weight.index_fill_(0, rows, 0.0)
            self.cpn.output.bias.index_fill_(0, rows, 0.0)
        elif initialization == "kaiming":
            replacement = torch.empty(
                (len(rows), self.cpn.output.weight.shape[1]), device=self.cpn.output.weight.device
            )
            nn.init.kaiming_uniform_(replacement, a=5**0.5)
            self.cpn.output.weight.index_copy_(0, rows, replacement)
            self.cpn.output.bias.index_fill_(0, rows, 0.0)
        else:
            raise ValueError(f"Unknown classifier initialization: {initialization}")
        if optimizer is not None:
            for parameter in (self.cpn.output.weight, self.cpn.output.bias):
                state = optimizer.state.get(parameter, {})
                for value in state.values():
                    if torch.is_tensor(value) and value.shape == parameter.shape:
                        value.index_fill_(0, rows, 0.0)

    def forward(self, x: torch.Tensor, fast_weights=None):
        z = self.srn.encode(x)
        reconstruction = self.srn.decode(z)
        return {"latent": z, "reconstruction": reconstruction, "logits": self.cpn(z, fast_weights)}

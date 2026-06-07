from __future__ import annotations

import torch
from torch import nn


class ResidualLatentAligner(nn.Module):
    """Identity-initialized residual correction in the latent domain."""

    def __init__(self, latent_dim: int = 512) -> None:
        super().__init__()
        self.residual = nn.Linear(latent_dim, latent_dim)
        nn.init.zeros_(self.residual.weight)
        nn.init.zeros_(self.residual.bias)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return latent + self.residual(latent)


class ClassConditionalBias(nn.Module):
    def __init__(self, num_classes: int = 10, latent_dim: int = 512) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_classes, latent_dim)
        nn.init.zeros_(self.embedding.weight)

    def forward(self, labels: torch.Tensor) -> torch.Tensor:
        return self.embedding(labels)


class ULDAAdaptor(nn.Module):
    """Trainable target-domain modules; the UVAE backbone remains frozen."""

    def __init__(self, latent_dim: int = 512, num_classes: int = 10) -> None:
        super().__init__()
        self.aligner = ResidualLatentAligner(latent_dim)
        self.class_bias = ClassConditionalBias(num_classes, latent_dim)

    def forward(self, latent: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return self.aligner(latent) + self.class_bias(labels)


@torch.no_grad()
def compute_prototypes(
    backbone: nn.Module,
    loader: torch.utils.data.DataLoader,
    num_classes: int,
    latent_dim: int,
    device: torch.device,
) -> torch.Tensor:
    sums = torch.zeros(num_classes, latent_dim, device=device)
    counts = torch.zeros(num_classes, device=device)
    for batch in loader:
        speckles = batch["speckle"].to(device)
        labels = batch["label"].to(device)
        latent_mean, _ = backbone.encode(speckles)
        sums.index_add_(0, labels, latent_mean)
        counts.index_add_(0, labels, torch.ones_like(labels, dtype=torch.float32))
    return sums / counts.clamp_min(1.0).unsqueeze(1)

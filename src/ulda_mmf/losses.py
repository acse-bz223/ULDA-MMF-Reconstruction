from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


def heteroscedastic_gaussian_nll(
    mean: torch.Tensor, logvar: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    logvar = logvar.clamp(-8.0, 4.0)
    return 0.5 * (torch.exp(-logvar) * (target - mean).square() + logvar).mean()


def latent_kl(mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return -0.5 * (1.0 + logvar - mean.square() - logvar.exp()).mean()


def uvae_objective(
    output: dict[str, torch.Tensor],
    target: torch.Tensor,
    labels: torch.Tensor,
    beta: float = 1.0e-4,
    classification_weight: float = 1.0,
) -> dict[str, torch.Tensor]:
    reconstruction = heteroscedastic_gaussian_nll(
        output["reconstruction_mean"], output["reconstruction_logvar"], target
    )
    kl = latent_kl(output["latent_mean"], output["latent_logvar"])
    classification = F.cross_entropy(output["logits"], labels)
    total = reconstruction + beta * kl + classification_weight * classification
    return {
        "total": total,
        "reconstruction": reconstruction,
        "kl": kl,
        "classification": classification,
    }


def batch_covariance(latent: torch.Tensor) -> torch.Tensor:
    centered = latent - latent.mean(dim=0, keepdim=True)
    denominator = max(latent.shape[0] - 1, 1)
    covariance = centered.T @ centered / denominator
    return 0.5 * (covariance + covariance.T)


def orthogonality_regularization(
    residual_weight: torch.Tensor, residual: bool = True
) -> torch.Tensor:
    identity = torch.eye(
        residual_weight.shape[0],
        device=residual_weight.device,
        dtype=residual_weight.dtype,
    )
    effective_weight = identity + residual_weight if residual else residual_weight
    return (effective_weight.T @ effective_weight - identity).square().mean()


def symmetric_gaussian_kl(
    mean_a: torch.Tensor,
    logvar_a: torch.Tensor,
    mean_b: torch.Tensor,
    logvar_b: torch.Tensor,
) -> torch.Tensor:
    logvar_a = logvar_a.clamp(-8.0, 4.0)
    logvar_b = logvar_b.clamp(-8.0, 4.0)
    variance_a = logvar_a.exp()
    variance_b = logvar_b.exp()
    squared_delta = (mean_a - mean_b).square()
    kl_ab = 0.5 * (
        logvar_b - logvar_a + (variance_a + squared_delta) / variance_b - 1.0
    )
    kl_ba = 0.5 * (
        logvar_a - logvar_b + (variance_b + squared_delta) / variance_a - 1.0
    )
    return 0.5 * (kl_ab.mean() + kl_ba.mean())


@dataclass(frozen=True)
class ULDALossWeights:
    prototype: float = 1.0
    covariance: float = 1.0
    classification: float = 1.0
    uncertainty: float = 1.0


def ulda_objective(
    source_latent: torch.Tensor,
    aligned_target_latent: torch.Tensor,
    target_logits: torch.Tensor,
    target_labels: torch.Tensor,
    prototypes: torch.Tensor,
    source_prediction: tuple[torch.Tensor, torch.Tensor],
    target_prediction: tuple[torch.Tensor, torch.Tensor],
    weights: ULDALossWeights,
) -> dict[str, torch.Tensor]:
    prototype = F.mse_loss(aligned_target_latent, prototypes[target_labels])
    covariance = F.mse_loss(
        batch_covariance(aligned_target_latent), batch_covariance(source_latent)
    )
    classification = F.cross_entropy(target_logits, target_labels)
    uncertainty = symmetric_gaussian_kl(
        source_prediction[0],
        source_prediction[1],
        target_prediction[0],
        target_prediction[1],
    )
    total = (
        weights.prototype * prototype
        + weights.covariance * covariance
        + weights.classification * classification
        + weights.uncertainty * uncertainty
    )
    return {
        "total": total,
        "prototype": prototype,
        "covariance": covariance,
        "classification": classification,
        "uncertainty": uncertainty,
    }

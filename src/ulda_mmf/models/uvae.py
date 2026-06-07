from __future__ import annotations

import torch
from torch import nn


def _encoder_block(in_channels: int, out_channels: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Dropout2d(dropout),
        nn.MaxPool2d(2),
    )


def _decoder_block(in_channels: int, out_channels: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False
        ),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Dropout2d(dropout),
    )


class UncertaintyVAE(nn.Module):
    """Convolutional UVAE with a heteroscedastic image decoder."""

    def __init__(
        self,
        image_size: int = 256,
        latent_dim: int = 512,
        base_channels: int = 32,
        encoder_dropout: float = 0.2,
        decoder_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if image_size % 16:
            raise ValueError("image_size must be divisible by 16")
        self.image_size = image_size
        self.latent_dim = latent_dim
        self.enc1 = _encoder_block(1, base_channels, encoder_dropout)
        self.enc2 = _encoder_block(base_channels, base_channels * 2, encoder_dropout)
        self.enc3 = _encoder_block(base_channels * 2, base_channels * 4, encoder_dropout)
        self.enc4 = _encoder_block(base_channels * 4, base_channels * 8, encoder_dropout)
        feature_size = image_size // 16
        self.enc_shape = (base_channels * 8, feature_size, feature_size)
        flat_dim = int(torch.tensor(self.enc_shape).prod().item())
        self.flat_dim = flat_dim
        self.fc_mu = nn.Linear(flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(flat_dim, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, flat_dim)
        self.dec1 = _decoder_block(base_channels * 8, base_channels * 4, decoder_dropout)
        self.dec2 = _decoder_block(base_channels * 4, base_channels * 2, decoder_dropout)
        self.dec3 = _decoder_block(base_channels * 2, base_channels, decoder_dropout)
        final_channels = max(base_channels // 2, 4)
        self.dec4 = _decoder_block(base_channels, final_channels, decoder_dropout)
        self.out_mu = nn.Conv2d(final_channels, 1, kernel_size=3, padding=1)
        self.out_logvar = nn.Conv2d(final_channels, 1, kernel_size=3, padding=1)

    def _enc(self, speckle: torch.Tensor) -> torch.Tensor:
        speckle = self.enc1(speckle)
        speckle = self.enc2(speckle)
        speckle = self.enc3(speckle)
        return self.enc4(speckle)

    def encode(self, speckle: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self._enc(speckle).flatten(1)
        return self.fc_mu(features), self.fc_logvar(features)

    @staticmethod
    def reparameterize(mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return mean + torch.randn_like(mean) * torch.exp(0.5 * logvar)

    def decode(self, latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.fc_dec(latent).view(-1, *self.enc_shape)
        features = self.dec1(features)
        features = self.dec2(features)
        features = self.dec3(features)
        features = self.dec4(features)
        mean = torch.sigmoid(self.out_mu(features))
        logvar = self.out_logvar(features).clamp(-8.0, 4.0)
        return mean, logvar

    def forward(self, speckle: torch.Tensor) -> dict[str, torch.Tensor]:
        latent_mean, latent_logvar = self.encode(speckle)
        latent = self.reparameterize(latent_mean, latent_logvar)
        reconstruction_mean, reconstruction_logvar = self.decode(latent)
        return {
            "reconstruction_mean": reconstruction_mean,
            "reconstruction_logvar": reconstruction_logvar,
            "latent_mean": latent_mean,
            "latent_logvar": latent_logvar,
        }


class LatentDigitHead(nn.Module):
    """Semantic head used to organize the calibrated latent manifold."""

    def __init__(
        self, latent_dim: int = 512, num_classes: int = 10, dropout: float = 0.2
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(latent_dim, num_classes),
        )

    def forward(self, latent_mean: torch.Tensor) -> torch.Tensor:
        return self.net(latent_mean)

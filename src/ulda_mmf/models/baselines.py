from __future__ import annotations

import torch
from torch import nn


class MLPReconstructor(nn.Module):
    """Fully connected baseline reported in Supporting Information Table S3."""

    def __init__(
        self, image_size: int = 256, hidden_dim: int = 4096, dropout: float = 0.2
    ) -> None:
        super().__init__()
        pixels = image_size * image_size
        self.image_size = image_size
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(pixels, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, pixels),
            nn.Sigmoid(),
        )

    def forward(self, speckle: torch.Tensor) -> torch.Tensor:
        return self.network(speckle).view(-1, 1, self.image_size, self.image_size)


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.network(tensor)


class UNet(nn.Module):
    """Four-level deterministic U-Net baseline."""

    def __init__(self, base_channels: int = 32) -> None:
        super().__init__()
        channels = [base_channels * 2**index for index in range(5)]
        self.down1 = DoubleConv(1, channels[0])
        self.down2 = DoubleConv(channels[0], channels[1])
        self.down3 = DoubleConv(channels[1], channels[2])
        self.down4 = DoubleConv(channels[2], channels[3])
        self.bottleneck = DoubleConv(channels[3], channels[4])
        self.pool = nn.MaxPool2d(2)
        self.up4 = nn.ConvTranspose2d(channels[4], channels[3], 2, 2)
        self.decode4 = DoubleConv(channels[4], channels[3])
        self.up3 = nn.ConvTranspose2d(channels[3], channels[2], 2, 2)
        self.decode3 = DoubleConv(channels[3], channels[2])
        self.up2 = nn.ConvTranspose2d(channels[2], channels[1], 2, 2)
        self.decode2 = DoubleConv(channels[2], channels[1])
        self.up1 = nn.ConvTranspose2d(channels[1], channels[0], 2, 2)
        self.decode1 = DoubleConv(channels[1], channels[0])
        self.output = nn.Sequential(nn.Conv2d(channels[0], 1, 1), nn.Sigmoid())

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        down1 = self.down1(tensor)
        down2 = self.down2(self.pool(down1))
        down3 = self.down3(self.pool(down2))
        down4 = self.down4(self.pool(down3))
        bottleneck = self.bottleneck(self.pool(down4))
        up4 = self.decode4(torch.cat((self.up4(bottleneck), down4), dim=1))
        up3 = self.decode3(torch.cat((self.up3(up4), down3), dim=1))
        up2 = self.decode2(torch.cat((self.up2(up3), down2), dim=1))
        up1 = self.decode1(torch.cat((self.up1(up2), down1), dim=1))
        return self.output(up1)

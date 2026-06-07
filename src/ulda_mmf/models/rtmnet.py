from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch.fft import irfft, rfft


class RTPad(nn.Module):
    """Mixed zero/reflection padding used by the rotational-memory baseline."""

    def __init__(self, width: int, zero_pad_width: bool = False) -> None:
        super().__init__()
        self.width = width
        self.zero_pad_width = zero_pad_width

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.zero_pad_width:
            tensor = F.pad(tensor, (self.width, self.width, 0, 0))
        top = torch.flip(tensor[:, :, -self.width :, :], dims=[3])
        bottom = torch.flip(tensor[:, :, : self.width, :], dims=[3])
        return torch.cat((top, tensor, bottom), dim=2)


class DoubleConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
        rotational_padding: bool = False,
    ) -> None:
        super().__init__()
        if rotational_padding:
            first = nn.Sequential(
                RTPad(padding, zero_pad_width=True),
                nn.Conv2d(in_channels, out_channels, kernel_size),
            )
            second = nn.Sequential(
                RTPad(1, zero_pad_width=True),
                nn.Conv2d(out_channels, out_channels, 3),
            )
        else:
            first = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
            second = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.network = nn.Sequential(
            first,
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            second,
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.network(tensor)


class FourierFilter1D(nn.Module):
    def __init__(self, height: int, width: int, channels: int) -> None:
        super().__init__()
        self.pad = width // 2
        self.complex_weight = nn.Parameter(
            torch.randn(channels, height, width + 1, 2) / channels
        )

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        padded = F.pad(tensor, (self.pad, self.pad, 0, 0))
        spectrum = rfft(padded, dim=-1, norm="ortho")
        weight = torch.view_as_complex(self.complex_weight.contiguous())
        filtered = irfft(spectrum * weight, dim=-1, norm="ortho")
        return filtered[:, :, :, self.pad : 3 * self.pad]


class SpatialFourierBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        height: int,
        width: int,
        rotational_padding: bool,
    ) -> None:
        super().__init__()
        half = out_channels // 2
        self.fourier_input = nn.Conv2d(in_channels, half, 3, padding=1)
        self.fourier = FourierFilter1D(height, width, half)
        self.spatial = DoubleConv(
            in_channels, half, rotational_padding=rotational_padding
        )

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            (self.spatial(tensor), self.fourier(self.fourier_input(tensor))), dim=1
        )


class RTMNet(nn.Module):
    """Rotational-memory-effect baseline extracted from the project notebooks."""

    def __init__(
        self, image_size: int = 256, base_channels: int = 16, rotational_padding: bool = True
    ) -> None:
        super().__init__()
        if image_size % 8:
            raise ValueError("image_size must be divisible by 8")
        dim = base_channels
        self.conv0 = SpatialFourierBlock(
            1, dim, image_size, image_size, rotational_padding
        )
        self.conv1 = SpatialFourierBlock(
            dim, dim * 2, image_size // 2, image_size // 2, rotational_padding
        )
        self.conv2 = SpatialFourierBlock(
            dim * 2, dim * 4, image_size // 4, image_size // 4, rotational_padding
        )
        self.conv3 = SpatialFourierBlock(
            dim * 4, dim * 4, image_size // 8, image_size // 8, rotational_padding
        )
        self.pool = nn.MaxPool2d(2)
        self.up3 = nn.ConvTranspose2d(dim * 4, dim * 4, 2, 2)
        self.decode3 = DoubleConv(
            dim * 8, dim * 2, rotational_padding=rotational_padding
        )
        self.up2 = nn.ConvTranspose2d(dim * 2, dim * 2, 2, 2)
        self.decode2 = DoubleConv(
            dim * 4, dim, rotational_padding=rotational_padding
        )
        self.up1 = nn.ConvTranspose2d(dim, dim, 2, 2)
        self.decode1 = DoubleConv(
            dim * 2, dim // 2, rotational_padding=rotational_padding
        )
        self.output = nn.Sequential(
            DoubleConv(dim // 2, 1, rotational_padding=rotational_padding),
            nn.Sigmoid(),
        )

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        level0 = self.conv0(tensor)
        level1 = self.conv1(self.pool(level0))
        level2 = self.conv2(self.pool(level1))
        level3 = self.conv3(self.pool(level2))
        decoded = self.decode3(torch.cat((level2, self.up3(level3)), dim=1))
        decoded = self.decode2(torch.cat((level1, self.up2(decoded)), dim=1))
        decoded = self.decode1(torch.cat((level0, self.up1(decoded)), dim=1))
        return self.output(decoded)

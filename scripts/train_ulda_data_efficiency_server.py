#!/usr/bin/env python3
"""
Self-contained ULDA data-efficiency experiments for a training server.

This script runs two independent, leakage-controlled experiments:

1. Epoch trajectory:
   Train a fresh ULDA aligner for every target bending state using the full
   target training split. Evaluate the same held-out test split at selected
   adaptation epochs.

2. Target-data sweep:
   Train a fresh ULDA aligner for every target bending state and every nested
   target-training fraction. Evaluate each run on the same held-out test split.

Target-domain ground-truth images are never read during adaptation. They are
read only by evaluate() on held-out test indices. Results are written as CSV
and Markdown tables and printed to stdout at the end.

Expected historical server data layout:

  DATA_ROOT/
    bending0_sorted/
      speckles_sorted.npy
      images_sorted.npy
      labels_sorted.npy
    bending1_sorted/
      ...

Here labels_sorted.npy contains stable pattern IDs. Digit classes are inferred
with --per_digit_hint, matching the original server training scripts.

Typical server command:

  python train_ulda_data_efficiency_server.py \
    --data_root /path/to/data \
    --vae_ckpt /path/to/uvae_checkpoint.pt \
    --target_domains 1-10 --experiments epoch,data --amp

The output directory contains raw results, mean-across-bending tables, detailed
per-bending tables, fixed split indices, and a combined results_tables.md.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


METRIC_KEYS = ("MSE", "PSNR", "SSIM", "Accuracy")
RESULT_FIELDS = (
    "experiment",
    "bending",
    "epoch",
    "fraction",
    "train_samples",
    "test_samples",
    *METRIC_KEYS,
)


def log(message: str) -> None:
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{stamp}] {message}", flush=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def autocast_context(device: torch.device, enabled: bool):
    if not enabled or device.type != "cuda":
        return nullcontext()
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast("cuda", enabled=True)
    return torch.cuda.amp.autocast(enabled=True)


def make_grad_scaler(device: torch.device, enabled: bool):
    return torch.cuda.amp.GradScaler(enabled=bool(enabled and device.type == "cuda"))


def parse_int_spec(spec: str) -> List[int]:
    values: List[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_text, end_text = part.split("-", 1)
            start, end = int(start_text), int(end_text)
            step = 1 if end >= start else -1
            values.extend(range(start, end + step, step))
        else:
            values.append(int(part))
    return sorted(set(values))


def parse_float_spec(spec: str) -> List[float]:
    values = sorted(set(float(part.strip()) for part in spec.split(",") if part.strip()))
    if not values or any(value <= 0.0 or value > 1.0 for value in values):
        raise argparse.ArgumentTypeError("fractions must be in the interval (0, 1]")
    return values


def label_to_digit(labels: np.ndarray, per_digit_hint: int) -> np.ndarray:
    zero_based = labels.astype(np.int64) - 1
    digits = (zero_based % (per_digit_hint * 10)) // per_digit_hint
    if digits.size and (digits.min() < 0 or digits.max() > 9):
        raise ValueError("Could not infer digit classes from labels_sorted.npy")
    return digits.astype(np.int64)


def load_bending_arrays(root: Path, bending: int):
    directory = root / f"bending{bending}_sorted"
    required = {
        "speckles": directory / "speckles_sorted.npy",
        "images": directory / "images_sorted.npy",
        "labels": directory / "labels_sorted.npy",
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required arrays:\n" + "\n".join(missing))
    arrays = tuple(np.load(path, mmap_mode="r") for path in required.values())
    lengths = {len(array) for array in arrays}
    if len(lengths) != 1:
        raise ValueError(f"Array lengths do not match in {directory}")
    return arrays


def make_label_to_index(labels: np.ndarray) -> Dict[int, int]:
    mapping: Dict[int, int] = {}
    for index, label in enumerate(labels.tolist()):
        mapping.setdefault(int(label), index)
    return mapping


def stratified_train_test_split(
    labels: np.ndarray,
    per_digit_hint: int,
    test_ratio: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if not 0.0 < test_ratio < 1.0:
        raise ValueError("--test_ratio must be between 0 and 1")
    digits = label_to_digit(labels, per_digit_hint)
    rng = np.random.default_rng(seed)
    train_parts: List[np.ndarray] = []
    test_parts: List[np.ndarray] = []
    for digit in range(10):
        indices = np.where(digits == digit)[0].astype(np.int64)
        if len(indices) < 2:
            raise ValueError(f"Digit {digit} needs at least two target samples")
        rng.shuffle(indices)
        test_count = max(1, int(round(test_ratio * len(indices))))
        test_count = min(test_count, len(indices) - 1)
        test_parts.append(indices[:test_count])
        train_parts.append(indices[test_count:])
    train_indices = np.concatenate(train_parts)
    test_indices = np.concatenate(test_parts)
    rng.shuffle(train_indices)
    rng.shuffle(test_indices)
    return train_indices, test_indices


def build_nested_class_orders(
    train_indices: np.ndarray,
    labels: np.ndarray,
    per_digit_hint: int,
    seed: int,
) -> Dict[int, np.ndarray]:
    digits = label_to_digit(labels[train_indices], per_digit_hint)
    rng = np.random.default_rng(seed)
    orders: Dict[int, np.ndarray] = {}
    for digit in range(10):
        values = train_indices[digits == digit].copy()
        rng.shuffle(values)
        orders[digit] = values
    return orders


def nested_subset(
    class_orders: Dict[int, np.ndarray],
    fraction: float,
    min_per_class: int,
    seed: int,
) -> np.ndarray:
    pieces: List[np.ndarray] = []
    for digit in range(10):
        values = class_orders[digit]
        count = max(min_per_class, int(math.floor(fraction * len(values))))
        pieces.append(values[: min(count, len(values))])
    subset = np.concatenate(pieces).astype(np.int64)
    np.random.default_rng(seed).shuffle(subset)
    return subset


def to_speckle_tensor(batch: np.ndarray, size: int, device: torch.device) -> torch.Tensor:
    tensor = torch.from_numpy(batch.astype(np.float32, copy=True))
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0).unsqueeze(0)
    elif tensor.ndim == 3:
        tensor = tensor.unsqueeze(1)
    elif tensor.ndim != 4:
        raise ValueError(f"Unexpected speckle dimensions: {tensor.shape}")
    flat = tensor.view(tensor.shape[0], -1)
    minimum = flat.min(dim=1, keepdim=True).values
    maximum = flat.max(dim=1, keepdim=True).values
    tensor = ((flat - minimum) / (maximum - minimum + 1e-8)).view(
        tensor.shape[0], 1, tensor.shape[-2], tensor.shape[-1]
    )
    if tensor.shape[-2:] != (size, size):
        tensor = F.interpolate(tensor, size=(size, size), mode="bilinear", align_corners=False)
    return tensor.to(device, non_blocking=True)


def to_image_tensor(batch: np.ndarray, size: int, device: torch.device) -> torch.Tensor:
    source_dtype = batch.dtype
    tensor = torch.from_numpy(batch.astype(np.float32, copy=True))
    if np.issubdtype(source_dtype, np.integer):
        tensor = tensor / float(np.iinfo(source_dtype).max)
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0).unsqueeze(0)
    elif tensor.ndim == 3:
        tensor = tensor.unsqueeze(1)
    elif tensor.ndim != 4:
        raise ValueError(f"Unexpected image dimensions: {tensor.shape}")
    if tensor.shape[-2:] != (size, size):
        tensor = F.interpolate(tensor, size=(size, size), mode="bilinear", align_corners=False)
    return tensor.to(device, non_blocking=True)


def conv_block(in_channels: int, out_channels: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Dropout2d(dropout),
        nn.MaxPool2d(2, 2),
    )


def deconv_block(in_channels: int, out_channels: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Dropout2d(dropout),
    )


class SimpleVAEWithUncertainty(nn.Module):
    def __init__(self, image_size: int = 256, latent_dim: int = 512):
        super().__init__()
        if image_size != 256:
            raise ValueError("The server UVAE architecture expects 256 x 256 inputs")
        self.enc1 = conv_block(1, 32, 0.2)
        self.enc2 = conv_block(32, 64, 0.2)
        self.enc3 = conv_block(64, 128, 0.2)
        self.enc4 = conv_block(128, 256, 0.2)
        with torch.no_grad():
            encoded = self._encode_features(torch.zeros(1, 1, image_size, image_size))
        self.enc_shape = encoded.shape[1:]
        self.flat_dim = int(encoded.flatten(1).shape[1])
        self.fc_mu = nn.Linear(self.flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flat_dim, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, self.flat_dim)
        self.dec1 = deconv_block(256, 128, 0.1)
        self.dec2 = deconv_block(128, 64, 0.1)
        self.dec3 = deconv_block(64, 32, 0.1)
        self.dec4 = deconv_block(32, 16, 0.1)
        self.out_mu = nn.Conv2d(16, 1, 3, 1, 1)
        self.out_logvar = nn.Conv2d(16, 1, 3, 1, 1)

    def _encode_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.enc4(self.enc3(self.enc2(self.enc1(x))))

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self._encode_features(x).flatten(1)
        return self.fc_mu(features), self.fc_logvar(features)

    def decode(
        self, z: torch.Tensor, out_hw: Optional[Tuple[int, int]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.fc_dec(z).view(-1, *self.enc_shape)
        features = self.dec4(self.dec3(self.dec2(self.dec1(features))))
        mean = torch.sigmoid(self.out_mu(features))
        logvar = self.out_logvar(features)
        if out_hw is not None and mean.shape[-2:] != out_hw:
            mean = F.interpolate(mean, size=out_hw, mode="bilinear", align_corners=False)
            logvar = F.interpolate(logvar, size=out_hw, mode="bilinear", align_corners=False)
        return mean, logvar


class LatentDigitHead(nn.Module):
    def __init__(self, latent_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(latent_dim, 10),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.net(latent)


class AlignLayer(nn.Module):
    def __init__(self, latent_dim: int = 512):
        super().__init__()
        self.fc = nn.Linear(latent_dim, latent_dim, bias=True)
        nn.init.zeros_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return latent + self.fc(latent)


class ClassConditionalBias(nn.Module):
    def __init__(self, latent_dim: int = 512):
        super().__init__()
        self.bias = nn.Embedding(10, latent_dim)
        nn.init.zeros_(self.bias.weight)

    def forward(self, latent: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return latent + self.bias(labels)


def strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {
        key[7:] if key.startswith("module.") else key: value
        for key, value in state_dict.items()
    }


def load_frozen_backbone(
    checkpoint_path: Path,
    image_size: int,
    latent_dim: int,
    device: torch.device,
) -> Tuple[SimpleVAEWithUncertainty, LatentDigitHead]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint must be a dictionary containing VAE and head states")
    vae_state = checkpoint.get("vae", checkpoint)
    head_state = checkpoint.get("head")
    if not isinstance(head_state, dict) or not head_state:
        raise ValueError(
            "Checkpoint does not contain a trained 'head' state; accuracy would be invalid"
        )
    vae = SimpleVAEWithUncertainty(image_size=image_size, latent_dim=latent_dim)
    head = LatentDigitHead(latent_dim=latent_dim)
    missing_vae, unexpected_vae = vae.load_state_dict(strip_module_prefix(vae_state), strict=False)
    missing_head, unexpected_head = head.load_state_dict(strip_module_prefix(head_state), strict=False)
    if missing_vae or unexpected_vae or missing_head or unexpected_head:
        raise ValueError(
            "Checkpoint architecture mismatch:\n"
            f"VAE missing={missing_vae}, unexpected={unexpected_vae}\n"
            f"Head missing={missing_head}, unexpected={unexpected_head}"
        )
    vae = vae.to(device).eval()
    head = head.to(device).eval()
    for parameter in list(vae.parameters()) + list(head.parameters()):
        parameter.requires_grad_(False)
    return vae, head


def covariance_from_moments(
    count: int, total: torch.Tensor, outer_total: torch.Tensor
) -> torch.Tensor:
    if count < 2:
        return torch.zeros_like(outer_total)
    mean = total / count
    covariance = (outer_total - count * torch.outer(mean, mean)) / (count - 1)
    return (covariance + covariance.T) / 2.0


@torch.no_grad()
def compute_source_anchors(
    vae: SimpleVAEWithUncertainty,
    speckles: np.ndarray,
    labels: np.ndarray,
    per_digit_hint: int,
    image_size: int,
    batch_size: int,
    device: torch.device,
    amp: bool,
    classwise: bool,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[List[torch.Tensor]]]:
    latent_dim = vae.fc_mu.out_features
    global_sum = torch.zeros(latent_dim, device=device)
    global_outer = torch.zeros(latent_dim, latent_dim, device=device)
    class_sum = torch.zeros(10, latent_dim, device=device)
    class_outer = torch.zeros(10, latent_dim, latent_dim, device=device)
    class_count = torch.zeros(10, dtype=torch.long, device=device)
    total_count = 0
    for start in range(0, len(labels), batch_size):
        stop = min(start + batch_size, len(labels))
        inputs = to_speckle_tensor(speckles[start:stop], image_size, device)
        with autocast_context(device, amp):
            latent = vae.encode(inputs)[0].float()
        digits = torch.from_numpy(label_to_digit(labels[start:stop], per_digit_hint)).to(device)
        global_sum += latent.sum(0)
        global_outer += latent.T @ latent
        total_count += latent.shape[0]
        for digit in range(10):
            mask = digits == digit
            if mask.any():
                selected = latent[mask]
                class_sum[digit] += selected.sum(0)
                class_outer[digit] += selected.T @ selected
                class_count[digit] += selected.shape[0]
    prototypes = class_sum / class_count.clamp_min(1).unsqueeze(1)
    global_covariance = covariance_from_moments(total_count, global_sum, global_outer)
    class_covariances = None
    if classwise:
        class_covariances = [
            covariance_from_moments(
                int(class_count[digit]), class_sum[digit], class_outer[digit]
            )
            for digit in range(10)
        ]
    return prototypes.detach(), global_covariance.detach(), class_covariances


def batch_covariance(latent: torch.Tensor) -> torch.Tensor:
    centered = latent - latent.mean(0, keepdim=True)
    if latent.shape[0] < 2:
        return torch.zeros(latent.shape[1], latent.shape[1], device=latent.device)
    covariance = centered.T @ centered / (latent.shape[0] - 1)
    return (covariance + covariance.T) / 2.0


def coral_loss(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    dimension = source.shape[0]
    return (source - target).pow(2).sum() / (4.0 * dimension * dimension)


def orthogonality_loss(aligner: AlignLayer, weight: float) -> torch.Tensor:
    residual = aligner.fc.weight
    identity = torch.eye(residual.shape[0], device=residual.device, dtype=residual.dtype)
    effective = identity + residual
    return weight * (effective.T @ effective - identity).pow(2).mean()


def symmetric_gaussian_kl(
    mean_a: torch.Tensor,
    logvar_a: torch.Tensor,
    mean_b: torch.Tensor,
    logvar_b: torch.Tensor,
) -> torch.Tensor:
    logvar_a = logvar_a.clamp(-6, 3)
    logvar_b = logvar_b.clamp(-6, 3)
    var_a, var_b = logvar_a.exp(), logvar_b.exp()
    delta2 = (mean_a - mean_b).pow(2)
    kl_ab = 0.5 * ((var_a + delta2) / var_b - 1.0 + logvar_b - logvar_a)
    kl_ba = 0.5 * ((var_b + delta2) / var_a - 1.0 + logvar_a - logvar_b)
    return (0.5 * (kl_ab + kl_ba)).flatten(1).mean(1).mean()


def supervised_contrastive_loss(latent: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    latent = F.normalize(latent, dim=1)
    logits = latent @ latent.T / 0.07
    count = latent.shape[0]
    same = labels[:, None].eq(labels[None, :]).float()
    eye = torch.eye(count, device=latent.device)
    logits = logits - eye * 1e9
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    positives = same - eye
    return (-(positives * log_prob).sum(1) / positives.sum(1).clamp_min(1.0)).mean()


def ssim_per_image(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    radius = 11
    padding = radius // 2
    coordinates = torch.arange(radius, device=prediction.device, dtype=prediction.dtype) - padding
    gaussian = torch.exp(-(coordinates**2) / (2 * 1.5**2))
    gaussian = (gaussian / gaussian.sum()).view(1, 1, -1, 1)
    kernel = gaussian @ gaussian.transpose(-2, -1)
    pred_pad = F.pad(prediction, (padding,) * 4, "reflect")
    target_pad = F.pad(target, (padding,) * 4, "reflect")
    pred_mean = F.conv2d(pred_pad, kernel)
    target_mean = F.conv2d(target_pad, kernel)
    pred_var = F.conv2d(pred_pad * pred_pad, kernel) - pred_mean * pred_mean
    target_var = F.conv2d(target_pad * target_pad, kernel) - target_mean * target_mean
    covariance = F.conv2d(pred_pad * target_pad, kernel) - pred_mean * target_mean
    numerator = (2 * pred_mean * target_mean + 0.01**2) * (2 * covariance + 0.03**2)
    denominator = (pred_mean.square() + target_mean.square() + 0.01**2) * (
        pred_var + target_var + 0.03**2
    )
    return (numerator / denominator).flatten(1).mean(1)


@torch.no_grad()
def evaluate(
    vae: SimpleVAEWithUncertainty,
    head: LatentDigitHead,
    aligner: AlignLayer,
    class_bias: Optional[ClassConditionalBias],
    speckles: np.ndarray,
    images: np.ndarray,
    labels: np.ndarray,
    indices: np.ndarray,
    per_digit_hint: int,
    image_size: int,
    batch_size: int,
    device: torch.device,
    amp: bool,
) -> Dict[str, float]:
    aligner.eval()
    if class_bias is not None:
        class_bias.eval()
    totals = dict(MSE=0.0, PSNR=0.0, SSIM=0.0, Accuracy=0.0)
    for start in range(0, len(indices), batch_size):
        stop = min(start + batch_size, len(indices))
        selected = indices[start:stop]
        inputs = to_speckle_tensor(speckles[selected], image_size, device)
        targets = to_image_tensor(images[selected], image_size, device).float()
        digits_np = label_to_digit(labels[selected], per_digit_hint)
        digits = torch.from_numpy(digits_np).to(device)
        with autocast_context(device, amp):
            latent = aligner(vae.encode(inputs)[0].float())
            if class_bias is not None:
                latent = class_bias(latent, digits)
            reconstruction = vae.decode(latent, out_hw=(image_size, image_size))[0]
            logits = head(latent)
        reconstruction = reconstruction.clamp(0, 1).float()
        mse = (reconstruction - targets).pow(2).flatten(1).mean(1)
        totals["MSE"] += float(mse.sum())
        totals["PSNR"] += float((10.0 * torch.log10(1.0 / mse.clamp_min(1e-12))).sum())
        totals["SSIM"] += float(ssim_per_image(reconstruction, targets).sum())
        totals["Accuracy"] += float((logits.argmax(1).cpu().numpy() == digits_np).sum())
    count = max(1, len(indices))
    totals["Accuracy"] = 100.0 * totals["Accuracy"] / count
    for key in ("MSE", "PSNR", "SSIM"):
        totals[key] /= count
    return totals


def train_one_epoch(
    *,
    vae: SimpleVAEWithUncertainty,
    head: LatentDigitHead,
    aligner: AlignLayer,
    class_bias: Optional[ClassConditionalBias],
    optimizer: torch.optim.Optimizer,
    scaler,
    target_speckles: np.ndarray,
    target_labels: np.ndarray,
    train_indices: np.ndarray,
    source_speckles: np.ndarray,
    source_label_to_index: Dict[int, int],
    source_prototypes: torch.Tensor,
    source_covariance: torch.Tensor,
    source_class_covariances: Optional[List[torch.Tensor]],
    args: argparse.Namespace,
    device: torch.device,
    epoch: int,
    total_epochs: int,
    run_seed: int,
) -> float:
    aligner.train()
    if class_bias is not None:
        class_bias.train()
    target_batch_size = args.batch_size
    permutation = np.random.default_rng(run_seed + epoch * 1009).permutation(len(train_indices))
    natural_steps = max(1, math.ceil(len(train_indices) / target_batch_size))
    steps = args.fixed_steps_per_epoch if args.fixed_steps_per_epoch > 0 else natural_steps
    schedule = epoch / max(1, total_epochs)
    coral_weight = args.w_coral * schedule
    uncertainty_weight = args.w_uncertainty * schedule
    total_loss = 0.0
    source_rng = np.random.default_rng(run_seed + epoch * 2027)

    for step in range(steps):
        offset = (step * target_batch_size) % len(permutation)
        positions = permutation[offset : offset + target_batch_size]
        if len(positions) < target_batch_size and steps > natural_steps:
            extra = source_rng.choice(len(train_indices), target_batch_size - len(positions), replace=True)
            positions = np.concatenate([positions, extra])
        selected = train_indices[positions]
        inputs = to_speckle_tensor(target_speckles[selected], args.image_size, device)
        digits_np = label_to_digit(target_labels[selected], args.per_digit_hint)
        digits = torch.from_numpy(digits_np).to(device)
        pattern_ids = target_labels[selected].astype(np.int64)

        with autocast_context(device, args.amp):
            target_latent = aligner(vae.encode(inputs)[0].float())
            if class_bias is not None:
                target_latent = class_bias(target_latent, digits)
            classification = F.cross_entropy(head(target_latent), digits)
            prototype_target = source_prototypes[digits]
            if class_bias is not None:
                prototype_target = prototype_target + class_bias.bias(digits)
            prototype = F.mse_loss(target_latent, prototype_target)
            target_covariance = batch_covariance(target_latent)
            covariance = coral_loss(source_covariance, target_covariance)
            if source_class_covariances is not None:
                class_terms = []
                for digit in digits.unique():
                    mask = digits == digit
                    if int(mask.sum()) >= 2:
                        class_terms.append(
                            coral_loss(
                                source_class_covariances[int(digit)],
                                batch_covariance(target_latent[mask]),
                            )
                        )
                if class_terms:
                    covariance = covariance + torch.stack(class_terms).mean()

            uncertainty = torch.zeros((), device=device)
            paired_positions = [
                (position, source_label_to_index.get(int(pattern_id)))
                for position, pattern_id in enumerate(pattern_ids)
            ]
            paired_positions = [(a, b) for a, b in paired_positions if b is not None]
            if args.use_uncertainty and paired_positions:
                target_positions = [pair[0] for pair in paired_positions]
                source_positions = [pair[1] for pair in paired_positions]
                source_inputs = to_speckle_tensor(
                    source_speckles[source_positions], args.image_size, device
                )
                with torch.no_grad():
                    source_latent = vae.encode(source_inputs)[0].float()
                    source_mean, source_logvar = vae.decode(
                        source_latent, out_hw=(args.image_size, args.image_size)
                    )
                target_mean, target_logvar = vae.decode(
                    target_latent[target_positions],
                    out_hw=(args.image_size, args.image_size),
                )
                uncertainty = symmetric_gaussian_kl(
                    target_mean.clamp(0, 1),
                    target_logvar,
                    source_mean.clamp(0, 1),
                    source_logvar,
                )

            contrastive = torch.zeros((), device=device)
            if args.use_supcon and target_latent.shape[0] >= 16:
                contrastive = supervised_contrastive_loss(target_latent, digits)
            orthogonal = orthogonality_loss(aligner, args.w_orthogonality)
            loss = (
                args.w_classification * classification
                + args.w_prototype * prototype
                + coral_weight * covariance
                + uncertainty_weight * uncertainty
                + args.w_supcon * contrastive
                + orthogonal
            )

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += float(loss.detach())
    return total_loss / steps


def run_adaptation(
    *,
    vae: SimpleVAEWithUncertainty,
    head: LatentDigitHead,
    target_speckles: np.ndarray,
    target_images: np.ndarray,
    target_labels: np.ndarray,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    source_speckles: np.ndarray,
    source_label_to_index: Dict[int, int],
    source_prototypes: torch.Tensor,
    source_covariance: torch.Tensor,
    source_class_covariances: Optional[List[torch.Tensor]],
    evaluation_epochs: Sequence[int],
    args: argparse.Namespace,
    device: torch.device,
    run_seed: int,
) -> List[Tuple[int, Dict[str, float]]]:
    set_seed(run_seed)
    aligner = AlignLayer(args.latent_dim).to(device)
    class_bias = ClassConditionalBias(args.latent_dim).to(device) if args.use_class_bias else None
    parameters = list(aligner.parameters())
    if class_bias is not None:
        parameters += list(class_bias.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)
    scaler = make_grad_scaler(device, args.amp)
    checkpoints = sorted(set(evaluation_epochs))
    final_epoch = max(checkpoints)
    results: List[Tuple[int, Dict[str, float]]] = []

    if 0 in checkpoints:
        results.append(
            (
                0,
                evaluate(
                    vae,
                    head,
                    aligner,
                    class_bias,
                    target_speckles,
                    target_images,
                    target_labels,
                    test_indices,
                    args.per_digit_hint,
                    args.image_size,
                    args.eval_batch_size,
                    device,
                    args.amp,
                ),
            )
        )

    for epoch in range(1, final_epoch + 1):
        loss = train_one_epoch(
            vae=vae,
            head=head,
            aligner=aligner,
            class_bias=class_bias,
            optimizer=optimizer,
            scaler=scaler,
            target_speckles=target_speckles,
            target_labels=target_labels,
            train_indices=train_indices,
            source_speckles=source_speckles,
            source_label_to_index=source_label_to_index,
            source_prototypes=source_prototypes,
            source_covariance=source_covariance,
            source_class_covariances=source_class_covariances,
            args=args,
            device=device,
            epoch=epoch,
            total_epochs=final_epoch,
            run_seed=run_seed,
        )
        if epoch in checkpoints:
            metrics = evaluate(
                vae,
                head,
                aligner,
                class_bias,
                target_speckles,
                target_images,
                target_labels,
                test_indices,
                args.per_digit_hint,
                args.image_size,
                args.eval_batch_size,
                device,
                args.amp,
            )
            results.append((epoch, metrics))
            log(
                f"epoch={epoch}/{final_epoch} loss={loss:.5f} "
                f"MSE={metrics['MSE']:.6f} PSNR={metrics['PSNR']:.3f} "
                f"SSIM={metrics['SSIM']:.4f} Acc={metrics['Accuracy']:.2f}%"
            )
    return results


def read_result_rows(path: Path) -> List[Dict[str, object]]:
    if not path.exists():
        return []
    rows: List[Dict[str, object]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            converted: Dict[str, object] = dict(row)
            for key in ("bending", "epoch", "train_samples", "test_samples"):
                converted[key] = int(float(str(row[key])))
            for key in ("fraction", *METRIC_KEYS):
                converted[key] = float(str(row[key]))
            rows.append(converted)
    return rows


def write_result_rows(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def aggregate_rows(
    rows: Sequence[Dict[str, object]], group_key: str
) -> List[Dict[str, object]]:
    grouped: Dict[float, List[Dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(float(row[group_key]), []).append(row)
    output: List[Dict[str, object]] = []
    for value in sorted(grouped):
        group = grouped[value]
        result: Dict[str, object] = {group_key: value, "Bendings": len(group)}
        result["Train samples"] = int(round(np.mean([float(row["train_samples"]) for row in group])))
        for metric in METRIC_KEYS:
            values = np.asarray([float(row[metric]) for row in group], dtype=np.float64)
            result[f"{metric} mean"] = float(values.mean())
            result[f"{metric} std"] = float(values.std(ddof=0))
        output.append(result)
    return output


def format_mean_std(mean: float, std: float, digits: int) -> str:
    return f"{mean:.{digits}f} +/- {std:.{digits}f}"


def aggregate_markdown(rows: Sequence[Dict[str, object]], group_key: str, title: str) -> str:
    aggregates = aggregate_rows(rows, group_key)
    label = "Epoch" if group_key == "epoch" else "Fraction"
    lines = [
        f"## {title}",
        "",
        f"| {label} | Mean train N | MSE | PSNR (dB) | SSIM | Accuracy (%) | Bendings |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregates:
        group_value = int(row[group_key]) if group_key == "epoch" else f"{row[group_key]:.2f}"
        lines.append(
            "| "
            + " | ".join(
                [
                    str(group_value),
                    str(row["Train samples"]),
                    format_mean_std(row["MSE mean"], row["MSE std"], 5),
                    format_mean_std(row["PSNR mean"], row["PSNR std"], 2),
                    format_mean_std(row["SSIM mean"], row["SSIM std"], 4),
                    format_mean_std(row["Accuracy mean"], row["Accuracy std"], 2),
                    str(row["Bendings"]),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def epoch_detailed_markdown(rows: Sequence[Dict[str, object]], title: str) -> str:
    lines = [
        f"## {title}",
        "",
        "| Bending | Epoch | Train N | Test N | MSE | PSNR (dB) | SSIM | Accuracy (%) |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(rows, key=lambda item: (int(item["bending"]), int(item["epoch"]))):
        lines.append(
            f"| {row['bending']} | {row['epoch']} | {row['train_samples']} | "
            f"{row['test_samples']} | {float(row['MSE']):.5f} | {float(row['PSNR']):.2f} | "
            f"{float(row['SSIM']):.4f} | {float(row['Accuracy']):.2f} |"
        )
    return "\n".join(lines)


def data_detailed_markdown(rows: Sequence[Dict[str, object]], title: str) -> str:
    lines = [
        f"## {title}",
        "",
        "| Bending | Fraction | Train N | Test N | MSE | PSNR (dB) | SSIM | Accuracy (%) |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(rows, key=lambda item: (int(item["bending"]), float(item["fraction"]))):
        lines.append(
            f"| {row['bending']} | {float(row['fraction']):.2f} | {row['train_samples']} | "
            f"{row['test_samples']} | {float(row['MSE']):.5f} | {float(row['PSNR']):.2f} | "
            f"{float(row['SSIM']):.4f} | {float(row['Accuracy']):.2f} |"
        )
    return "\n".join(lines)


def write_aggregate_csv(
    path: Path, rows: Sequence[Dict[str, object]], group_key: str
) -> None:
    aggregates = aggregate_rows(rows, group_key)
    if not aggregates:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(aggregates[0].keys()))
        writer.writeheader()
        writer.writerows(aggregates)


def write_reports(output_dir: Path, rows: Sequence[Dict[str, object]]) -> str:
    sections = [
        "# ULDA Data-efficiency Results",
        "",
        "Metrics are evaluated on fixed held-out target-domain test splits.",
        "Target-domain ground-truth images are not used during adaptation.",
    ]
    epoch_rows = [row for row in rows if row["experiment"] == "epoch_trajectory"]
    data_rows = [row for row in rows if row["experiment"] == "data_fraction"]
    if epoch_rows:
        write_result_rows(output_dir / "epoch_trajectory_by_bending.csv", epoch_rows)
        write_aggregate_csv(output_dir / "epoch_trajectory_mean.csv", epoch_rows, "epoch")
        sections.extend(
            [
                "",
                aggregate_markdown(epoch_rows, "epoch", "Epoch Trajectory (Mean Across Bendings)"),
                "",
                epoch_detailed_markdown(epoch_rows, "Epoch Trajectory by Bending"),
            ]
        )
    if data_rows:
        write_result_rows(output_dir / "data_fraction_by_bending.csv", data_rows)
        write_aggregate_csv(output_dir / "data_fraction_mean.csv", data_rows, "fraction")
        sections.extend(
            [
                "",
                aggregate_markdown(data_rows, "fraction", "Target-data Sweep (Mean Across Bendings)"),
                "",
                data_detailed_markdown(data_rows, "Target-data Sweep by Bending"),
            ]
        )
    report = "\n".join(sections) + "\n"
    (output_dir / "results_tables.md").write_text(report, encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Independent ULDA epoch- and target-data-efficiency experiments"
    )
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--vae_ckpt", required=True)
    parser.add_argument("--output_dir", default="")
    parser.add_argument("--source_domain", type=int, default=0)
    parser.add_argument("--target_domains", default="1-10", help="Example: 1-10 or 1,3,10")
    parser.add_argument(
        "--experiments",
        default="epoch,data",
        help="Comma-separated subset of: epoch,data",
    )
    parser.add_argument("--epoch_checkpoints", default="0,1,5,10,20,50")
    parser.add_argument("--fractions", default="0.01,0.05,0.10,0.25,0.50,1.00")
    parser.add_argument("--data_epochs", type=int, default=50)
    parser.add_argument("--test_ratio", type=float, default=0.2)
    parser.add_argument("--min_per_class", type=int, default=1)
    parser.add_argument(
        "--fixed_steps_per_epoch",
        type=int,
        default=0,
        help="0 uses natural epochs; positive values equalize optimization steps across fractions",
    )
    parser.add_argument("--per_digit_hint", type=int, default=800)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--latent_dim", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--eval_batch_size", type=int, default=256)
    parser.add_argument("--anchor_batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--w_classification", type=float, default=1.0)
    parser.add_argument("--w_prototype", type=float, default=2.0)
    parser.add_argument("--w_coral", type=float, default=0.5)
    parser.add_argument("--w_uncertainty", type=float, default=0.5)
    parser.add_argument("--w_orthogonality", type=float, default=1e-3)
    parser.add_argument("--w_supcon", type=float, default=0.1)
    parser.add_argument("--no_class_bias", dest="use_class_bias", action="store_false")
    parser.add_argument("--no_classwise_coral", dest="use_classwise_coral", action="store_false")
    parser.add_argument("--no_uncertainty", dest="use_uncertainty", action="store_false")
    parser.add_argument("--use_supcon", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.set_defaults(use_class_bias=True, use_classwise_coral=True, use_uncertainty=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.target_domains = parse_int_spec(args.target_domains)
    args.epoch_checkpoints = parse_int_spec(args.epoch_checkpoints)
    args.fractions = parse_float_spec(args.fractions)
    experiments = {value.strip() for value in args.experiments.split(",") if value.strip()}
    unknown = experiments - {"epoch", "data"}
    if unknown:
        raise ValueError(f"Unknown experiments: {sorted(unknown)}")
    if not experiments:
        raise ValueError("At least one experiment is required")
    if any(epoch < 0 for epoch in args.epoch_checkpoints):
        raise ValueError("Epoch checkpoints must be non-negative")
    if "epoch" in experiments and not args.epoch_checkpoints:
        raise ValueError("Epoch experiment requires at least one checkpoint")
    if args.data_epochs < 1 or args.batch_size < 1 or args.eval_batch_size < 1:
        raise ValueError("Epoch and batch-size arguments must be positive")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) if args.output_dir else Path("runs") / "ulda_data_efficiency" / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "all_results.csv"
    rows = read_result_rows(result_path)
    serializable_args = vars(args).copy()
    serializable_args["target_domains"] = list(args.target_domains)
    serializable_args["epoch_checkpoints"] = list(args.epoch_checkpoints)
    serializable_args["fractions"] = list(args.fractions)
    (output_dir / "args.json").write_text(json.dumps(serializable_args, indent=2), encoding="utf-8")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.amp = bool(args.amp and device.type == "cuda")
    log(f"device={device}, amp={args.amp}, output={output_dir}")

    root = Path(args.data_root)
    source_speckles, _, source_labels = load_bending_arrays(root, args.source_domain)
    source_label_to_index = make_label_to_index(source_labels)
    vae, head = load_frozen_backbone(Path(args.vae_ckpt), args.image_size, args.latent_dim, device)
    log("Computing source prototypes and covariance anchors")
    source_prototypes, source_covariance, source_class_covariances = compute_source_anchors(
        vae,
        source_speckles,
        source_labels,
        args.per_digit_hint,
        args.image_size,
        args.anchor_batch_size,
        device,
        args.amp,
        args.use_classwise_coral,
    )

    for bending in args.target_domains:
        log(f"Loading bending{bending}")
        target_speckles, target_images, target_labels = load_bending_arrays(root, bending)
        train_indices, test_indices = stratified_train_test_split(
            target_labels,
            args.per_digit_hint,
            args.test_ratio,
            args.seed + bending * 101,
        )
        np.savez(
            output_dir / f"split_bending{bending}.npz",
            train_indices=train_indices,
            test_indices=test_indices,
        )
        match_count = sum(int(label) in source_label_to_index for label in target_labels[train_indices])
        log(
            f"bending{bending}: train={len(train_indices)}, test={len(test_indices)}, "
            f"source-paired train IDs={match_count}/{len(train_indices)}"
        )
        if args.use_uncertainty and match_count == 0:
            raise ValueError(
                f"bending{bending} has no pattern IDs matching the source domain; "
                "uncertainty consistency cannot be evaluated"
            )
        class_orders = build_nested_class_orders(
            train_indices, target_labels, args.per_digit_hint, args.seed + bending * 103
        )

        if "epoch" in experiments:
            rows = [
                row
                for row in rows
                if not (row["experiment"] == "epoch_trajectory" and int(row["bending"]) == bending)
            ]
            log(f"Epoch trajectory: bending{bending}")
            trajectory = run_adaptation(
                vae=vae,
                head=head,
                target_speckles=target_speckles,
                target_images=target_images,
                target_labels=target_labels,
                train_indices=train_indices,
                test_indices=test_indices,
                source_speckles=source_speckles,
                source_label_to_index=source_label_to_index,
                source_prototypes=source_prototypes,
                source_covariance=source_covariance,
                source_class_covariances=source_class_covariances,
                evaluation_epochs=args.epoch_checkpoints,
                args=args,
                device=device,
                run_seed=args.seed + bending * 1009,
            )
            for epoch, metrics in trajectory:
                rows.append(
                    {
                        "experiment": "epoch_trajectory",
                        "bending": bending,
                        "epoch": epoch,
                        "fraction": 1.0,
                        "train_samples": len(train_indices),
                        "test_samples": len(test_indices),
                        **metrics,
                    }
                )
            write_result_rows(result_path, rows)
            write_reports(output_dir, rows)

        if "data" in experiments:
            for fraction in args.fractions:
                completed = any(
                    row["experiment"] == "data_fraction"
                    and int(row["bending"]) == bending
                    and abs(float(row["fraction"]) - fraction) < 1e-9
                    and int(row["epoch"]) == args.data_epochs
                    for row in rows
                )
                if completed:
                    log(f"Skipping completed bending{bending}, fraction={fraction:.2f}")
                    continue
                subset = nested_subset(
                    class_orders,
                    fraction,
                    args.min_per_class,
                    args.seed + bending * 107 + int(round(fraction * 10000)),
                )
                log(
                    f"Data sweep: bending{bending}, fraction={fraction:.2f}, "
                    f"train samples={len(subset)}"
                )
                final_result = run_adaptation(
                    vae=vae,
                    head=head,
                    target_speckles=target_speckles,
                    target_images=target_images,
                    target_labels=target_labels,
                    train_indices=subset,
                    test_indices=test_indices,
                    source_speckles=source_speckles,
                    source_label_to_index=source_label_to_index,
                    source_prototypes=source_prototypes,
                    source_covariance=source_covariance,
                    source_class_covariances=source_class_covariances,
                    evaluation_epochs=[args.data_epochs],
                    args=args,
                    device=device,
                    run_seed=args.seed + bending * 2003 + int(round(fraction * 10000)),
                )[-1][1]
                rows.append(
                    {
                        "experiment": "data_fraction",
                        "bending": bending,
                        "epoch": args.data_epochs,
                        "fraction": fraction,
                        "train_samples": len(subset),
                        "test_samples": len(test_indices),
                        **final_result,
                    }
                )
                write_result_rows(result_path, rows)
                write_reports(output_dir, rows)

    report = write_reports(output_dir, rows)
    print("\n" + report)
    log(f"Finished. Tables: {output_dir / 'results_tables.md'}")
    log(f"Raw results: {result_path}")


if __name__ == "__main__":
    main()

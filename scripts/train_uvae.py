#!/usr/bin/env python3
"""Train the calibrated uncertainty-aware VAE backbone."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

from ulda_mmf.data import DomainDataset
from ulda_mmf.losses import uvae_objective
from ulda_mmf.models import LatentDigitHead, UncertaintyVAE
from ulda_mmf.utils import resolve_device, save_checkpoint, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="Calibrated source-domain directory")
    parser.add_argument("--output", default="outputs/uvae/best.pt")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--latent_dim", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--beta", type=float, default=1e-4)
    parser.add_argument("--classification_weight", type=float, default=1.0)
    parser.add_argument("--val_fraction", type=float, default=0.1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def run_epoch(
    vae: UncertaintyVAE,
    head: LatentDigitHead,
    loader: DataLoader,
    device: torch.device,
    beta: float,
    classification_weight: float,
    optimizer: torch.optim.Optimizer | None,
) -> float:
    training = optimizer is not None
    vae.train(training)
    head.train(training)
    total = 0.0
    for batch in loader:
        speckle = batch["speckle"].to(device)
        image = batch["image"].to(device)
        labels = batch["label"].to(device)
        with torch.set_grad_enabled(training):
            output = vae(speckle)
            output["logits"] = head(output["latent_mean"])
            losses = uvae_objective(
                output, image, labels, beta, classification_weight
            )
            if training:
                optimizer.zero_grad(set_to_none=True)
                losses["total"].backward()
                optimizer.step()
        total += losses["total"].item() * speckle.shape[0]
    return total / len(loader.dataset)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    dataset = DomainDataset(args.source, require_images=True, require_labels=True)
    val_size = max(1, int(len(dataset) * args.val_fraction))
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(
        dataset,
        (train_size, val_size),
        generator=torch.Generator().manual_seed(args.seed),
    )
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size)
    vae = UncertaintyVAE(args.image_size, args.latent_dim).to(device)
    head = LatentDigitHead(args.latent_dim).to(device)
    optimizer = torch.optim.AdamW(
        list(vae.parameters()) + list(head.parameters()), lr=args.lr
    )
    best = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(
            vae, head, train_loader, device, args.beta, args.classification_weight, optimizer
        )
        val_loss = run_epoch(
            vae, head, val_loader, device, args.beta, args.classification_weight, None
        )
        print(f"epoch={epoch:03d} train={train_loss:.6f} val={val_loss:.6f}", flush=True)
        if val_loss < best:
            best = val_loss
            save_checkpoint(
                args.output,
                vae=vae.state_dict(),
                head=head.state_dict(),
                image_size=args.image_size,
                latent_dim=args.latent_dim,
                epoch=epoch,
                val_loss=val_loss,
            )


if __name__ == "__main__":
    main()

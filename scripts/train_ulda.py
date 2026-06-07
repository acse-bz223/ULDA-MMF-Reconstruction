#!/usr/bin/env python3
"""Adapt a frozen UVAE to one target fiber state without target-domain images."""

from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader

from ulda_mmf.data import DomainDataset, PairedDomainDataset
from ulda_mmf.losses import (
    ULDALossWeights,
    orthogonality_regularization,
    ulda_objective,
)
from ulda_mmf.models import LatentDigitHead, ULDAAdaptor, UncertaintyVAE
from ulda_mmf.models.ulda import compute_prototypes
from ulda_mmf.utils import resolve_device, save_checkpoint, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--uvae_checkpoint", required=True)
    parser.add_argument("--output", default="outputs/ulda/target_adaptor.pt")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--latent_dim", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--w_prototype", type=float, default=2.0)
    parser.add_argument("--w_covariance", type=float, default=0.5)
    parser.add_argument("--w_classification", type=float, default=1.0)
    parser.add_argument("--w_uncertainty", type=float, default=0.5)
    parser.add_argument("--w_orthogonality", type=float, default=1e-3)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    source = DomainDataset(
        args.source, require_labels=True, require_ids=True
    )
    target = DomainDataset(
        args.target, require_labels=True, require_ids=True
    )
    pairs = PairedDomainDataset(source, target)
    pair_loader = DataLoader(pairs, batch_size=args.batch_size, shuffle=True)
    source_loader = DataLoader(source, batch_size=args.batch_size)

    vae = UncertaintyVAE(args.image_size, args.latent_dim).to(device)
    head = LatentDigitHead(args.latent_dim).to(device)
    checkpoint = torch.load(args.uvae_checkpoint, map_location="cpu")
    vae.load_state_dict(checkpoint["vae"])
    head.load_state_dict(checkpoint["head"])
    vae.eval()
    head.eval()
    for parameter in list(vae.parameters()) + list(head.parameters()):
        parameter.requires_grad_(False)

    adaptor = ULDAAdaptor(args.latent_dim).to(device)
    optimizer = torch.optim.AdamW(adaptor.parameters(), lr=args.lr)
    prototypes = compute_prototypes(
        vae, source_loader, 10, args.latent_dim, device
    )
    weights = ULDALossWeights(
        prototype=args.w_prototype,
        covariance=args.w_covariance,
        classification=args.w_classification,
        uncertainty=args.w_uncertainty,
    )

    for epoch in range(1, args.epochs + 1):
        total = 0.0
        for batch in pair_loader:
            source_speckle = batch["source"]["speckle"].to(device)
            target_speckle = batch["target"]["speckle"].to(device)
            target_labels = batch["target"]["label"].to(device)
            with torch.no_grad():
                source_latent, _ = vae.encode(source_speckle)
                target_latent, _ = vae.encode(target_speckle)
                source_prediction = vae.decode(source_latent)
            aligned_target = adaptor(target_latent, target_labels)
            target_prediction = vae.decode(aligned_target)
            target_logits = head(aligned_target)
            bias_aware_prototypes = prototypes + adaptor.class_bias.embedding.weight
            losses = ulda_objective(
                source_latent,
                aligned_target,
                target_logits,
                target_labels,
                bias_aware_prototypes,
                source_prediction,
                target_prediction,
                weights,
            )
            orthogonality = orthogonality_regularization(
                adaptor.aligner.residual.weight
            )
            loss = losses["total"] + args.w_orthogonality * orthogonality
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += loss.item() * source_speckle.shape[0]
        print(f"epoch={epoch:03d} loss={total / len(pairs):.6f}", flush=True)

    save_checkpoint(
        args.output,
        adaptor=adaptor.state_dict(),
        image_size=args.image_size,
        latent_dim=args.latent_dim,
    )


if __name__ == "__main__":
    main()

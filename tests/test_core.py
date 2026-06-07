import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ulda_mmf.data import DomainDataset, PairedDomainDataset
from ulda_mmf.losses import symmetric_gaussian_kl
from ulda_mmf.models import (
    LatentDigitHead,
    MLPReconstructor,
    ResidualLatentAligner,
    RTMNet,
    UNet,
    UncertaintyVAE,
)


def test_uvae_and_head_shapes():
    model = UncertaintyVAE(image_size=32, latent_dim=16, base_channels=4)
    head = LatentDigitHead(latent_dim=16)
    output = model(torch.rand(2, 1, 32, 32))
    assert output["reconstruction_mean"].shape == (2, 1, 32, 32)
    assert head(output["latent_mean"]).shape == (2, 10)


def test_residual_aligner_starts_as_identity():
    aligner = ResidualLatentAligner(8)
    latent = torch.randn(3, 8)
    assert torch.allclose(aligner(latent), latent)


def test_symmetric_kl_is_zero_for_identical_distributions():
    mean = torch.randn(2, 1, 8, 8)
    logvar = torch.randn(2, 1, 8, 8).clamp(-2, 2)
    assert symmetric_gaussian_kl(mean, logvar, mean, logvar).item() == pytest.approx(0)


def test_baseline_shapes():
    tensor = torch.rand(2, 1, 32, 32)
    assert MLPReconstructor(image_size=32, hidden_dim=32)(tensor).shape == tensor.shape
    assert UNet(base_channels=4)(tensor).shape == tensor.shape
    assert RTMNet(image_size=32, base_channels=4)(tensor).shape == tensor.shape


def test_paired_domain_dataset(tmp_path):
    for name, offset in (("source", 0), ("target", 1)):
        root = tmp_path / name
        root.mkdir()
        np.save(root / "speckles.npy", np.ones((3, 8, 8), dtype=np.float32) * offset)
        np.save(root / "labels.npy", np.array([0, 1, 2]))
        np.save(root / "ids.npy", np.array([10, 11, 12]))
    source = DomainDataset(tmp_path / "source", require_labels=True, require_ids=True)
    target = DomainDataset(tmp_path / "target", require_labels=True, require_ids=True)
    paired = PairedDomainDataset(source, target)
    assert len(paired) == 3
    assert paired[0]["source"]["id"].item() == paired[0]["target"]["id"].item()
    del paired
    source.close()
    target.close()

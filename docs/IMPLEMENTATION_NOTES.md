# Implementation notes

This release extracts the reusable method from the original experimental
notebooks into importable modules and command-line training scripts.

The public implementation follows the method described in the manuscript and
Supporting Information:

- a probabilistic UVAE backbone predicts an image mean and pixel-wise log
  variance;
- the calibrated backbone remains frozen during target-domain adaptation;
- a lightweight residual latent aligner and class-conditional bias are trained
  using prototype, covariance, classification, and uncertainty-consistency
  losses;
- target-domain ground-truth images are never consumed during adaptation.

The latent dimension is configurable. The released command-line defaults use
`latent_dim=512`, matching the ULDA aligner configuration used by the core
experimental code and reported in Supporting Information Table S4.

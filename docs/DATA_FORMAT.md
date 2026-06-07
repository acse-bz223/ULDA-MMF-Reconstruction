# Data interface

Datasets are deliberately not distributed in this repository.

Each physical fiber state should be represented by one directory containing
NumPy arrays with a shared first dimension:

```text
bending0/
  speckles.npy     # required: [N, H, W] or [N, 1, H, W]
  images.npy       # required for supervised source training/evaluation
  labels.npy       # integer class labels, required by ULDA
  ids.npy          # stable DMD pattern IDs, required to pair source and target

bending1/
  speckles.npy
  labels.npy
  ids.npy
```

The loader also recognizes the historical names `speckles_sorted.npy`,
`images_sorted.npy`, and `labels_sorted.npy`.

Input and target images should be normalized to `[0, 1]`. Integer arrays are
automatically divided by the maximum value of their dtype. Target-domain
ground-truth images are not read by the ULDA training script.

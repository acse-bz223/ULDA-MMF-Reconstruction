# Code review

The original project is an experiment workspace rather than a publishable
software repository. It contains large datasets and checkpoints, repeated
model definitions across notebooks, hard-coded paths, and training/evaluation
logic coupled in single files. This release extracts the reusable method and
copies the server-side reproduction scripts without any data or weights.

## Corrected in this release

1. **Residual aligner initialization.** The server scripts initialized
   `fc.weight` to identity and then returned `z + fc(z)`, so the nominal
   identity initialization was approximately `2z`. Public copies now
   zero-initialize the residual branch and regularize the effective transform
   `I + W`.
2. **Class-bias prototype conflict.** The main continual ULDA script applied a
   class-conditional bias to target features but compared them with unshifted
   prototypes, directly penalizing the bias. The prototype anchor is now
   bias-aware.
3. **Broken RTMNet loss option.** The server script exposed `--loss vae` but
   depended on an unavailable external `vae.py` API. The public reproduction
   script exposes the self-contained `l1`, `l2`, and `npcc` options only.

## Remaining research-script limitations

- The continual scripts concatenate all replay domains with
  `np.concatenate`, which can create a large RAM peak. The compact
  `scripts/train_ulda.py` entry point is preferred when memory is limited.
- Research scripts load target-domain images for offline evaluation. Their
  adaptation losses do not consume those images, but the scripts still require
  them to exist. The compact `scripts/train_ulda.py` entry point never opens a
  target-domain image array.
- Historical `labels_sorted.npy` arrays encode global DMD pattern IDs, and
  digit classes are inferred using `per_digit_hint=800`. The modular data
  interface separates `labels.npy` and `ids.npy` to avoid this dataset-specific
  assumption.
- RTMNet is trained in the sinogram domain and optionally requires
  `torch-radon` for its image-domain auxiliary term.

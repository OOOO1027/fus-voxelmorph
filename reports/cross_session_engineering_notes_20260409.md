# Cross-Session Engineering Notes

## 1. Why The Old Summary Was Removed

The removed report `training_summary_cross_session_300epochs_20260406.md`
claimed aggregate results such as:

- overall `DL NCC = 0.977`
- val/test `DL NCC` near `1.0`
- clear superiority over Stage 3B

Those claims are no longer safe to use because the repository also contains:

- a later audit showing the summary and plan used different metric spaces
- evaluation files that conflict with the summary-level conclusion
- a newer April 9, 2026 GPU training run whose checkpoints were never paired
  with a fresh evaluation export

The file was removed to reduce the chance of future reporting drift.

## 2. The Mask-Aware Grad Scaling Issue

### Current code path

See `losses/losses.py`, `Grad.forward()`:

```python
mask_dy = mask[:, :, 1:, :] * mask[:, :, :-1, :]
loss_dy = (dy.pow(2) * mask_dy).sum() / mask_dy.sum().clamp(min=1.0)
```

### What is happening

- `dy` has shape `(B, 2, H-1, W)` because the flow has two channels:
  horizontal displacement and vertical displacement.
- `mask_dy` has shape `(B, 1, H-1, W)`.
- During multiplication, `mask_dy` is broadcast to both flow channels.

So the numerator sums **two channels**, but the denominator counts only the
valid pixels in **one channel**.

### Simple example

If one valid location has:

- x-gradient squared = `0.01`
- y-gradient squared = `0.01`

then the current masked formula gives:

```text
(0.01 + 0.01) / 1 = 0.02
```

but the channel-aware mean should be:

```text
(0.01 + 0.01) / 2 = 0.01
```

### Practical consequence

Under `mask_aware=true`, the gradient regularization term is effectively scaled
up by about `2x` relative to the unmasked formulation.

That means:

- the configured `reg_weight` is not the true effective strength
- the model is pushed more strongly toward smoother / smaller deformations
- comparisons across runs are slightly less clean unless the same bug exists in
  all compared runs

This is a calibration issue, not proof that the model is invalid. A constant
factor can be compensated by retuning `reg_weight`, but it should be fixed
before using the run for strict quantitative comparison.

## 3. Run Versioning Convention

Artifacts are now organized as:

```text
checkpoints/cross_session/
  runs/
    20260406_cpu_legacy_91k/
    20260409_gpu_ms_ncc_337k/
```

Convention for future runs:

- directory name: `YYYYMMDD_<device>_<short-description>/`
- fresh run: new directory
- resumed run: same directory + explicit `--resume`

`train_v2.py` now blocks fresh training if `save_dir` already contains
checkpoints or a training log.

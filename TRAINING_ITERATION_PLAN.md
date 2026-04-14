# VoxelMorph Training Iteration Plan

## Current Baseline (2026-04-13)

| Item | Value |
|------|-------|
| Model | VxmDense2D, 91K params |
| Input mode | Mode B (rigid-prealigned) |
| Training | bidir + mask-aware MS-NCC + Grad, cosine LR w/ warmup |
| Train NCC | ~-0.50 (high variance) |
| Val NCC | ~-0.46 (saturated at epoch ~250) |
| Train pairs | 32 (16 + reverse) |
| Device | CPU, ~8.6s/epoch |
| Problem | Data too few, val saturated, can't beat Stage 3B reliably |

## Phase 0: Baseline Evaluation

**Goal**: Establish exact reference numbers before any changes.

```bash
python evaluate.py --config configs/cross_session.yaml \
    --model checkpoints/cross_session/best_model.pth
```

**Record**: test NCC, SSIM, MSE, Jacobian folding %, comparison vs Stage 3B.

**Output dir**: `checkpoints/phase0_baseline/`

---

## Phase 1: Data-Only Swap (trial-level sub-maps)

**Change**: Point `cross_session_dir` to new trial-level data.
**Do NOT change**: model, loss, augmentation params, any hyperparameters.

```yaml
# configs/cross_session_phase1.yaml
data:
  cross_session_dir: "data/trial_level/"  # only this line changes
  # everything else identical to cross_session.yaml
```

```bash
python train_v2.py --config configs/cross_session_phase1.yaml
```

**Expected**: Train pairs 32 -> ~400-1000. Val NCC should significantly beat -0.46.

**Go/No-go**: Val NCC < -0.50? Yes -> Phase 2. No -> investigate sub-map quality.

**Output dir**: `checkpoints/phase1_trial_data/`

---

## Phase 2: Reduce Augmentation Intensity

**Change**: Only augmentation params in yaml. No code changes.

```yaml
augmentation:
  affine:
    rotation: 3          # was 5
    translation: 3        # was 5
    scale: [0.97, 1.03]   # was [0.95, 1.05]
    p: 0.5
  elastic:
    grid_size: 6          # was 4 (smoother)
    magnitude: 5.0        # was 8.0 (less aggressive)
    p: 0.5
  intensity:
    noise_std: 0.015      # was 0.02
    multiplicative_range: [0.9, 1.1]  # was [0.95, 1.05], widened
    brightness_range: [-0.03, 0.03]
    p: 0.5
  gamma:
    range: [0.8, 1.2]    # was [0.7, 1.3]
    p: 0.3               # was 0.5
  flip:
    p: 0.3               # was 0.5
  crop:
    enabled: false        # was true
```

**Comparison**: Phase 1 config (full aug) vs Phase 2 config (reduced aug), same data.

**Output dir**: `checkpoints/phase2_aug_tuned/`

---

## Phase 3: Hyperparameter Adaptation

Each sub-step is independent. Run A/B comparison for each.

### 3a. Batch size

```yaml
dataloader:
  batch_size: 8           # was 4
train:
  accumulate_steps: 2     # was 4 (effective batch stays 16)
```

### 3b. Learning rate

```yaml
train:
  lr: 2.0e-4              # was 1e-4
  warmup_epochs: 10       # was 20
```

### 3c. Training duration

```yaml
train:
  epochs: 200             # was 500
  early_stopping: 40      # was 80
```

### 3d. Regularization schedule

```yaml
train:
  reg_schedule:
    start: 0.02           # was 0.01
    end: 0.08             # was 0.1
    warmup_fraction: 0.2  # was 0.33
```

**Output dirs**: `checkpoints/phase3a_batchsize/`, `phase3b_lr/`, etc.

---

## Phase 4: Loss Function Refinement

### 4a. Add Edge-NCC

New loss component: NCC computed on Sobel-filtered (edge) images.
Targets vessel boundary alignment.

```yaml
loss:
  edge_ncc_weight: 0.3    # total = MS-NCC + 0.3 * EdgeNCC + reg_weight * Grad
```

Requires adding ~20 lines to `losses/losses.py`.

### 4b. Diffusion regularizer

```yaml
loss:
  reg_type: "diffusion"   # was "grad" (second-order, smoother fields)
```

Config-only change, already implemented in `losses.py`.

**Output dirs**: `checkpoints/phase4a_edge_ncc/`, `phase4b_diffusion/`

---

## Phase 5: Model Architecture (only if underfitting confirmed)

**Enter only if**: train loss stops improving (not just val loss).

### 5a. Wider channels

```yaml
model:
  enc_channels: [32, 64, 128, 128]      # was [32, 64, 64, 64]
  dec_channels: [128, 64, 64, 32, 32, 16]
```

### 5b. Deeper encoder

```yaml
model:
  enc_channels: [32, 64, 64, 128, 128]
  dec_channels: [128, 64, 64, 64, 32, 32, 16]
```

---

## Discipline Rules

1. **Never skip phases**. Phase 1 (data) likely provides more gain than Phase 3-5 combined.
2. **One variable per experiment**. If multiple things change, failures can't be diagnosed.
3. **Separate checkpoints per phase**. Named directories, not overwritten.
4. **Same test set throughout**. Leave-Session-Out with S9 as test.
5. **Record every experiment** in a comparison table (append to this file).
6. **If a phase shows no improvement**, understand why before proceeding.

## Results Log

| Phase | Val NCC | Test NCC | vs Stage3B | Notes |
|-------|---------|----------|------------|-------|
| 0 (baseline) | | | | |
| 1 (trial data) | | | | |
| 2 (aug tuned) | | | | |
| ... | | | | |

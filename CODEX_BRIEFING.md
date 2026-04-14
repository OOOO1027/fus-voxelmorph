# Codex Project Briefing — fUS-VoxelMorph

> Last updated: 2026-04-13

---

## Competition Abstract / 竞赛报名简介

### 中文（约375字）

功能性超声（fUS）成像以其高时空分辨率成为非人灵长类脑机接口研究的新兴手段，但跨Session图像配准与脑区自动划分两大关键问题尚缺乏系统性解决方案。本项目基于Nature Neuroscience（2024）发表的猕猴闭环超声脑机接口数据集，构建从跨Session可变形配准到功能性脑区精细划分的端到端计算管线。

在配准阶段，本项目**首次将深度学习可变形配准框架（VoxelMorph）引入fUS成像领域**，提出适配2D冠状面Power Doppler图像的轻量级微分同胚配准模型VxmDense2D。模型以刚性预对齐图像对为输入，通过缩放平方积分保证形变场的拓扑保持性；训练采用双向掩模感知多尺度归一化互相关损失，并设计**试次级时序增广策略**——利用fDOP完整时间序列中的精确试次边界，将训练样本从24对扩展至数百对，系统缓解小样本瓶颈。

在脑区划分阶段，本项目提出**解剖先验与功能数据双驱动的并行划分框架**：（1）图谱方案（Plan A）将Calabrese猕猴DTI脑图谱通过跨模态边缘-地标配准映射至fUS空间，借助**配准形变场级联传递**实现一次图谱配准、全Session标签迁移；（2）数据驱动方案（Plan B）在配准建立的公共空间中提取试次平均方向图、调谐强度等时序功能特征，采用谱聚类等无监督方法自动发现功能分区；（3）融合方案（Plan A+B）以**配准置信度空间调制**融合权重——高置信区域信任跨Session功能数据，低置信区域依赖图谱解剖先验——实现自适应优势互补。三套结果均以专业研究员手工分割为金标准进行定量评估。

本项目的创新点在于：将深度学习配准引入fUS这一新兴模态、提出试次级时序增广突破小样本限制、设计配准-划分紧耦合的级联架构、以及提出置信度调制的双驱动融合划分策略。

### English (~375 words)

Functional ultrasound (fUS) imaging has emerged as a powerful tool for primate brain-machine interface (BMI) research due to its superior spatiotemporal resolution, yet systematic solutions for cross-session image registration and automatic brain region parcellation remain lacking. Building on a macaque closed-loop ultrasonic BMI dataset published in Nature Neuroscience (2024), this project develops an end-to-end computational pipeline spanning deformable cross-session registration to fine-grained functional brain region parcellation.

For registration, we **introduce deep learning deformable registration (VoxelMorph) to fUS imaging for the first time**, proposing VxmDense2D — a lightweight diffeomorphic model tailored for 2D coronal Power Doppler images. The model takes rigid-prealigned image pairs as input and guarantees topology preservation via scaling-and-squaring integration. Training employs bidirectional mask-aware multi-scale normalized cross-correlation loss. Critically, we design a **trial-level temporal augmentation strategy** that exploits precise trial boundaries within the full fDOP time series to expand training pairs from 24 to several hundred, systematically alleviating the small-sample bottleneck.

For parcellation, we propose a **dual-driven parallel framework combining anatomical priors with functional data**: (1) An atlas-based pipeline (Plan A) registers the Calabrese macaque DTI brain atlas to fUS space via cross-modal edge-landmark registration and **cascades labels to all sessions through VoxelMorph warp fields** — one atlas registration propagates to every session. (2) A data-driven pipeline (Plan B) extracts temporal functional features (trial-averaged directional maps, tuning strength, temporal correlation) in the VoxelMorph-aligned common space and applies unsupervised clustering (spectral clustering, K-means, NMF) to discover functional parcels. (3) An adaptive fusion pipeline (Plan A+B) combines both results using **registration-confidence-modulated spatial weighting** — trusting cross-session functional data where registration is reliable and falling back to atlas anatomical priors where it is not. All three output lines are quantitatively evaluated against expert manual segmentation as the gold standard.

The key innovations of this project are: introducing deep learning registration to the emerging fUS modality; proposing trial-level temporal augmentation to overcome data scarcity; designing a tightly coupled registration-parcellation cascade architecture where warp fields serve as both alignment tools and feature sources; and developing a confidence-modulated dual-driven fusion parcellation strategy that adaptively leverages the complementary strengths of anatomical and functional information.

---

## Codex Technical Briefing / 技术交接文档

---

## Current Codebase Structure

```
fus-voxelmorph/
├── models/vxm2d.py              # VxmDense2D: UNet + VecInt + SpatialTransformer (91K params)
├── losses/losses.py             # NCC, MultiScaleNCC, MSE, Grad, Diffusion, BendingEnergy
├── data/
│   ├── cross_session_dataset.py # CrossSessionPairDataset (Mode A/B, NPZ bundles)
│   ├── fus_dataset.py           # Augmentation: RandomAffine2D, Elastic, Gamma, etc.
│   └── synthetic_fus.py         # Synthetic vessel trees + deformation for pretraining
├── configs/
│   └── cross_session.yaml       # Main training config (augmentation, model, loss, train)
├── train_v2.py                  # Main training script (708 lines, bidir, grad accum, SWA)
├── evaluate.py                  # 3-way comparison: pre vs stage3b vs dl_output
├── baselines/traditional.py     # Non-DL baselines (ANTs, elastix, demons)
├── utils/metrics.py             # NCC, MSE, SSIM, MS-SSIM, DSC, HaarPSI, Jacobian
├── checkpoints/
│   ├── best_model.pth           # Current best VxmDense2D
│   └── cross_session/           # Training logs + per-epoch checkpoints
├── scripts/                     # Visualization, demo, TensorBoard launch
│   └── (parcellation scripts will be added here)
├── TRAINING_ITERATION_PLAN.md   # 5-phase iterative training improvement plan
└── PARCELLATION_PLAN.md         # Full parcellation plan (Plan A + B + A+B fusion)
```

## Stage 1 Status: Cross-Session Registration

### What exists
- Full training + evaluation pipeline, functional on CPU
- VxmDense2D with bidir + mask-aware MS-NCC + Grad loss
- 16 train pairs (32 with reverse), 4 val, 4 test (Leave-Session-Out, S9 as hard test)
- Train NCC ~-0.50, Val NCC ~-0.46, saturated at epoch ~250
- Non-DL baselines (ANTs SyN, elastix, optical flow) for comparison

### What's next (5-phase plan, strict order)
| Phase | Change | Expected Impact |
|-------|--------|----------------|
| 0 | Baseline eval — record exact test metrics | Reference numbers |
| 1 | Data-only swap to trial-level sub-maps (24 → ~400+ pairs) | Major NCC improvement |
| 2 | Reduce augmentation intensity (less needed with more data) | Reduce noise |
| 3 | Hyperparams: batch size, LR, epochs, reg schedule | Fine-tune convergence |
| 4 | Loss: add Edge-NCC (Sobel), try Diffusion regularizer | Boundary precision |
| 5 | Architecture: wider/deeper (only if underfitting confirmed) | Capacity |

**Critical discipline**: one variable per phase, separate checkpoint dirs, same test set.

### Trial-level data augmentation (not yet executed)
- Each .mat session file contains full `dop` time series (z x x x n frames)
- Currently only the static `neurovascular_map` is used → 24 pairs total
- Plan: segment dop into trial groups using `behavior` + `state_num`, average each group → multiple sub-maps per session → hundreds of training pairs
- Precise trial boundaries available: `behavior.trial_start`, `state_num` encodes task states (fixation, cue, memory, movement, hold, ITI)
- Codex prompt for this extraction has been drafted (requires server with .mat files)

## Stage 2 Status: Brain Region Parcellation

### Three parallel output lines

```
Plan A (atlas-based)          Plan B (data-driven)         Plan A+B (fusion)
─────────────────────         ──────────────────────       ──────────────────
Calabrese 2015 atlas          fDOP temporal features       Weighted voting
  ↓                             ↓                          Boundary consensus
Slice localization            VoxelMorph warp to           Atlas prior +
  (slice ~113, -22.8mm)        common space                 data-driven refinement
  ↓                             ↓                            ↓
Cross-modal 2D reg            Spectral clustering          Fused label map
  (landmark → edge)            K-means, NMF                  ↓
  ↓                             ↓                          Evaluate vs GT
Label transfer                Cluster label map
  (nearest-neighbor warp)       ↓
  ↓                           Evaluate vs GT
Evaluate vs GT
```

### Gold standard
- Expert manual segmentation in `定位.pptx` — professional researcher hand-labeled regions on fUS neurovascular map
- Regions: area5d, MIP, VIP, PIP, LIP, area7a, 7op
- This is the single source of truth for evaluating all three output lines (Dice/IoU per region)

### How DL registration feeds into parcellation (critical coupling)

| Registration output | Used in Plan A | Used in Plan B | Used in A+B |
|---|---|---|---|
| best_model.pth | A3-3: atlas-fUS warp inference | — | — |
| Cross-session warp fields | A4: cascade label transfer (atlas→ref→target, one atlas-reg needed) | B1: warp dop time series to common space | Common space for comparison |
| Warped neurovascular maps | A2: slice localization in common space (once for all sessions) | B2: clustering in aligned pixel space | Overlay visualization |
| Jacobian / deformation stats | — | B2: extra clustering feature dimension | Confidence-weighted spatial fusion |
| Local NCC (registration quality) | — | — | Spatial modulation: trust Plan B where registration is good, Plan A where it's poor |

### Key cross-modal challenge
- fUS = Power Doppler (blood vessel signal) vs Atlas = DTI/T1 (tissue structure)
- Direct intensity matching impossible — pixel values have completely different physics
- Bridge strategy: edge maps (Sobel), structural landmarks (IPS = intraparietal sulcus visible in both modalities), contour shape matching
- Recommended approach: landmark-based TPS first (fast), then edge-to-edge automation

## Technical Details to Watch

### Data format (.mat files)
```
dop:                z × x × n    Power Doppler time series (~2Hz)
actual_labels:      n × 1        movement direction per frame
behavior:           1 × b struct  trial_start, effector, success, target_pos
state_num:          n × 1        task state (fixation/cue/memory/movement/hold/ITI)
neurovascular_map:  z × x        static average Power Doppler
```

### Model architecture (VxmDense2D)
- Encoder: [32, 64, 64, 64] channels, 4 downsampling levels
- Decoder: [64, 64, 64, 32, 32, 16] with SEBlock attention + residual at full-res
- Integration: 7 scaling-and-squaring steps for diffeomorphic guarantee
- Bidir: integrates +v and -v for forward/inverse warps
- Flow head initialized near-zero → starts from identity

### Loss function
- **Similarity**: Multi-scale NCC at scales [1, 2, 4], mask-aware (ignores background)
- **Regularization**: Grad L2 (first-order finite differences), with linear warmup schedule (0.01 → 0.1 over first 33% of training)
- **Bidir**: loss computed in both directions, averaged

### Training config
- Optimizer: Adam, lr=1e-4, cosine schedule with 20-epoch warmup
- Gradient accumulation: 4 steps (effective batch 16 from batch_size 4)
- SWA: enabled after 80% of training
- Early stopping: patience 80 epochs on val NCC

### Preprocessing
- Percentile normalization (1st-99th) → log1p transform
- Mode B: model sees rigid-prealigned moving image, learns residual deformation only

## Roadmap & Expected Timeline

```
NOW ─────────────────────────────────────────────────────────────── DONE
 │
 ├─ Trial-level data extraction (Codex task, needs .mat files on server)
 │    Output: data/trial_level/ with hundreds of NPZ pair bundles
 │
 ├─ Phase 0: Baseline evaluation (record test NCC/SSIM/Jacobian)
 │
 ├─ Phase 1-2: Data swap + aug tuning (biggest expected gain)
 │    Target: Val NCC < -0.55 (currently -0.46)
 │
 ├─ Phase 3-4: Hyperparams + loss (if needed)
 │
 ╞══ REGISTRATION DONE ════════════════════════════════════════════
 │
 ├─ Parcellation Stage 1: Atlas infrastructure
 │    Download Calabrese 2015, explore_atlas.py, fUS landmark extraction
 │
 ├─ Parcellation Stage 2 & 3 (PARALLEL):
 │    Stage 2: Plan A — slice localization + atlas-fUS registration + label transfer
 │    Stage 3: Plan B — feature extraction + clustering + independent evaluation
 │
 ├─ Parcellation Stage 4: A/B comparison + A+B fusion
 │    Three final label sets, all evaluated vs gold standard
 │    Functional decoding validation (LIP→saccade, MIP→reach)
 │
 ╞══ PARCELLATION DONE ════════════════════════════════════════════
 │
 └─ Competition deliverables
      Pipeline figure, parcellation overlays, Dice bar charts,
      cross-session consistency heatmaps, ROI decoding accuracy comparison
```

### Success criteria
- **Registration**: DL output (Stage 1) consistently outperforms Stage 3B rigid baseline on test set (NCC, SSIM)
- **Parcellation Plan A**: atlas-derived labels match gold standard with Dice > 0.6 per region
- **Parcellation Plan B**: data-driven clusters align with known functional anatomy (ARI > 0.4 vs gold standard)
- **Fusion A+B**: outperforms both standalone plans on at least 4/7 regions
- **Functional validation**: ROI-based decoding accuracy (LIP for saccade, MIP for reach) consistent with paper's searchlight results

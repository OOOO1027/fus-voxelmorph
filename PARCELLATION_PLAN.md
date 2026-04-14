# fUS Brain Region Parcellation Plan

## Overview

配准完成后的下游任务：将跨 session 对齐后的 fUS 图像划分为功能性脑区。
**三条产出线并行推进**，最终各自独立出结果，并融合比较：

- **Plan A**: Atlas-based — 定位 atlas 切片 + 2D 配准 + 标签迁移 → 产出 A 方案划分结果
- **Plan B**: Data-driven — 无监督聚类 fDOP 时序特征 → 产出 B 方案划分结果
- **Plan A+B**: 融合方案 — 将 A、B 结果交叉对比 + 加权融合 → 产出综合最优划分结果

A 方案相对更易落地，优先推进；B 方案同步开展；最后 A+B 融合看能否得到更好的划分。

---

## 已知约束与输入

### fUS 数据特征
- 2D 冠状面（coronal），FOV ~12.8mm x 10-13mm，分辨率 100um/pixel
- Power Doppler 图 (neurovascular_map): 128 x 127 或 128 x 132 像素
- 后顶叶皮层 (PPC) 区域，包含: LIP, MIP, VIP, PIP, area 5, area 7a, 7op
- 关键解剖 landmark: 顶内沟 (IPS, intraparietal sulcus) 在 fUS 中清晰可见
- 每个 session 有 fDOP 时间序列 (z x x x n 帧, ~2Hz)

### 已有参考与金标准
- **金标准 (Ground Truth)**: PPT (定位.pptx) 中的手工分割标注，由专业研究员完成，是最终评判所有方案的参考依据
- PPT 中同时标注了 Calabrese et al. (2015) DTI atlas (Paxinos 脑区命名)
- Scalable Brain Atlas 中定位到 slice ~113 (-22.8mm ant. ac)
- 论文 Fig.1 已人工标注了 Monkey P/L 的 fUS 平面上的脑区

### 论文中的脑区证据
- 2-target saccade decoding: 最信息性体素在 dorsal LIP + area 7a
- 8-target saccade decoding: 更大范围 LIP (含 ventral LIP)
- 2-target reach decoding: dorsal LIP + MIP + area 5 + area 7a/tpt
- Searchlight 分析提供了每个体素的解码精度 → 可作为功能性分区的 ground truth

---

## Plan A: Atlas-Based Slice Localization + Parcellation

### A1. Atlas 数据获取

**数据源**: Calabrese et al. (2015) DTI-based macaque brain atlas
- 下载地址: Scalable Brain Atlas (scalablebrainatlas.incf.org) 或原始发布
- 格式: NIfTI (.nii.gz), 包含:
  - T1/T2 模板卷 (3D, ~0.5mm 各向同性)
  - DTI 模板卷 (FA, MD 等)
  - 标签卷 (Paxinos parcellation, 整数标签)
  - 标签查找表 (label ID -> region name)

**需要提取的信息**:
```
atlas_t1.nii.gz          # 解剖模板
atlas_labels.nii.gz      # 分区标签
atlas_lut.csv            # label_id, region_name, color
```

**工具**: `nibabel` 读取 NIfTI, `numpy` 提取冠状切片

### A2. Slice-to-Volume Localization (创新点)

**问题**: 给定一张 2D fUS neurovascular_map，确定它对应 atlas 中的哪一层冠状切片。

**难点**: fUS 是 Power Doppler (血管信号)，atlas 是 DTI/T1 (组织结构) — 模态差异大。

**策略: 基于结构 landmark 的多尺度匹配**

#### Step 1: Atlas 预处理
```python
# 对 atlas 的每个冠状切片提取结构特征
for slice_idx in range(n_coronal_slices):
    atlas_slice = atlas_volume[:, slice_idx, :]  # (D, W)
    label_slice = label_volume[:, slice_idx, :]

    # 提取特征:
    # 1. 边缘图 (Sobel/Canny) — 灰白质边界近似血管分布
    # 2. IPS 位置和走向 (从 label 中提取 IPS 区域)
    # 3. 区域组成 (哪些 Paxinos 区域出现在此切片中)
    # 4. 外轮廓形状
```

#### Step 2: fUS 特征提取
```python
# 从 neurovascular_map 提取匹配特征
fus_map = load_neurovascular_map(session)

# 1. 血管骨架提取 (形态学细化)
# 2. IPS 定位 (fUS 中 IPS 是一条亮的纵向血管带)
# 3. 大脑外轮廓
# 4. 主要血管分布的空间统计量 (质心、方向、密度梯度)
```

#### Step 3: 切片搜索
```python
# 在 atlas 的冠状切片中搜索最佳匹配
# 搜索范围: 根据手术记录/PPT标注缩小到 ±5mm (~10 slices)
# 相似度: 加权组合
score = w1 * ips_position_similarity   # IPS 位置匹配
      + w2 * edge_ncc                   # 边缘图 NCC
      + w3 * contour_hausdorff          # 轮廓形状
      + w4 * region_overlap             # 预期区域组成

best_slice = argmax(score)
```

**备选简化方案**: 如果自动定位效果不好，使用 PPT 中已标注的 slice ~113 作为先验，
在 ±3 slices 范围内微调。论文中 monkey P 和 L 各有固定的 imaging plane，
这个先验很强。

**配准管线带来的简化**: 因为所有 session 已通过 VoxelMorph 对齐到 common space，
**只需对 reference session 做一次 slice localization**，结果自动适用于所有其他 session。

### A3. 2D fUS-Atlas Registration

**输入**: fUS neurovascular_map (128xW) 和 atlas 最佳匹配切片 (resampled)

**模态桥接问题**: Power Doppler ≠ T1/DTI，直接配准困难。

**策略: 中间表示 (intermediate representation)**

```
方案 A3-1: 边缘到边缘配准
  fUS → Sobel edge → skeleton
  Atlas → Sobel edge → skeleton
  配准两个 edge map (模态无关)

方案 A3-2: 分段对应点配准
  在 fUS 和 atlas 上各标注 5-8 个 landmark:
  - IPS 上下端点 (2)
  - 脑表面 3-4 个特征点
  - 沟底最深点 (1)
  用 TPS (thin-plate spline) 或 affine + 局部变形
  
方案 A3-3: 用 VoxelMorph 配准 (复用已有模型)
  将 atlas edge map 作为 "moving"
  将 fUS edge map 作为 "fixed"
  用已训练的 VoxelMorph 预测 warp field
  (注意: 模型在血管图上训练，edge map 分布可能不同，需要 fine-tuning 或单独训练)
```

**推荐**: 先用 A3-2 (landmark) 快速出结果，再用 A3-1 (edge) 自动化。

**关键简化**: 只需对 reference session 做 atlas↔fUS 配准一次。其余 session 的标签通过
**warp 级联**获得: `atlas → (atlas配准) → ref_session → (VoxelMorph warp) → target_session`。
这避免了对每个 session 独立做跨模态配准，也让 DL 配准的质量直接转化为标签迁移的精度。

### A4. Label Transfer

```python
# 将 atlas 标签通过 warp field 映射到 fUS 空间
label_slice_resampled = resample_atlas_slice(atlas_labels, best_slice, fus_shape)
fus_labels = apply_warp(label_slice_resampled, warp_field, mode='nearest')
# nearest-neighbor 插值保持标签离散性

# 输出: fUS 空间中的脑区标签图
# 每个像素 -> region_id (LIP=1, MIP=2, VIP=3, area5=4, area7a=5, ...)
```

### A5. 验证 (对照金标准)

- **与金标准对比 (首要指标)**: 将 Plan A 输出的脑区边界与 PPT 中专业研究员手工分割结果逐区域对比 (Dice/IoU)，这是最权威的评判依据
- **与论文标注对比**: 论文 Fig.1 和 searchlight 图提供辅助参考
- **Searchlight 一致性**: 论文中 saccade decoder 最信息性体素应落在 atlas-derived LIP 区域内
- **跨 session 一致性**: 配准后不同 session 的标签应高度重合

---

## Plan B: Data-Driven fUS Parcellation

### B1. 功能特征提取

**前置步骤**: 用 VoxelMorph 推理出的 warp field 将每个 session 的 dop 时序对齐到 common space。
对齐后，不同 session 的同一像素坐标对应同一解剖位置，才能做有意义的跨 session 聚合。

对每个 session（已对齐到 common space）的 fDOP 时间序列 (z x x x n)，在每个像素上提取时序特征:

```python
# 基础统计量
temporal_mean = np.mean(dop, axis=2)       # 平均 Power Doppler
temporal_std = np.std(dop, axis=2)         # 时间变异性
temporal_snr = temporal_mean / (temporal_std + eps)

# Trial-averaged responses (利用 behavior + state_num)
for direction in unique_directions:
    trial_mask = (actual_labels == direction) & (state_num == MEMORY)
    trial_avg = np.mean(dop[:, :, trial_mask], axis=2)
    # shape: (z, x) per direction

# 时间相关矩阵
# 对每个像素对 (i,j), 计算其时序相关性
# 下采样到 ~1000 ROI 做相关矩阵 (full pixel 太大)

# 方向选择性指标
# 对每个像素, 计算 8 个方向的 trial-averaged response
# tuning_strength = max(response) - min(response)
# preferred_direction = argmax(response)
```

### B2. 无监督聚类

```python
# 方法 1: Spectral Clustering (推荐)
# 基于像素间时序相关性的图分割
from sklearn.cluster import SpectralClustering

# 构建相似度矩阵 (基于时序相关性)
# 使用 k-nearest-neighbor 图稀疏化
n_clusters = 6  # 预期 ~6 个主要功能区 (LIP, MIP, VIP, area5, area7a, 7op)

# 方法 2: K-Means on feature vectors
feature_vector = np.stack([
    temporal_mean, temporal_std, temporal_snr,
    *[trial_avg_dir_i for i in range(n_directions)],
    tuning_strength, preferred_direction,
], axis=-1)  # (z, x, n_features)

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=n_clusters)
labels = kmeans.fit_predict(feature_vector.reshape(-1, n_features))
labels = labels.reshape(z, x)

# 方法 3: Hierarchical clustering
# 更好的层级结构 (e.g., PPC -> dorsal/ventral -> LIP/MIP/VIP)
from scipy.cluster.hierarchy import fcluster, linkage

# 方法 4: NMF (Non-negative Matrix Factorization)
# 时间维度降维, 空间分量作为功能网络
from sklearn.decomposition import NMF
# dop_2d: (n_pixels, n_frames)
nmf = NMF(n_components=n_clusters)
spatial_components = nmf.fit_transform(dop_2d)  # (n_pixels, n_components)
temporal_components = nmf.components_             # (n_components, n_frames)
# 每个 spatial component 可视为一个功能网络
```

### B3. 最优聚类数选择

```python
# Silhouette score
from sklearn.metrics import silhouette_score
for k in range(3, 12):
    labels = cluster(features, n_clusters=k)
    score = silhouette_score(features, labels)

# 也可用 Calinski-Harabasz index, Davies-Bouldin index
# 以及领域知识 (预期 ~5-8 个区域)
```

### B4. 跨 Session 一致性验证

```python
# 在配准后的空间中比较不同 session 的聚类结果
for session_pair in registered_pairs:
    labels_A = cluster(session_A_features)
    labels_B = cluster(session_B_features)
    # 用 Hungarian algorithm 对齐 cluster ID
    # 计算 Dice/IoU overlap
    consistency = adjusted_rand_index(labels_A_aligned, labels_B_aligned)
```

### B5. 独立验证 (对照金标准) + A/B 交叉对比

```python
# ---- Step 1: Plan B 独立对照金标准 ----
gt_labels = load_ground_truth_from_ppt()  # 专业研究员手工分割
data_labels = plan_b_results[session]
# Dice/IoU per region
for region in ['LIP', 'MIP', 'VIP', 'area5', 'area7a', '7op']:
    dice = compute_dice(gt_labels == region_id, data_labels == cluster_id)

# ---- Step 2: Plan A vs Plan B 交叉对比 ----
atlas_labels = plan_a_results[session]

# 定量: Adjusted Rand Index, Normalized Mutual Information
ari = adjusted_rand_index(atlas_labels.ravel(), data_labels.ravel())
nmi = normalized_mutual_info(atlas_labels.ravel(), data_labels.ravel())

# 定性: 三方叠加可视化 (金标准 + Plan A + Plan B)
fig, axes = plt.subplots(1, 4)
axes[0].imshow(neurovascular_map, cmap='hot')
axes[0].contour(gt_labels, colors='white', linewidths=1.0)
axes[0].set_title('Gold standard (manual)')

axes[1].imshow(neurovascular_map, cmap='hot')
axes[1].contour(atlas_labels, colors='cyan', linewidths=0.5)
axes[1].set_title('Atlas parcellation (Plan A)')

axes[2].imshow(neurovascular_map, cmap='hot')
axes[2].contour(data_labels, colors='yellow', linewidths=0.5)
axes[2].set_title('Data-driven parcellation (Plan B)')

axes[3].imshow(neurovascular_map, cmap='hot')
axes[3].contour(gt_labels, colors='white', linewidths=1.0)
axes[3].contour(atlas_labels, colors='cyan', linewidths=0.5)
axes[3].contour(data_labels, colors='yellow', linewidths=0.5, linestyles='dashed')
axes[3].set_title('All overlay')
```

---

## Implementation Roadmap

### Stage 1: Atlas Infrastructure (1-2 days)

```
Task 1.1: Download Calabrese 2015 atlas
  - Source: scalablebrainatlas.incf.org or original publication
  - Extract NIfTI volumes + label LUT
  - Verify coronal orientation matches fUS

Task 1.2: Atlas exploration script
  - scripts/explore_atlas.py
  - 可视化冠状切片序列
  - 标注 PPC 区域的切片范围
  - 提取 IPS、LIP、MIP 等区域的位置

Task 1.3: fUS landmark extraction
  - scripts/extract_fus_landmarks.py
  - IPS 自动检测 (亮带, 纵向结构)
  - 脑表面轮廓提取
  - 主要血管骨架
```

### Stage 2: Plan A — Slice Localization + Registration (2-3 days)

```
Task 2.1: Slice localization prototype
  - scripts/slice_localization.py
  - 先用 PPT 标注的 slice ~113 作为先验
  - 在 ±5 slices 范围内用 edge NCC 搜索
  - 验证: 对所有 8 个 session 定位, 结果应一致 (同一 imaging plane)

Task 2.2: 2D cross-modal registration
  - scripts/atlas_fus_registration.py
  - 先实现 landmark-based (A3-2), 快速验证
  - 再尝试 edge-to-edge (A3-1) 自动化
  - 输出 warp field

Task 2.3: Label transfer + visualization
  - scripts/transfer_labels.py
  - Apply warp to atlas labels
  - Overlay on neurovascular_map
  - 对所有 session 生成标签图
```

### Stage 3: Plan B — Data-Driven Parcellation (与 Stage 2 并行, 2-3 days)

**注意: Stage 2 和 Stage 3 同步推进，互不依赖。**

```
Task 3.1: Feature extraction
  - scripts/extract_functional_features.py
  - 从 .mat 的 dop + behavior + state_num 提取
  - Trial-averaged maps per direction
  - Temporal statistics per pixel

Task 3.2: Clustering experiments
  - scripts/data_driven_parcellation.py
  - Spectral clustering (primary)
  - K-means, NMF (comparison)
  - 扫描 k=3..10, 选最优

Task 3.3: Plan B 独立评估
  - 对照金标准 (PPT 手工分割) 计算 Dice/IoU per region
  - 跨 session 一致性 (ARI)
  - Plan B 独立产出最终聚类标签图
```

### Stage 4: A/B 交叉对比 + A+B 融合 (1-2 days)

**三条产出线在此汇合，产出三份独立结果 + 综合分析。**

```
Task 4.1: A/B 交叉对比
  - scripts/parcellation_comparison.py
  - Plan A vs Plan B: ARI, NMI 定量指标
  - Plan A vs 金标准, Plan B vs 金标准: Dice/IoU per region
  - 三方叠加可视化 (金标准 + Plan A + Plan B)

Task 4.2: A+B 融合方案
  - scripts/fuse_parcellations.py
  - 策略 1: 加权投票 — Plan A 和 Plan B 标签在每个像素上投票,
    权重由各自与金标准的 Dice 决定
  - 策略 2: 边界共识 — 取 A、B 一致的边界 (高置信),
    不一致区域由金标准更近的方案决定
  - 策略 3: Plan A 提供区域先验 + Plan B 细调边界位置
  - 对照金标准评估融合方案是否优于单独 A 或 B

Task 4.3: 三套最终结果输出
  - parcellation/plan_a/final/  → Plan A 最终标签图 (每个 session)
  - parcellation/plan_b/final/  → Plan B 最终标签图 (每个 session)
  - parcellation/fused/final/   → A+B 融合标签图 (每个 session)
  - 每套结果都附带 vs 金标准的定量评估

Task 4.4: 功能解码验证
  - 用三套标签分别提取 ROI 平均信号
  - 验证: LIP ROI 的 saccade 方向选择性
  - 验证: MIP ROI 的 reach 方向选择性
  - 对比论文的 searchlight 结果
  - 哪套标签的 ROI 解码精度最高？→ 最终推荐

Task 4.5: Competition 图表
  - Pipeline 全流程示意图
  - 三套脑区分区叠加在 neurovascular_map 上
  - Plan A vs Plan B vs A+B 融合 三方对比图
  - 各方案 vs 金标准的 Dice 柱状图
  - 跨 session 一致性热力图
```

---

## Key Technical Decisions

### 1. Atlas 选择
Calabrese 2015 是最适合的，因为:
- 猕猴脑 (Macaca mulatta) 种属一致
- Paxinos 命名系统与论文一致
- 包含 PPC 的详细分区 (LIP, MIP, VIP, area 5, 7a 等)
- 有 3D NIfTI 格式可供切片提取
- 在 Scalable Brain Atlas 上有交互式查看

### 2. 跨模态匹配策略
fUS (Power Doppler) 与 DTI/T1 之间的模态鸿沟是最大挑战:
- **不用直接强度匹配** — 两种模态的像素值物理意义完全不同
- **用结构 landmark 桥接** — IPS 在两种模态中都可识别
- **用边缘/形态特征** — 减少模态依赖
- 论文中已有手动标注结果可作为 sanity check

### 3. Plan B 的聚类数
- 论文确认 PPC 内的主要功能区: LIP (dorsal/ventral), MIP, VIP, area 5, area 7a, 7op
- 预期 k = 5-8
- 但 fUS 空间分辨率 100um 可能无法分辨所有子区
- 实际 k 由 silhouette score + 领域知识共同决定

### 4. 与 DL 配准管线的衔接 (核心依赖)

配准阶段的产出物是划分阶段的**直接输入**，不是可选项：

**配准阶段产出物**:
| 产出物 | 路径 | 说明 |
|--------|------|------|
| 训练好的模型 | `checkpoints/best_model.pth` | VxmDense2D, 可推理任意 session pair |
| Session pair warp fields | `evaluate.py` 输出 | 每对 session 间的形变场 φ |
| Warped neurovascular maps | evaluate 输出 | 对齐后的 neurovascular map |
| Rigid-prealigned images | `dl_prep/bundles/` | Stage 3B 刚性对齐后的图像 |

**在 Plan A 中的具体使用**:
- **A2 (Slice Localization)**: 只需定位一次 — 在配准后的 common space 中定位 atlas 切片，所有 session 共享同一定位结果（因为已经对齐到同一空间了）
- **A3 (Atlas-fUS Registration)**: 方案 A3-3 直接复用 VoxelMorph 推理 atlas edge map → fUS edge map 的 warp field；即使用 landmark/edge 方案，也只需对 reference session 做一次 atlas 配准，其余 session 通过 VoxelMorph 的 cross-session warp field 级联传递标签
- **A4 (Label Transfer)**: `atlas → ref_session (atlas配准) → target_session (VoxelMorph warp)` — 两步 warp 复合，不必对每个 session 独立做 atlas 配准

**在 Plan B 中的具体使用**:
- **B1 (Feature Extraction)**: 每个 session 的 dop 时序需先用 VoxelMorph warp field 对齐到 common space，才能跨 session 聚合特征做聚类
- **B2 (Clustering)**: 在 common space 中聚类 — 保证不同 session 的同一像素对应同一解剖位置
- **B4 (Cross-session Consistency)**: 配准质量直接决定跨 session 一致性的上限；如果配准差，即使聚类算法完美也会显示低一致性
- **Warp field 本身作为特征**: 形变场的局部特性（Jacobian determinant, 形变梯度）可作为 Plan B 聚类的额外特征维度 — 形变模式一致的区域更可能属于同一脑区

**在 A+B 融合中的使用**:
- 配准的 common space 是三套结果对齐比较的基础
- 配准置信度（如 local NCC）可作为融合权重的空间调制 — 配准可信区域更信任 Plan B（跨 session 数据一致），配准不可信区域更信任 Plan A（不依赖跨 session 对齐）

**其他依赖**:
- 两个 Plan 都需要 Phase 1 trial-level 数据中的 behavior/state_num 元数据

---

## 文件结构

```
fus-voxelmorph/
├── atlas/                         # Atlas 数据 (新建)
│   ├── calabrese2015/
│   │   ├── atlas_t1.nii.gz
│   │   ├── atlas_labels.nii.gz
│   │   └── label_lut.csv
│   └── README.md
├── parcellation/                  # 分区结果 (新建)
│   ├── ground_truth/              # 金标准 (PPT 手工分割数字化)
│   │   ├── gt_labels.npy
│   │   └── gt_region_mapping.json
│   ├── plan_a/                    # Atlas-based results
│   │   ├── S1_atlas_labels.npy
│   │   ├── S1_warp_field.npy
│   │   ├── final/                 # Plan A 最终输出
│   │   └── ...
│   ├── plan_b/                    # Data-driven results
│   │   ├── S1_cluster_labels.npy
│   │   ├── S1_features.npz
│   │   ├── final/                 # Plan B 最终输出
│   │   └── ...
│   ├── fused/                     # A+B 融合结果
│   │   ├── final/                 # 融合方案最终输出
│   │   └── fusion_weights.json
│   ├── comparison/                # 三方对比 + vs 金标准
│   │   ├── dice_iou_vs_gt.csv     # 各方案 vs 金标准
│   │   ├── ari_nmi_ab.csv         # Plan A vs Plan B
│   │   └── overlay_figures/
│   └── manifests/
│       └── parcellation_summary.csv
├── scripts/
│   ├── explore_atlas.py           # Atlas 探索
│   ├── extract_fus_landmarks.py   # fUS landmark 提取
│   ├── slice_localization.py      # Slice-to-volume 定位
│   ├── atlas_fus_registration.py  # 2D 跨模态配准
│   ├── transfer_labels.py         # Atlas 标签迁移
│   ├── extract_functional_features.py  # fDOP 功能特征
│   ├── data_driven_parcellation.py     # 无监督聚类
│   ├── parcellation_comparison.py      # Plan A vs B vs 金标准 三方对比
│   └── fuse_parcellations.py           # A+B 融合方案
└── ...
```

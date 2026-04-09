# fUS-VoxelMorph 训练计划报告 v1
**日期：** 2026-04-06  
**项目：** 猴子 fUS 跨 Session 非刚性配准（生医工比赛）  
**状态：** 数据就绪，待启动首轮训练

---

## 1. 任务定义

**任务：** 跨 Session 解剖学图像配准（anatomy-only, image-to-image）  
**输入：** `neurovascular_map`（神经血管图）  
**目标：** DL 非刚性残差配准，证明超过 Stage 3B 刚体 backbone  
**比赛约束：** 仅 24 对数据，不支撑强泛化结论，定位为 dataset-specific 原型

**训练模式：** Mode B（rigid-prealigned）
- `moving_rigid`（Stage 3B 刚性预对齐结果） + `fixed_raw` 作为输入
- 保留 Stage 3B 的刚性 backbone 优势，DL 学习残差非刚性形变

---

## 2. 数据集统计

### 2.1 数据集规模

| 子集 | 配对数 | 无向组数 | 涉及 Session |
|------|--------|----------|-------------|
| train | 16 | 8 | S1, S2, S3, S5, S7, S9, S14, S16 |
| val   | 4  | 2 | S2, S5, S14, S16 |
| test  | 4  | 2 | S2, S3, S9, S14  |
| **合计** | **24** | **12** | **8 Sessions** |

**形状分布（两个 shape cohort 均衡）：**
- train: `(128,127)` × 8 对，`(128,132)` × 8 对
- val:   `(128,127)` × 2 对，`(128,132)` × 2 对
- test:  `(128,127)` × 2 对，`(128,132)` × 2 对

**canonical canvas：** `128 × 132`（padded NPZ 统一尺寸）

### 2.2 难例分布

| 子集 | 总对数 | 难例对数 | 难例类型 |
|------|--------|----------|---------|
| train | 16 | 3 | stage3b_tier2_diagnostic |
| val   | 4  | 2 | methodological_drop + tier2_diagnostic |
| test  | 4  | 3 | methodological_drop（×2）+ tier2_diagnostic |

> **注意：** test 集包含 2 个 methodological drop 对（`S14↔S9`），Stage 3B 在这两对上 NCC 不升反降，是 DL 方法的潜在得分点。

### 2.3 分割泄漏控制

- 分割单元：无向 pair group（同一无向组的两个有向对强制同组）
- 无有向对跨分割泄漏
- 未启用更严格的 session-aware holdout（proposal-only，v2 考虑）

---

## 3. Baseline 基准性能

基准方法：Stage 3B 刚性 backbone（rigid-only registration）  
指标来源：`baseline_reference.csv`，mask-aware 计算

### 3.1 全集（24 对）

| 指标 | 配准前 (pre) | Stage 3B | 提升 |
|------|-------------|----------|------|
| NCC  | 0.781       | 0.843    | **+0.062** |
| Edge NCC | 0.694   | 0.756    | **+0.062** |

### 3.2 分 split 统计

| Split | pre NCC | Stage3B NCC | delta |
|-------|---------|-------------|-------|
| train (16) | 0.790 | 0.869 | +0.079 |
| val   (4)  | 0.733 | 0.795 | +0.062 |
| test  (4)  | 0.795 | **0.784** | **−0.011** |

> **关键发现：** test 集 Stage 3B 均值低于 pre（因 methodological drop 对失败），
> 说明 DL 方法只需在这两对上部分恢复即可超越 Stage 3B 的 test NCC。
> **DL 的竞争目标是 val NCC > 0.795，test NCC > 0.784。**

### 3.3 逐对 baseline（val + test 细节）

**Val 集：**

| pair_id | pre NCC | Stage3B NCC | 难例 |
|---------|---------|-------------|------|
| S14→S16 | 0.860 | 0.897 | — |
| S16→S14 | 0.860 | 0.856 | methodological_drop + tier2 |
| S2→S5  | 0.607 | 0.726 | tier2_diagnostic |
| S5→S2  | 0.607 | 0.700 | — |

**Test 集：**

| pair_id | pre NCC | Stage3B NCC | 难例 |
|---------|---------|-------------|------|
| S14→S9 | 0.735 | 0.721 | **methodological_drop** |
| S9→S14 | 0.735 | 0.698 | **methodological_drop** |
| S2→S3  | 0.855 | 0.858 | — |
| S3→S2  | 0.855 | 0.860 | tier2_diagnostic |

---

## 4. 模型架构

**VxmDense2D**（`models/vxm2d.py`）

| 参数 | 值 |
|------|----|
| 输入通道 | 1（灰度，neurovascular_map） |
| 编码器 | [16, 32, 32, 32] |
| 解码器 | [32, 32, 32, 32, 16, 16] |
| 积分步数 | 7（diffeomorphic，拓扑保持） |
| 双向训练 | 是（bidir=True，有效翻倍配对） |
| 参数量 | ~91K |

---

## 5. 训练配置（cross_session.yaml）

### 5.1 预处理

| 配置项 | 值 | 说明 |
|--------|----|------|
| 归一化 | percentile [1, 99] | 抑制 fUS 极值噪声 |
| log_transform | 是 | 压缩 Power Doppler 偏斜分布 |
| 使用 padded NPZ | 是 | 统一 canvas 128×132 |

### 5.2 数据增强

| 增强类型 | 参数 | 概率 |
|----------|------|------|
| 随机仿射 | 旋转±5°, 平移±5px, 缩放0.95-1.05 | 0.5 |
| 强度抖动 | 噪声std=0.02, 乘性0.95-1.05, 亮度±0.05 | 0.5 |
| 随机裁剪 | scale 0.9-1.0 | 0.3 |
| **弹性变形** | grid=5, magnitude=4px（B-spline风格） | 0.5 |
| 水平翻转 | — | 0.5 |

### 5.3 损失函数

| 项 | 配置 |
|----|------|
| 相似度 | NCC，window=9 |
| 正则化 | Gradient L2，weight=1.0 |
| 双向权重 | 1.0 |
| mask-aware | 是（避免 padding 区域干扰） |

### 5.4 训练超参数

| 参数 | 值 |
|------|----|
| epochs | 500 |
| batch_size | 4 |
| lr | 1e-4 |
| lr_scheduler | warmup_cosine（warmup=20 epoch） |
| early_stopping patience | 80 |
| checkpoint 保存间隔 | 50 epoch |
| 设备 | cuda |

---

## 6. 评估协议

### 6.1 三方对比

每对评估三个状态：

| 状态 | 说明 |
|------|------|
| pre | 配准前（原始移动图像） |
| stage3b | Stage 3B 刚体 backbone 结果 |
| **dl_output** | **本 DL 模型输出** |

### 6.2 主指标

- **mask-aware NCC**（neurovascular_map）
- **mask-aware Edge NCC**（neurovascular_map）

### 6.3 可视化

每对输出：overlay / checkerboard / absolute_difference / triptych

### 6.4 输出文件

- `results/cross_session/three_way_comparison.csv`
- 每对：`warped_*.npy`，`flow_*.npy`

---

## 7. 训练命令

```bash
# 首轮训练（Mode B，推荐起点）
python train_v2.py --config configs/cross_session.yaml

# 评估（三方对比）
python evaluate.py --config configs/cross_session.yaml \
    --checkpoint checkpoints/cross_session/best_model.pth

# 结果汇总
cat results/cross_session/three_way_comparison.csv
```

---

## 8. 超参数调优优先级

数据量仅 16 对训练，以下参数对结果影响最大：

1. **`reg_weight`**（正则化权重）：初始值 1.0，若形变场过于平滑则降低（尝试 0.1, 0.5），过于激进则升高（2.0, 5.0）
2. **`ncc_win_size`**（NCC 窗口）：初始 9，对 fUS 纹理稀疏特征可尝试 7 或 11
3. **`lr`**（学习率）：若 val loss 震荡则降至 5e-5
4. **`elastic.magnitude`**（弹性变形幅度）：若小数据过拟合则增大（6-8px）

---

## 9. 就绪清单

- [x] 24 对 NPZ 训练资产已导入（native + padded_canonical）
- [x] `pair_dataset.csv` 及 train/val/test splits 已冻结
- [x] baseline_reference.csv（Stage 3B NCC/Edge NCC）已记录
- [x] `models/vxm2d.py` VxmDense2D 模型已完成
- [x] `losses/losses.py` mask-aware NCC/Grad 损失已完成
- [x] `utils/metrics.py` 全套评估指标已完成
- [x] `data/cross_session_dataset.py` Mode B 数据加载已完成
- [x] `train_v2.py` 完整训练流程已完成
- [x] `evaluate.py` 三方对比评估已完成
- [x] `configs/cross_session.yaml` 主配置已完成
- [ ] **首轮训练启动**
- [ ] val NCC > 0.795（超越 Stage 3B val baseline）
- [ ] test NCC > 0.784（超越 Stage 3B test baseline）
- [ ] 三方对比报告生成

---

## 10. 风险与注意事项

| 风险 | 缓解措施 |
|------|----------|
| 训练集仅 16 对，严重过拟合风险 | bidir 翻倍 + 弹性变形增广 + early stopping |
| Test 含 2 个 methodological drop 对 | 这是 DL 得分机会，Stage 3B 在此失败 |
| val 包含 1 个 hard case（S2↔S5，pre NCC 仅 0.61） | 重点监控 S2↔S5 的改善曲线 |
| Stage 3B 不是 ground truth | 无法评估绝对形变准确性，只评估相对图像相似度改善 |
| shape cohort 不同（127 vs 132 宽） | padded NPZ 已统一为 128×132，valid_mask 保护 |

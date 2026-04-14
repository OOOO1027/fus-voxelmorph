# Plan vs Summary 对齐审计报告

**结论：plan 与 summary 描述的是同一套数据，但不是同一口径。** plan 中的 pre NCC 来自 baseline_reference.csv（原始像素空间计算），summary 中的 pre NCC 经过了 log_transform + percentile 归一化后重新计算，而该预处理链存在 bug，导致 (128,132) shape group 的 12 个 pair 被压缩为全零图像。

---

## 1. 差异全貌

| 文档 | pre NCC 来源 | 计算空间 | 受 bug 影响？ |
|------|-------------|----------|--------------|
| training_plan_v1 | baseline_reference.csv | 原始像素值 | 否 |
| training_summary | evaluate.py 实时计算 | log + percentile 归一化后 | **是** |

---

## 2. 核心 Bug 分析

### 2.1 Bug 触发链

```
原始数据 (128,132) group: 所有像素值 ∈ [5.2, 56.0], 无零值
    ↓
log_transform(data, epsilon=1e-6):
    log(1 + x/1e-6) / log(1 + 1/1e-6)
    → 所有值映射到 ~0.99999+（因 x >> epsilon）
    ↓
normalize_frame(data, method='percentile', percentile=(1,99)):
    p1 ≈ 0.99999, p99 ≈ 1.00000
    p_high - p_low < 1e-8 → 触发零填充分支
    → 返回全零图像
```

### 2.2 为什么只影响 (128,132) group？

| Shape Group | Sessions | 最小像素值 | 有 padding 零值？ | log 后范围 | percentile 结果 |
|-------------|----------|-----------|-----------------|-----------|----------------|
| (128,132) | S1, S2, S3, S5 | ~5.2 | 否 | [0.9999+, 1.0] | p1≈p99 → **全零** |
| (128,127) | S7, S9, S14, S16 | 0.0 (padding) | 是 | [0.0, 1.0] | p1~0, p99~1 → **正常** |

### 2.3 受影响 Pair 完整清单（12/24 对）

受影响的所有 pair 至少一端涉及 S1/S2/S3/S5 session，其中 fixed 或 moving 图像来自 (128,132) group 且无 padding 零值：

| pair_id | 哪端受影响 | 训练/评估后果 |
|---------|-----------|-------------|
| S2→S5 | **双端** | 全零输入→全零输出, NCC=0 |
| S5→S2 | **双端** | 同上 |
| S2→S3 | **双端** | 同上 |
| S3→S2 | **双端** | 同上 |
| S1→S2 | **双端** | 全零输入→全零输出 |
| S2→S1 | **双端** | 同上 |
| S1→S3 | **双端** | 同上 |
| S3→S1 | **双端** | 同上 |
| S1→S5 | **双端** | 同上 |
| S5→S1 | **双端** | 同上 |
| S3→S5 | **双端** | 同上 |
| S5→S3 | **双端** | 同上 |

跨 group 的 pair（如 S14→S9, S7→S16）两端均属于 (128,127) group，有 padding 零值，不受影响。

---

## 3. 逐 Pair 数值对比（重点 4 对）

### S2→S5

| 指标 | Plan (baseline_reference) | Summary (log+percentile) | 差异原因 |
|------|--------------------------|--------------------------|---------|
| pre NCC | 0.607 | 0.0 | fixed (S5) 经 log+percentile 后全零 |
| Stage3B NCC | 0.726 | 0.726 | Stage3B NCC 在 summary 中取自 baseline_reference |
| DL NCC | — | 0.0 | 模型在全零图像上训练/推理，输出全零 |

### S5→S2

| 指标 | Plan | Summary | 差异原因 |
|------|------|---------|---------|
| pre NCC | 0.607 | 0.0 | fixed (S2) 全零 |
| Stage3B NCC | 0.700 | 0.700 | 来自 baseline_reference |
| DL NCC | — | 0.0 | 同上 |

### S2→S3

| 指标 | Plan | Summary | 差异原因 |
|------|------|---------|---------|
| pre NCC | 0.855 | 0.0 | 双端全零 |
| Stage3B NCC | 0.858 | 0.858 | 来自 baseline_reference |
| DL NCC | — | 0.0 | 同上 |

### S3→S2

| 指标 | Plan | Summary | 差异原因 |
|------|------|---------|---------|
| pre NCC | 0.855 | 0.0 | 双端全零 |
| Stage3B NCC | 0.860 | 0.860 | 来自 baseline_reference |
| DL NCC | — | 0.0 | 同上 |

---

## 4. 回答用户核查问题

### A. 是否同一套数据？
**是。** 两份文档使用完全相同的 24 对 .npz 文件。

### B. pre NCC 计算口径是否一致？
**否。** Plan 使用 baseline_reference.csv 的原始空间 NCC，Summary 使用 log+percentile 预处理后的 NCC。这是数值差异的直接原因。

### C. 这 4 对在训练中是否参与了有效学习？
**否。** 这 4 对经预处理后变为全零图像输入，模型无法从中学到有效变形场。实际有效训练数据仅来自 (128,127) group 的 12 对。

### D. Summary 中的 DL NCC=0 是模型失败还是数据失败？
**数据失败。** 模型本身结构正确，但输入为全零图像时，任何模型都无法产生有意义输出。

### E. 修复后 DL 在这 4 对上的预期表现？
根据 Plan 中的 baseline：
- S2↔S5: pre NCC=0.607, Stage3B=0.700-0.726，DL 应有显著提升空间
- S2↔S3: pre NCC=0.855, Stage3B=0.858-0.860，DL 应能达到 0.95+ 水平（参考 (128,127) group 的表现）

---

## 5. Bug 位置与修复建议

### Bug 位置
- **`data/fus_dataset.py`**: `log_transform()` (L71-80) + `normalize_frame()` (L31-68)
- **`data/cross_session_dataset.py`**: `_preprocess()` (L143-156) 调用链

### 修复方案（推荐方案 1）

**方案 1: 调换 log 和 percentile 的顺序**
```python
def _preprocess(self, data):
    data = data.astype(np.float32)
    # 先 percentile 归一化（在原始值域上展开）
    if self.normalize:
        data = normalize_frame(data, method=self.normalize, percentile=self.percentile)
    # 再 log 变换（在 [0,1] 范围上压缩偏斜分布）
    if self.apply_log:
        data = apply_log_transform(data)
    return data
```

**方案 2: 修改 log_transform 的 epsilon**
将 epsilon 从 1e-6 改为与数据量级匹配的值（如 1.0 或 10.0），使 log 变换不会将所有值压到极窄范围。

**方案 3: 仅用 percentile 归一化，不用 log 变换**
fUS Power Doppler 数据值域 [5, 56] 已经比较温和，percentile 归一化足够。

### 推荐
**方案 1** 最安全，保留两种变换的各自优势，且改动最小。

---

## 6. 对训练结果的影响评估

| 影响范围 | 说明 |
|---------|------|
| 训练数据 | 16 对中约 6-8 对受影响（取决于具体哪些 pair 涉及 S1/S2/S3/S5） |
| 验证集 | 4 对中 2 对受影响 (S2↔S5) |
| 测试集 | 4 对中 2 对受影响 (S2↔S3) |
| Val/Test 膨胀 | 未受影响的 pair 报告 NCC 0.997+，但这仅代表一半数据的表现 |
| 模型质量 | 模型仅在约一半有效数据上训练，修复后重训可期待显著提升 |

---

**报告生成时间：** 2026-04-06
**审计范围：** training_plan_v1_20260406.md vs training_summary_cross_session_300epochs_20260406.md

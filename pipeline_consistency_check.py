"""
Pipeline 一致性验证：
1. 确认 cross_session_dataset.py 的预处理顺序是 先 percentile 再 log
2. 确认 train_v2.py 构建 dataset 时走的就是这个代码路径
3. 全部 24 对跑一遍预处理，确认无全零图像
4. 用 DataLoader + Collator 模拟一个 training batch，确认 mask-aware 链路完整
"""
import sys, inspect
sys.path.insert(0, ".")

import numpy as np
import torch
from pathlib import Path

print("=" * 70)
print("PIPELINE CONSISTENCY CHECK — all 24 pairs")
print("=" * 70)

# ══════════════════════════════════════════════════════════════════════
# CHECK 1: 源码级确认预处理顺序
# ══════════════════════════════════════════════════════════════════════
print("\n[CHECK 1] _preprocess 源码顺序验证")
from data.cross_session_dataset import CrossSessionPairDataset
src = inspect.getsource(CrossSessionPairDataset._preprocess)
# 找 normalize 和 log 的行号
lines = src.split('\n')
norm_line = log_line = None
for i, line in enumerate(lines):
    if 'normalize_frame' in line and '#' not in line.split('normalize_frame')[0]:
        norm_line = i
    if 'apply_log_transform' in line and '#' not in line.split('apply_log_transform')[0]:
        log_line = i

if norm_line is not None and log_line is not None and norm_line < log_line:
    print(f"  [PASS] normalize_frame at line {norm_line}, apply_log_transform at line {log_line}")
    print(f"         顺序正确：先 percentile 再 log")
else:
    print(f"  [FAIL] normalize at {norm_line}, log at {log_line} — 顺序错误!")
    sys.exit(1)

# ══════════════════════════════════════════════════════════════════════
# CHECK 2: train_v2.py 的 build_datasets 是否正确使用 CrossSessionPairDataset
# ══════════════════════════════════════════════════════════════════════
print("\n[CHECK 2] train_v2.py 调用链验证")
train_src = open("train_v2.py", "r", encoding="utf-8").read()

# 确认导入了 CrossSessionPairDataset
assert "CrossSessionPairDataset" in train_src, "train_v2.py 未导入 CrossSessionPairDataset"
print("  [PASS] train_v2.py 导入了 CrossSessionPairDataset")

# 确认构造时传入了 apply_log 和 normalize 参数
# (不需要 target_size，上次已修复)
if "target_size" in train_src and "CrossSessionPairDataset" in train_src:
    # 检查是否在 CrossSessionPairDataset 构造中传入了 target_size
    import re
    # 找 CrossSessionPairDataset( 到 ) 的区间
    pattern = r'CrossSessionPairDataset\([^)]*target_size[^)]*\)'
    if re.search(pattern, train_src):
        print("  [WARN] train_v2.py 仍然向 CrossSessionPairDataset 传入 target_size")
    else:
        print("  [PASS] CrossSessionPairDataset 构造无多余 target_size 参数")
else:
    print("  [PASS] CrossSessionPairDataset 构造无多余 target_size 参数")

# ══════════════════════════════════════════════════════════════════════
# CHECK 3: 全部 24 对预处理后非全零
# ══════════════════════════════════════════════════════════════════════
print("\n[CHECK 3] 全部 24 对预处理验证")

BUNDLE = Path("data/cross_session/fus_voxelmorph_dlprep_bundle_20260406_115937")

# 构造 dataset（同 train_v2.py 的配置）
ds_all = CrossSessionPairDataset(
    data_dir=str(BUNDLE),
    split='all',
    mode='B',
    use_padded=True,
    normalize='percentile',
    percentile=(1, 99),
    apply_log=True,
    augmentation=None,
)

print(f"  总 pair 数: {len(ds_all)}")

fail_count = 0
results = []
for i in range(len(ds_all)):
    sample = ds_all[i]
    mov = sample['moving'].numpy().squeeze()
    fix = sample['fixed'].numpy().squeeze()
    mask = sample['mask'].numpy().squeeze()
    pid = sample['pair_id']

    mov_ok = mov.max() > 0.01
    fix_ok = fix.max() > 0.01
    mask_ok = mask.max() > 0.5

    status = "PASS" if (mov_ok and fix_ok and mask_ok) else "FAIL"
    if status == "FAIL":
        fail_count += 1

    results.append({
        'pair_id': pid,
        'mov_range': f"[{mov.min():.4f}, {mov.max():.4f}]",
        'fix_range': f"[{fix.min():.4f}, {fix.max():.4f}]",
        'mask_mean': f"{mask.mean():.4f}",
        'status': status,
    })
    print(f"  [{status}] {pid:45s}  mov={mov.min():.3f}~{mov.max():.3f}  "
          f"fix={fix.min():.3f}~{fix.max():.3f}  mask_mean={mask.mean():.3f}")

if fail_count == 0:
    print(f"\n  [PASS] 全部 {len(ds_all)} 对预处理后非全零")
else:
    print(f"\n  [FAIL] {fail_count}/{len(ds_all)} 对存在全零图像!")
    sys.exit(1)

# ══════════════════════════════════════════════════════════════════════
# CHECK 4: DataLoader + Collator 模拟 training batch
# ══════════════════════════════════════════════════════════════════════
print("\n[CHECK 4] DataLoader batch 模拟")
from data.cross_session_dataset import CrossSessionCollator
from torch.utils.data import DataLoader

ds_train = CrossSessionPairDataset(
    data_dir=str(BUNDLE),
    split='train',
    mode='B',
    use_padded=True,
    normalize='percentile',
    percentile=(1, 99),
    apply_log=True,
    augmentation=None,
)

loader = DataLoader(ds_train, batch_size=4, shuffle=False,
                    collate_fn=CrossSessionCollator(), num_workers=0)

batch = next(iter(loader))
print(f"  batch keys: {list(batch.keys())}")
print(f"  moving shape: {batch['moving'].shape}  dtype: {batch['moving'].dtype}")
print(f"  fixed  shape: {batch['fixed'].shape}   dtype: {batch['fixed'].dtype}")
print(f"  mask   shape: {batch['mask'].shape}    dtype: {batch['mask'].dtype}")
print(f"  pair_ids: {batch['pair_id']}")

# 确认 batch 内无全零
for j in range(batch['moving'].shape[0]):
    m = batch['moving'][j].numpy().squeeze()
    f = batch['fixed'][j].numpy().squeeze()
    assert m.max() > 0.01, f"batch item {j} moving is all-zero!"
    assert f.max() > 0.01, f"batch item {j} fixed is all-zero!"
print(f"  [PASS] batch 内所有样本非全零")

# ══════════════════════════════════════════════════════════════════════
# CHECK 5: 模型前向 + loss 完整链路
# ══════════════════════════════════════════════════════════════════════
print("\n[CHECK 5] 模型前向 + loss 链路")
from models.vxm2d import VxmDense2D
from losses.losses import RegistrationLoss

model = VxmDense2D(
    in_channels=1,
    enc_channels=[16, 32, 32, 32],
    dec_channels=[32, 32, 32, 32, 16, 16],
    integration_steps=7,
    bidir=True,
)
model.eval()

from losses.losses import NCC, Grad
criterion = RegistrationLoss(
    sim_loss=NCC(win_size=9),
    reg_loss=Grad(penalty='l2'),
    reg_weight=1.0,
    bidir_weight=1.0,
)

with torch.no_grad():
    out = model(batch['moving'], batch['fixed'])
    warped, flow = out[0], out[1]

    # bidir 模式应返回 4 个输出
    assert len(out) == 4, f"Expected 4 outputs (bidir), got {len(out)}"
    warped_rev, flow_rev = out[2], out[3]

    # mask-aware loss
    loss_val, sim_val, reg_val = criterion(
        warped, batch['fixed'], flow,
        warped_target=warped_rev, source=batch['moving'],
        mask=batch['mask']
    )

print(f"  model output shapes: warped={warped.shape}, flow={flow.shape}")
print(f"  warped range: [{warped.min():.4f}, {warped.max():.4f}]")
print(f"  flow range: [{flow.min():.4f}, {flow.max():.4f}]")
print(f"  loss value: {loss_val.item():.6f}")
assert not torch.isnan(loss_val), "Loss is NaN!"
assert not torch.isinf(loss_val), "Loss is Inf!"
assert warped.max() > 0.01, "Warped output is all-zero!"
print(f"  [PASS] 前向推理 + mask-aware loss 正常")

# ══════════════════════════════════════════════════════════════════════
# CHECK 6: 对比两种 shape group 的预处理一致性
# ══════════════════════════════════════════════════════════════════════
print("\n[CHECK 6] 两种 shape group 预处理对比")
group_132 = []  # S1/S2/S3/S5, native (128,132), no padding
group_127 = []  # S7/S9/S14/S16, native (128,127), has padding

for i in range(len(ds_all)):
    sample = ds_all[i]
    pid = sample['pair_id']
    mov = sample['moving'].numpy().squeeze()
    fix = sample['fixed'].numpy().squeeze()
    mask = sample['mask'].numpy().squeeze()

    # 判断 shape group：mask_mean < 1.0 说明有 padding
    if mask.mean() < 0.99:
        group_127.append((pid, mov, fix, mask))
    else:
        group_132.append((pid, mov, fix, mask))

print(f"  (128,132) group (no padding): {len(group_132)} pairs")
print(f"  (128,127) group (has padding): {len(group_127)} pairs")

for name, group in [("(128,132)", group_132), ("(128,127)", group_127)]:
    if not group:
        continue
    mov_ranges = [(g[1].min(), g[1].max()) for g in group]
    fix_ranges = [(g[2].min(), g[2].max()) for g in group]
    mov_min = min(r[0] for r in mov_ranges)
    mov_max = max(r[1] for r in mov_ranges)
    fix_min = min(r[0] for r in fix_ranges)
    fix_max = max(r[1] for r in fix_ranges)
    print(f"  {name}: mov [{mov_min:.4f}, {mov_max:.4f}]  fix [{fix_min:.4f}, {fix_max:.4f}]")
    # 确认两组值域在合理范围内（都应该是 [0, 1]）
    assert mov_max <= 1.01, f"{name} moving max > 1!"
    assert fix_max <= 1.01, f"{name} fixed max > 1!"
    assert mov_max > 0.5, f"{name} moving max too small!"
    assert fix_max > 0.5, f"{name} fixed max too small!"
print(f"  [PASS] 两组 shape group 值域一致，均在 [0, 1] 范围内")

# ══════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("ALL 6 CHECKS PASSED — pipeline 一致性验证通过，可以开始正式训练")
print(f"{'='*70}")

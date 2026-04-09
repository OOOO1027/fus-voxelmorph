"""
Smoke test: 验证修复后 4 个 hard pair 预处理不再全零，pre NCC 非零。
还会加载 best_model 做一次前向推理，确认输出非零。
"""
import sys
import numpy as np
import torch
from pathlib import Path

# ── 路径 ──
BUNDLE = Path("data/cross_session/fus_voxelmorph_dlprep_bundle_20260406_115937")
NPZ_DIR = BUNDLE / "dl_prep" / "training_assets" / "npz" / "padded_canonical"
CKPT = Path("checkpoints/cross_session/best_model.pth")

HARD_PAIRS = [
    "previous_S2__current_S5__plane",
    "previous_S5__current_S2__plane",
    "previous_S2__current_S3__plane",
    "previous_S3__current_S2__plane",
]

# ── 导入预处理 ──
sys.path.insert(0, ".")
from data.fus_dataset import normalize_frame, log_transform as apply_log_transform


def preprocess_fixed(data):
    """修复后的预处理：先 percentile 再 log。"""
    data = data.astype(np.float32)
    data = normalize_frame(data, method='percentile', percentile=(1, 99))
    data = apply_log_transform(data)
    return data


def preprocess_old(data):
    """旧的 buggy 预处理：先 log 再 percentile。"""
    data = data.astype(np.float32)
    data = apply_log_transform(data)
    data = normalize_frame(data, method='percentile', percentile=(1, 99))
    return data


def ncc_numpy(a, b, win=9):
    """简易 NCC 计算。"""
    from scipy.ndimage import uniform_filter
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    a_mean = uniform_filter(a, win)
    b_mean = uniform_filter(b, win)
    a_std = np.sqrt(np.maximum(uniform_filter(a**2, win) - a_mean**2, 0))
    b_std = np.sqrt(np.maximum(uniform_filter(b**2, win) - b_mean**2, 0))
    ab_mean = uniform_filter(a * b, win)
    numer = ab_mean - a_mean * b_mean
    denom = a_std * b_std + 1e-8
    cc = numer / denom
    return float(cc.mean())


print("=" * 70)
print("SMOKE TEST: 4 hard pairs — preprocess fix verification")
print("=" * 70)

all_pass = True

for pid in HARD_PAIRS:
    npz_path = NPZ_DIR / f"{pid}.npz"
    if not npz_path.exists():
        print(f"\n[SKIP] {pid}: file not found")
        continue

    npz = np.load(str(npz_path), allow_pickle=True)
    moving = npz['moving_rigid']
    fixed = npz['fixed_raw']
    mask = npz['valid_mask']

    # ── 旧预处理 (buggy) ──
    mov_old = preprocess_old(moving)
    fix_old = preprocess_old(fixed)

    # ── 新预处理 (fixed) ──
    mov_new = preprocess_fixed(moving)
    fix_new = preprocess_fixed(fixed)

    ncc_old = ncc_numpy(mov_old, fix_old)
    ncc_new = ncc_numpy(mov_new, fix_new)

    print(f"\n{'─'*60}")
    print(f"  {pid}")
    print(f"{'─'*60}")
    print(f"  Raw range:  moving [{moving.min():.2f}, {moving.max():.2f}]  "
          f"fixed [{fixed.min():.2f}, {fixed.max():.2f}]")
    print(f"  OLD preproc: moving [{mov_old.min():.4f}, {mov_old.max():.4f}] "
          f"fixed [{fix_old.min():.4f}, {fix_old.max():.4f}]  NCC={ncc_old:.4f}")
    print(f"  NEW preproc: moving [{mov_new.min():.4f}, {mov_new.max():.4f}] "
          f"fixed [{fix_new.min():.4f}, {fix_new.max():.4f}]  NCC={ncc_new:.4f}")

    # ── 检查 ──
    checks = {
        "moving not all-zero": mov_new.max() > 0.01,
        "fixed not all-zero": fix_new.max() > 0.01,
        "pre NCC > 0": ncc_new > 0.01,
    }
    for name, ok in checks.items():
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_pass = False
        print(f"    [{status}] {name}")

# ── 模型前向推理 ──
print(f"\n{'='*70}")
print("MODEL FORWARD PASS (best_model.pth)")
print(f"{'='*70}")

if CKPT.exists():
    from models.vxm2d import VxmDense2D
    device = torch.device('cpu')
    ckpt = torch.load(str(CKPT), map_location=device, weights_only=False)

    model = VxmDense2D(
        in_channels=1,
        enc_channels=[16, 32, 32, 32],
        dec_channels=[32, 32, 32, 32, 16, 16],
        integration_steps=7,
        bidir=True,
    )
    # checkpoint may be raw state_dict or wrapped dict
    sd = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
    model.load_state_dict(sd)
    model.eval()

    for pid in HARD_PAIRS:
        npz_path = NPZ_DIR / f"{pid}.npz"
        if not npz_path.exists():
            continue
        npz = np.load(str(npz_path), allow_pickle=True)
        mov = preprocess_fixed(npz['moving_rigid'])
        fix = preprocess_fixed(npz['fixed_raw'])

        mov_t = torch.from_numpy(mov[np.newaxis, np.newaxis]).float()
        fix_t = torch.from_numpy(fix[np.newaxis, np.newaxis]).float()

        with torch.no_grad():
            out = model(mov_t, fix_t)
            warped = out[0]
            flow = out[1]

        w = warped.squeeze().numpy()
        f = flow.squeeze().numpy()
        out_ncc = ncc_numpy(w, fix, win=9)

        print(f"\n  {pid}")
        print(f"    warped range: [{w.min():.4f}, {w.max():.4f}]  mean={w.mean():.4f}")
        print(f"    flow   range: [{f.min():.4f}, {f.max():.4f}]  mean_abs={np.abs(f).mean():.4f}")
        print(f"    output NCC (warped vs fixed_raw): {out_ncc:.4f}")

        ok = w.max() > 0.01
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_pass = False
        print(f"    [{status}] warped not all-zero")
else:
    print(f"  [SKIP] checkpoint not found: {CKPT}")

print(f"\n{'='*70}")
if all_pass:
    print("ALL CHECKS PASSED — ready for formal retraining")
else:
    print("SOME CHECKS FAILED — review above output")
print(f"{'='*70}")

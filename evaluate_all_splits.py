"""
全量评估：对 train/val/test 全部 24 对进行三方对比。
输出：results/cross_session_v2/three_way_comparison.csv
"""
import sys, os
sys.path.insert(0, ".")

import numpy as np
import torch
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader

from data.cross_session_dataset import CrossSessionPairDataset, CrossSessionCollator
from models.vxm2d import VxmDense2D
from scipy.ndimage import uniform_filter


def ncc_numpy(a, b, win=9):
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    a_mean = uniform_filter(a, win)
    b_mean = uniform_filter(b, win)
    a_var = uniform_filter(a**2, win) - a_mean**2
    b_var = uniform_filter(b**2, win) - b_mean**2
    ab_cov = uniform_filter(a * b, win) - a_mean * b_mean
    denom = np.sqrt(np.maximum(a_var, 0) * np.maximum(b_var, 0)) + 1e-8
    cc = ab_cov / denom
    return float(cc.mean())


def jacobian_stats(flow):
    """flow: (2, H, W)"""
    dydx = np.gradient(flow[0], axis=1)
    dydy = np.gradient(flow[0], axis=0)
    dxdx = np.gradient(flow[1], axis=1)
    dxdy = np.gradient(flow[1], axis=0)
    jac_det = (1 + dydx) * (1 + dxdx) - dydy * dxdy
    return float(jac_det.mean()), float((jac_det < 0).mean() * 100)


# ── Config ──
BUNDLE = Path("data/cross_session/fus_voxelmorph_dlprep_bundle_20260406_115937")
CKPT = Path("checkpoints/cross_session/best_model.pth")
OUTPUT_DIR = Path("results/cross_session_v2")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# ── Model ──
model = VxmDense2D(
    in_channels=1,
    enc_channels=[16, 32, 32, 32],
    dec_channels=[32, 32, 32, 32, 16, 16],
    integration_steps=7,
    bidir=True,
)
ckpt = torch.load(str(CKPT), map_location=device, weights_only=False)
sd = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
model.load_state_dict(sd)
model.to(device)
model.eval()

# ── Baseline reference ──
baseline_ref = {}
ref_csv = BUNDLE / "dl_prep" / "eval" / "baseline_reference.csv"
if ref_csv.exists():
    df_ref = pd.read_csv(ref_csv)
    for _, row in df_ref.iterrows():
        baseline_ref[row['pair_id']] = {
            'pre_ncc': row.get('pre_mean_ncc', np.nan),
            'stage3b_ncc': row.get('stage3b_backbone_mean_ncc', np.nan),
        }

# ── Evaluate all splits ──
all_results = []

for split in ['train', 'val', 'test']:
    ds = CrossSessionPairDataset(
        data_dir=str(BUNDLE),
        split=split,
        mode='B',
        use_padded=True,
        normalize='percentile',
        percentile=(1, 99),
        apply_log=True,
        augmentation=None,
    )

    loader = DataLoader(ds, batch_size=1, shuffle=False,
                        collate_fn=CrossSessionCollator(), num_workers=0)

    for batch in loader:
        pid = batch['pair_id'][0]
        source = batch['moving'].to(device)
        target = batch['fixed'].to(device)
        mask = batch['mask']

        with torch.no_grad():
            warped, flow, _, _ = model(source, target)

        src_np = source[0, 0].cpu().numpy()
        tgt_np = target[0, 0].cpu().numpy()
        warp_np = warped[0, 0].cpu().numpy()
        flow_np = flow[0].cpu().numpy()
        mask_np = mask[0, 0].numpy()

        pre_ncc = ncc_numpy(src_np, tgt_np)
        dl_ncc = ncc_numpy(warp_np, tgt_np)

        jac_mean, jac_neg_pct = jacobian_stats(flow_np)

        # baseline from csv
        ref = baseline_ref.get(pid, {})
        ref_pre_ncc = ref.get('pre_ncc', np.nan)
        ref_s3b_ncc = ref.get('stage3b_ncc', np.nan)

        # Percentage improvements
        dl_vs_pre_pct = ((dl_ncc - pre_ncc) / abs(pre_ncc) * 100) if pre_ncc != 0 else 0.0
        dl_vs_s3b_pct = ((dl_ncc - ref_s3b_ncc) / abs(ref_s3b_ncc) * 100) if (not np.isnan(ref_s3b_ncc) and ref_s3b_ncc != 0) else np.nan
        s3b_vs_pre_pct = ((ref_s3b_ncc - pre_ncc) / abs(pre_ncc) * 100) if (not np.isnan(ref_s3b_ncc) and pre_ncc != 0) else np.nan

        row = {
            'split': split,
            'pair_id': pid,
            'ref_pre_ncc': ref_pre_ncc,
            'ref_stage3b_ncc': ref_s3b_ncc,
            'preproc_pre_ncc': pre_ncc,
            'dl_ncc': dl_ncc,
            'dl_vs_stage3b': dl_ncc - ref_s3b_ncc if not np.isnan(ref_s3b_ncc) else np.nan,
            'dl_vs_stage3b_pct': dl_vs_s3b_pct,
            'dl_vs_pre': dl_ncc - pre_ncc,
            'dl_vs_pre_pct': dl_vs_pre_pct,
            's3b_vs_pre_pct': s3b_vs_pre_pct,
            'dl_beats_stage3b': (dl_ncc > ref_s3b_ncc) if not np.isnan(ref_s3b_ncc) else np.nan,
            'warped_min': warp_np.min(),
            'warped_max': warp_np.max(),
            'flow_mean_abs': np.abs(flow_np).mean(),
            'jac_mean': jac_mean,
            'jac_neg_pct': jac_neg_pct,
        }
        all_results.append(row)

        # Per-pair output with percentage
        if not np.isnan(ref_s3b_ncc):
            win = " WIN" if dl_ncc > ref_s3b_ncc else ""
            print(f"  [{split:5s}] {pid:40s}  pre={pre_ncc:.4f}  S3B={ref_s3b_ncc:.4f}({s3b_vs_pre_pct:+.1f}%)  "
                  f"DL={dl_ncc:.4f}({dl_vs_pre_pct:+.1f}%, vs3B:{dl_vs_s3b_pct:+.1f}%){win}")
        else:
            print(f"  [{split:5s}] {pid:40s}  pre={pre_ncc:.4f}  DL={dl_ncc:.4f}({dl_vs_pre_pct:+.1f}%)")

# ── Save CSV ──
df = pd.DataFrame(all_results)
csv_path = OUTPUT_DIR / "three_way_comparison_v2.csv"
df.to_csv(csv_path, index=False)
print(f"\nSaved: {csv_path}")

# ── Summary ──
print("\n" + "=" * 70)
print("SUMMARY BY SPLIT")
print("=" * 70)
for split in ['train', 'val', 'test', 'all']:
    subset = df if split == 'all' else df[df['split'] == split]
    n = len(subset)
    print(f"\n  {split.upper()} ({n} pairs):")
    print(f"    Pre NCC:            {subset['preproc_pre_ncc'].mean():.4f}")
    if subset['ref_stage3b_ncc'].notna().any():
        s3b_vals = subset['ref_stage3b_ncc'].dropna()
        s3b_pct = subset['s3b_vs_pre_pct'].dropna()
        print(f"    Stage3B NCC:        {s3b_vals.mean():.4f}  (vs pre: {s3b_pct.mean():+.1f}%)")
    print(f"    DL NCC:             {subset['dl_ncc'].mean():.4f}  (vs pre: {subset['dl_vs_pre_pct'].mean():+.1f}%)")
    if subset['dl_vs_stage3b_pct'].notna().any():
        pct = subset['dl_vs_stage3b_pct'].dropna()
        print(f"    DL vs Stage3B:      {pct.mean():+.1f}% NCC improvement")
    if subset['dl_beats_stage3b'].notna().any():
        wins = subset['dl_beats_stage3b'].dropna()
        n_win = int(wins.sum())
        n_total = len(wins)
        print(f"    DL beats Stage3B:   {n_win}/{n_total} pairs ({n_win/n_total*100:.0f}%)")
    print(f"    Jac neg %:          {subset['jac_neg_pct'].mean():.2f}%")

# Overall one-line verdict
print("\n" + "-" * 70)
if df['dl_vs_stage3b_pct'].notna().any():
    overall_pct = df['dl_vs_stage3b_pct'].dropna().mean()
    overall_wins = int(df['dl_beats_stage3b'].dropna().sum())
    overall_total = len(df['dl_beats_stage3b'].dropna())
    print(f"  VERDICT: DL vs Stage3B overall {overall_pct:+.1f}% NCC, "
          f"wins {overall_wins}/{overall_total} ({overall_wins/overall_total*100:.0f}%)")

print(f"\n{'='*70}")
print("DONE")

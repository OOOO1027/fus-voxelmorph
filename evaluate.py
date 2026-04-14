"""
Evaluation script for fUS-VoxelMorph registration results.

支持三方对比（DL-Prep 评估协议）:
  - pre:          配准前 (moving vs fixed)
  - stage3b:      Stage 3B plane-rigid backbone
  - dl_output:    深度学习模型输出

Usage:
    # 跨 session 模式（推荐，三方对比）
    python evaluate.py --config configs/cross_session.yaml --model checkpoints/cross_session/best_model.pth

    # 原有模式
    python evaluate.py --model checkpoints/best_model.pth --data_path data/fus_frames/
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from models import VxmDense2D
from data import FUSDataset, FUSPairDataset, CrossSessionPairDataset, CrossSessionCollator
from utils.metrics import (
    compute_ncc, compute_mse, compute_ssim, compute_ms_ssim,
    compute_dsc, compute_haar_psi, jacobian_determinant_2d, compute_all_metrics,
)


def evaluate_cross_session(model, cfg, device, output_dir):
    """
    跨 session 三方对比评估。

    对比: pre (配准前) / stage3b (刚体backbone) / dl_output (模型输出)
    """
    data_cfg = cfg['data']
    eval_cfg = cfg.get('eval', {})

    # 加载测试集（带 Stage 3B 结果）
    test_set = CrossSessionPairDataset(
        data_dir=data_cfg['cross_session_dir'],
        split='test',
        mode=data_cfg.get('training_mode', 'B'),
        use_padded=data_cfg.get('use_padded', True),
        normalize=data_cfg.get('normalize', 'percentile'),
        percentile=tuple(data_cfg.get('percentile', [1, 99])),
        apply_log=data_cfg.get('log_transform', True),
        augmentation=None,
    )

    loader = DataLoader(
        test_set,
        batch_size=1,
        shuffle=False,
        collate_fn=CrossSessionCollator(),
    )

    use_mask = cfg.get('loss', {}).get('mask_aware', False)
    is_bidir = cfg.get('model', {}).get('bidir', False)
    model.eval()

    all_results = []

    print("\n" + "=" * 70)
    print("CROSS-SESSION THREE-WAY EVALUATION")
    print("pre (before) vs stage3b (rigid backbone) vs dl_output (model)")
    print("=" * 70)

    with torch.no_grad():
        for i, batch in enumerate(loader):
            source = batch['moving'].to(device)
            target = batch['fixed'].to(device)
            mask = batch['mask'] if use_mask else None
            pair_id = batch['pair_id'][0]

            # DL 模型推理
            if is_bidir:
                warped, flow, _, _ = model(source, target)
            else:
                warped, flow = model(source, target)

            src_np = source[0, 0].cpu().numpy()
            tgt_np = target[0, 0].cpu().numpy()
            warp_np = warped[0, 0].cpu().numpy()
            flow_np = flow[0].cpu().numpy()
            mask_np = mask[0, 0].cpu().numpy() if mask is not None else None

            # ---- pre: 配准前 ----
            pre_metrics = {
                'ncc': compute_ncc(src_np, tgt_np),
                'mse': compute_mse(src_np, tgt_np),
                'ssim': compute_ssim(src_np, tgt_np),
            }

            # ---- dl_output: 模型输出 ----
            dl_metrics = compute_all_metrics(warp_np, tgt_np, flow_np, source=src_np)

            # ---- stage3b: rigid backbone ----
            stage3b_metrics = {}
            if 'stage3b_warped' in batch:
                s3b_np = batch['stage3b_warped'][0, 0].cpu().numpy()
                stage3b_metrics = {
                    'ncc': compute_ncc(s3b_np, tgt_np),
                    'mse': compute_mse(s3b_np, tgt_np),
                    'ssim': compute_ssim(s3b_np, tgt_np),
                }

            # Percentage improvements
            pre_ncc = pre_metrics['ncc']
            dl_ncc = dl_metrics['ncc']
            s3b_ncc = stage3b_metrics.get('ncc', None)

            dl_vs_pre_pct = ((dl_ncc - pre_ncc) / abs(pre_ncc) * 100) if pre_ncc != 0 else 0.0
            s3b_vs_pre_pct = ((s3b_ncc - pre_ncc) / abs(pre_ncc) * 100) if (s3b_ncc is not None and pre_ncc != 0) else None
            dl_vs_s3b_pct = ((dl_ncc - s3b_ncc) / abs(s3b_ncc) * 100) if (s3b_ncc is not None and s3b_ncc != 0) else None

            row = {
                'pair_id': pair_id,
                # pre
                'pre_ncc': pre_ncc,
                'pre_mse': pre_metrics['mse'],
                'pre_ssim': pre_metrics['ssim'],
                # stage3b
                'stage3b_ncc': s3b_ncc,
                'stage3b_mse': stage3b_metrics.get('mse', None),
                'stage3b_ssim': stage3b_metrics.get('ssim', None),
                'stage3b_vs_pre_pct': s3b_vs_pre_pct,
                # dl_output
                'dl_ncc': dl_ncc,
                'dl_mse': dl_metrics['mse'],
                'dl_ssim': dl_metrics['ssim'],
                'dl_ms_ssim': dl_metrics['ms_ssim'],
                'dl_dsc': dl_metrics['dsc'],
                'dl_haar_psi': dl_metrics['haar_psi'],
                'dl_jac_mean': dl_metrics['jac_mean'],
                'dl_jac_pct_neg': dl_metrics['jac_pct_neg'],
                # gain vs pre
                'dl_ncc_gain_vs_pre': dl_ncc - pre_ncc,
                'dl_vs_pre_pct': dl_vs_pre_pct,
                'dl_mse_gain_vs_pre': pre_metrics['mse'] - dl_metrics['mse'],
                # gain vs stage3b
                'dl_ncc_gain_vs_stage3b': (dl_ncc - s3b_ncc) if s3b_ncc is not None else None,
                'dl_vs_stage3b_pct': dl_vs_s3b_pct,
                'dl_mse_gain_vs_stage3b': (stage3b_metrics['mse'] - dl_metrics['mse']) if stage3b_metrics else None,
                'dl_beats_stage3b': (dl_ncc > s3b_ncc) if s3b_ncc is not None else None,
            }

            all_results.append(row)

            # 保存每对的推理结果
            pair_dir = os.path.join(output_dir, pair_id)
            os.makedirs(pair_dir, exist_ok=True)
            np.save(os.path.join(pair_dir, 'warped.npy'), warp_np)
            np.save(os.path.join(pair_dir, 'flow.npy'), flow_np)

            # Per-pair output with percentage
            print(f"\n  [{pair_id}]")
            print(f"    pre     NCC={pre_ncc:.4f}  SSIM={pre_metrics['ssim']:.4f}")
            if stage3b_metrics:
                print(f"    stage3b NCC={s3b_ncc:.4f}  SSIM={stage3b_metrics['ssim']:.4f}"
                      f"  (vs pre: {s3b_vs_pre_pct:+.1f}%)")
            win_marker = " WIN" if (dl_vs_s3b_pct is not None and dl_vs_s3b_pct > 0) else ""
            print(f"    DL      NCC={dl_ncc:.4f}  SSIM={dl_metrics['ssim']:.4f}"
                  f"  (vs pre: {dl_vs_pre_pct:+.1f}%"
                  f"{f', vs 3B: {dl_vs_s3b_pct:+.1f}%' if dl_vs_s3b_pct is not None else ''}"
                  f"){win_marker}")

    # 汇总
    df = pd.DataFrame(all_results)
    df.to_csv(os.path.join(output_dir, 'three_way_comparison.csv'), index=False)

    n_pairs = len(df)

    print("\n" + "=" * 70)
    print(f"SUMMARY ({n_pairs} pairs)")
    print("=" * 70)

    # Mean NCC / SSIM for each method
    print("\n  Method          NCC (mean±std)       SSIM (mean±std)")
    print("  " + "-" * 58)
    for method, prefix in [('pre', 'pre'), ('Stage3B', 'stage3b'), ('DL', 'dl')]:
        ncc_col = f'{prefix}_ncc'
        ssim_col = f'{prefix}_ssim'
        if ncc_col in df.columns and df[ncc_col].notna().any():
            ncc_v = df[ncc_col].dropna()
            ssim_str = ""
            if ssim_col in df.columns and df[ssim_col].notna().any():
                ssim_v = df[ssim_col].dropna()
                ssim_str = f"{ssim_v.mean():.4f}±{ssim_v.std():.4f}"
            print(f"  {method:<14s}  {ncc_v.mean():.4f}±{ncc_v.std():.4f}       {ssim_str}")

    # Percentage improvements
    print("\n  Improvements:")
    if df['dl_vs_pre_pct'].notna().any():
        pct = df['dl_vs_pre_pct'].dropna()
        print(f"    DL vs pre:      {pct.mean():+.1f}% NCC improvement (mean)")
    if df['dl_vs_stage3b_pct'].notna().any():
        pct = df['dl_vs_stage3b_pct'].dropna()
        print(f"    DL vs Stage3B:  {pct.mean():+.1f}% NCC improvement (mean)")

    # Win/loss count
    if 'dl_beats_stage3b' in df.columns and df['dl_beats_stage3b'].notna().any():
        wins = df['dl_beats_stage3b'].dropna()
        n_win = int(wins.sum())
        n_total = len(wins)
        print(f"\n  DL beats Stage3B: {n_win}/{n_total} pairs ({n_win/n_total*100:.0f}%)")

    # Jacobian quality
    if 'dl_jac_pct_neg' in df.columns:
        jac = df['dl_jac_pct_neg']
        print(f"  DL Jacobian folding: {jac.mean():.2f}% ± {jac.std():.2f}%")

    print("\n" + "=" * 70)
    print(f"Results saved to: {output_dir}/three_way_comparison.csv")

    return df


def evaluate_model(model, dataloader, device, output_dir=None, max_vis=5):
    """
    原有模式：评估模型在普通数据集上的表现。
    """
    model.eval()

    ncc_scores = []
    mse_scores = []
    ssim_scores = []
    jac_neg_pcts = []
    ncc_before = []
    mse_before = []

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if isinstance(batch, dict):
                source = batch['moving']
                target = batch['fixed']
            elif len(batch) == 3:
                source, target, _ = batch
            else:
                source, target = batch

            source = source.to(device)
            target = target.to(device)

            warped, flow = model(source, target)

            for b in range(source.shape[0]):
                src_np = source[b, 0].cpu().numpy()
                tgt_np = target[b, 0].cpu().numpy()
                warp_np = warped[b, 0].cpu().numpy()
                flow_np = flow[b].cpu().numpy()

                ncc_before.append(compute_ncc(src_np, tgt_np))
                mse_before.append(compute_mse(src_np, tgt_np))
                ncc_scores.append(compute_ncc(warp_np, tgt_np))
                mse_scores.append(compute_mse(warp_np, tgt_np))
                ssim_scores.append(compute_ssim(warp_np, tgt_np))

                _, jac_stats = jacobian_determinant_2d(flow_np)
                jac_neg_pcts.append(jac_stats['pct_neg'])

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Number of pairs evaluated: {len(ncc_scores)}")
    print()
    print("Before Registration:")
    print(f"  NCC:  {np.mean(ncc_before):.4f} +/- {np.std(ncc_before):.4f}")
    print(f"  MSE:  {np.mean(mse_before):.4f} +/- {np.std(mse_before):.4f}")
    print()
    print("After Registration:")
    print(f"  NCC:  {np.mean(ncc_scores):.4f} +/- {np.std(ncc_scores):.4f}")
    print(f"  MSE:  {np.mean(mse_scores):.4f} +/- {np.std(mse_scores):.4f}")
    print(f"  SSIM: {np.mean(ssim_scores):.4f} +/- {np.std(ssim_scores):.4f}")
    print(f"  Jac |det|<=0: {np.mean(jac_neg_pcts):.2f}% +/- {np.std(jac_neg_pcts):.2f}%")
    print("=" * 60)

    results = {
        'ncc_before': ncc_before, 'mse_before': mse_before,
        'ncc': ncc_scores, 'mse': mse_scores, 'ssim': ssim_scores,
        'jac_neg_pct': jac_neg_pcts,
    }

    if output_dir:
        np.savez(os.path.join(output_dir, 'metrics.npz'), **results)
        print(f"\nMetrics saved to {output_dir}/metrics.npz")

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate fUS-VoxelMorph')
    parser.add_argument('--config', type=str, default=None,
                        help='Config file (use cross_session.yaml for three-way eval)')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model')
    parser.add_argument('--data_path', type=str, default=None, help='Path to fUS data')
    parser.add_argument('--pair_mode', type=str, default='consecutive')
    parser.add_argument('--output_dir', type=str, default='eval_results/')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--integration_steps', type=int, default=7)
    parser.add_argument('--max_vis', type=int, default=10)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)

    if args.config:
        # ====== 配置文件模式（支持跨 session 三方对比）======
        import yaml
        with open(args.config, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)

        model_cfg = cfg['model']
        model = VxmDense2D(
            in_channels=model_cfg['in_channels'],
            enc_channels=model_cfg['enc_channels'],
            dec_channels=model_cfg['dec_channels'],
            integration_steps=model_cfg['integration_steps'],
            bidir=model_cfg.get('bidir', False),
        ).to(device)
        model.load_state_dict(torch.load(args.model, map_location=device))

        data_type = cfg['data'].get('type', 'real')
        output_dir = cfg.get('eval', {}).get('output_dir', args.output_dir)
        os.makedirs(output_dir, exist_ok=True)

        if data_type == 'cross_session':
            evaluate_cross_session(model, cfg, device, output_dir)
        else:
            # 用配置文件但非 cross_session
            base_dataset = FUSDataset(cfg['data']['data_path'])
            pair_dataset = FUSPairDataset(base_dataset, mode=cfg['data'].get('pair_mode', 'consecutive'))
            dataloader = DataLoader(pair_dataset, batch_size=args.batch_size, shuffle=False)
            evaluate_model(model, dataloader, device, output_dir=output_dir, max_vis=args.max_vis)
    else:
        # ====== 原有命令行参数模式 ======
        model = VxmDense2D(integration_steps=args.integration_steps).to(device)
        model.load_state_dict(torch.load(args.model, map_location=device))

        base_dataset = FUSDataset(args.data_path)
        pair_dataset = FUSPairDataset(base_dataset, mode=args.pair_mode)
        dataloader = DataLoader(pair_dataset, batch_size=args.batch_size, shuffle=False)

        evaluate_model(model, dataloader, device, output_dir=args.output_dir, max_vis=args.max_vis)

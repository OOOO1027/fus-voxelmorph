#!/usr/bin/env python
"""
Inference/registration script for fUS-VoxelMorph.

Registers a moving image to a fixed (target) image using a trained model.

Features:
- Single pair registration (--source + --target)
- Batch time-series registration (--data_path)
- Multiple input formats (.npy, .npz, .mat, .png, .tiff)
- Automatic evaluation metrics computation
- Comprehensive visualization outputs

Usage:
    # Single pair registration
    python register.py --model checkpoints/best_model.pth \\
        --source moving.npy --target fixed.npy --output_dir results/

    # Time-series registration
    python register.py --model checkpoints/best_model.pth \\
        --data_path timeseries.npy --ref_idx 0 --output_dir results/

    # With evaluation and visualization
    python register.py --model checkpoints/best_model.pth \\
        --source src.npy --target tgt.npy \\
        --eval --visualize --output_dir results/
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

from models import VxmDense2D
from utils import compute_ncc, compute_mse, compute_ssim, jacobian_determinant_2d
from utils.visualization import (
    create_registration_figure,
    create_overlay_comparison,
    create_flow_visualization,
    create_jacobian_visualization,
    create_difference_map,
    save_all_visualizations
)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='fUS-VoxelMorph Registration',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Model
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--integration_steps', type=int, default=7,
                       help='Integration steps (must match training)')
    parser.add_argument('--enc_channels', type=int, nargs='+', default=[16, 32, 32, 32],
                       help='Encoder channels (must match training)')
    parser.add_argument('--dec_channels', type=int, nargs='+', default=[32, 32, 32, 32, 16, 16],
                       help='Decoder channels (must match training)')

    # Input (single pair)
    parser.add_argument('--source', '--moving', type=str, dest='source', default=None,
                       help='Moving image path')
    parser.add_argument('--target', '--fixed', type=str, dest='target', default=None,
                       help='Fixed/target image path')

    # Input (time-series)
    parser.add_argument('--data_path', type=str, default=None,
                       help='Time-series data path (.npy or folder)')
    parser.add_argument('--ref_idx', type=int, default=0,
                       help='Reference frame index for time-series')

    # Output
    parser.add_argument('--output_dir', type=str, default='results/',
                       help='Output directory')
    parser.add_argument('--prefix', type=str, default='',
                       help='Output file prefix')

    # Evaluation
    parser.add_argument('--eval', action='store_true',
                       help='Compute and save evaluation metrics')
    parser.add_argument('--save_npy', action='store_true', default=True,
                       help='Save results as .npy files')
    parser.add_argument('--no_save_npy', dest='save_npy', action='store_false',
                       help='Do not save .npy files')

    # Visualization
    parser.add_argument('--visualize', '-v', action='store_true',
                       help='Generate visualization figures')
    parser.add_argument('--vis_format', type=str, default='png',
                       choices=['png', 'pdf', 'svg'],
                       help='Visualization format')
    parser.add_argument('--vis_dpi', type=int, default=150,
                       help='Visualization DPI')

    # System
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device: cuda or cpu')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size for time-series processing')

    return parser.parse_args()


def load_image(path, target_size=None, normalize=True):
    """
    Load image from various formats.

    Supports: .npy, .npz, .mat, .png, .jpg, .tiff, .tif
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    ext = path.suffix.lower()

    if ext == '.npy':
        img = np.load(str(path)).astype(np.float32)

    elif ext == '.npz':
        data = np.load(str(path))
        # Try common keys
        for key in ['data', 'image', 'img', 'arr_0']:
            if key in data:
                img = data[key].astype(np.float32)
                break
        else:
            img = data[data.files[0]].astype(np.float32)

    elif ext == '.mat':
        try:
            from scipy.io import loadmat
            mat = loadmat(str(path))
            # Try common variable names
            for key in ['data', 'image', 'img', 'volume']:
                if key in mat:
                    img = mat[key].astype(np.float32)
                    break
            else:
                # Use last variable (skip metadata)
                img = list(mat.values())[-1].astype(np.float32)
        except ImportError:
            raise ImportError("scipy required for .mat files: pip install scipy")

    elif ext in ['.png', '.jpg', '.jpeg', '.tiff', '.tif']:
        from PIL import Image
        img = np.array(Image.open(str(path))).astype(np.float32)
        if img.ndim == 3:
            img = img[:, :, 0]  # Take first channel

    else:
        raise ValueError(f"Unsupported file format: {ext}")

    # Ensure 2D
    if img.ndim == 3:
        if img.shape[0] == 1:
            img = img[0]
        elif img.shape[-1] == 1:
            img = img[..., 0]
        else:
            img = img[..., 0]  # Take first channel

    # Resize if needed
    if target_size is not None and img.shape != tuple(target_size):
        from scipy.ndimage import zoom
        factors = [t / s for t, s in zip(target_size, img.shape)]
        img = zoom(img, factors, order=1)

    # Normalize
    if normalize:
        img_min, img_max = img.min(), img.max()
        if img_max > img_min:
            img = (img - img_min) / (img_max - img_min)
        else:
            img = np.zeros_like(img)

    return img


def save_image(img, path, format=None):
    """Save image to various formats."""
    path = Path(path)
    os.makedirs(path.parent, exist_ok=True)

    if format is None:
        format = path.suffix[1:] if path.suffix else 'npy'

    if format in ['npy', 'npz']:
        np.save(str(path), img)
    elif format in ['png', 'jpg', 'tiff']:
        from PIL import Image
        # Convert to 8-bit
        img_8bit = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        Image.fromarray(img_8bit).save(str(path))
    else:
        raise ValueError(f"Unsupported save format: {format}")


def register_pair(model, source, target, device):
    """
    Register a single source-target pair.

    Parameters
    ----------
    model : VxmDense2D
    source : np.ndarray (H, W) or torch.Tensor (1, 1, H, W)
    target : np.ndarray (H, W) or torch.Tensor (1, 1, H, W)
    device : torch.device

    Returns
    -------
    warped : np.ndarray (H, W)
    flow : np.ndarray (2, H, W)
    """
    model.eval()

    # Convert to tensor if needed
    if isinstance(source, np.ndarray):
        if source.ndim == 2:
            source = source[np.newaxis, np.newaxis, :, :]  # (1, 1, H, W)
        source = torch.from_numpy(source).float()

    if isinstance(target, np.ndarray):
        if target.ndim == 2:
            target = target[np.newaxis, np.newaxis, :, :]
        target = torch.from_numpy(target).float()

    with torch.no_grad():
        source = source.to(device)
        target = target.to(device)
        warped, flow = model(source, target)

    return warped[0, 0].cpu().numpy(), flow[0].cpu().numpy()


def compute_metrics(source, target, warped, flow):
    """
    Compute registration evaluation metrics.

    Returns
    -------
    dict : Dictionary of metrics
    """
    metrics = {}

    # Image similarity metrics
    metrics['ncc_before'] = compute_ncc(source, target)
    metrics['ncc_after'] = compute_ncc(warped, target)
    metrics['mse_before'] = compute_mse(source, target)
    metrics['mse_after'] = compute_mse(warped, target)
    metrics['ssim_after'] = compute_ssim(warped, target)

    # Improvement
    metrics['ncc_improvement'] = metrics['ncc_after'] - metrics['ncc_before']
    metrics['mse_improvement'] = metrics['mse_before'] - metrics['mse_after']

    # Jacobian analysis
    jac_det, jac_stats = jacobian_determinant_2d(flow)
    metrics['jac_mean'] = jac_stats['mean']
    metrics['jac_std'] = jac_stats['std']
    metrics['jac_min'] = jac_stats['min']
    metrics['jac_pct_neg'] = jac_stats['pct_neg']

    # Displacement statistics
    flow_mag = np.sqrt(flow[0]**2 + flow[1]**2)
    metrics['flow_mean'] = flow_mag.mean()
    metrics['flow_max'] = flow_mag.max()
    metrics['flow_std'] = flow_mag.std()

    return metrics


def print_metrics(metrics, title="Registration Metrics"):
    """Pretty print metrics."""
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)

    print("\nImage Similarity:")
    print(f"  NCC  - Before: {metrics['ncc_before']:.4f}  After: {metrics['ncc_after']:.4f}  "
          f"(Δ {metrics['ncc_improvement']:+.4f})")
    print(f"  MSE  - Before: {metrics['mse_before']:.4f}  After: {metrics['mse_after']:.4f}  "
          f"(Δ {metrics['mse_improvement']:+.4f})")
    print(f"  SSIM - After:  {metrics['ssim_after']:.4f}")

    print("\nDeformation Field:")
    print(f"  Mean displacement: {metrics['flow_mean']:.2f} px")
    print(f"  Max displacement:  {metrics['flow_max']:.2f} px")
    print(f"  Std displacement:  {metrics['flow_std']:.2f} px")

    print("\nJacobian Analysis:")
    print(f"  Mean det(J):  {metrics['jac_mean']:.4f}")
    print(f"  Min det(J):   {metrics['jac_min']:.4f}")
    print(f"  |det|<=0:     {metrics['jac_pct_neg']:.2f}%")

    print("=" * 60)


def save_metrics(metrics, save_path):
    """Save metrics to JSON file."""
    import json
    with open(save_path, 'w') as f:
        json.dump(metrics, f, indent=2)


def register_single_pair(args, model, device):
    """Register a single pair of images."""
    print(f"Loading images...")
    source = load_image(args.source)
    target = load_image(args.target)

    print(f"Source: {source.shape}, range: [{source.min():.3f}, {source.max():.3f}]")
    print(f"Target: {target.shape}, range: [{target.min():.3f}, {target.max():.3f}]")

    print(f"\nRegistering...")
    warped, flow = register_pair(model, source, target, device)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    prefix = args.prefix + '_' if args.prefix else ''

    # Save results
    if args.save_npy:
        np.save(os.path.join(args.output_dir, f'{prefix}warped.npy'), warped)
        np.save(os.path.join(args.output_dir, f'{prefix}flow.npy'), flow)
        np.save(os.path.join(args.output_dir, f'{prefix}source.npy'), source)
        np.save(os.path.join(args.output_dir, f'{prefix}target.npy'), target)
        print(f"Saved results to {args.output_dir}")

    # Compute and print metrics
    if args.eval:
        metrics = compute_metrics(source, target, warped, flow)
        print_metrics(metrics)
        save_metrics(metrics, os.path.join(args.output_dir, f'{prefix}metrics.json'))
    else:
        metrics = None

    # Generate visualizations
    if args.visualize:
        print("\nGenerating visualizations...")
        save_all_visualizations(
            source, target, warped, flow,
            output_dir=args.output_dir,
            prefix=prefix,
            format=args.vis_format,
            dpi=args.vis_dpi
        )
        print(f"Visualizations saved to {args.output_dir}")

    return warped, flow, metrics


def register_time_series(args, model, device):
    """Register a time-series to a reference frame."""
    print(f"Loading time-series data from {args.data_path}...")

    # Load data
    if os.path.isdir(args.data_path):
        # Directory of images
        files = sorted(Path(args.data_path).glob('*.npy'))
        if not files:
            raise ValueError(f"No .npy files found in {args.data_path}")

        data_list = [np.load(str(f)).astype(np.float32) for f in files]
        data = np.stack(data_list)
    else:
        # Single file
        data = load_image(args.data_path, normalize=False)

    if data.ndim == 2:
        print("Only one frame, nothing to register.")
        return

    if data.ndim == 3:
        T, H, W = data.shape
    else:
        raise ValueError(f"Unexpected data shape: {data.shape}")

    print(f"Time-series: {T} frames of size {H}x{W}")
    print(f"Reference frame: {args.ref_idx}")

    # Normalize entire series
    dmin, dmax = data.min(), data.max()
    if dmax > dmin:
        data = (data - dmin) / (dmax - dmin)

    ref = data[args.ref_idx]
    ref_tensor = torch.from_numpy(ref[np.newaxis, np.newaxis]).float()

    # Output arrays
    warped_all = np.zeros_like(data)
    flows_all = np.zeros((T, 2, H, W), dtype=np.float32)
    metrics_all = []

    warped_all[args.ref_idx] = data[args.ref_idx]

    # Process in batches
    print(f"\nRegistering...")
    batch_size = args.batch_size

    for start_idx in range(0, T, batch_size):
        end_idx = min(start_idx + batch_size, T)
        batch_indices = [i for i in range(start_idx, end_idx) if i != args.ref_idx]

        if not batch_indices:
            continue

        # Prepare batch
        sources = torch.stack([
            torch.from_numpy(data[i][np.newaxis, np.newaxis]).float()
            for i in batch_indices
        ]).squeeze(1).to(device)

        targets = ref_tensor.expand(len(batch_indices), -1, -1, -1).to(device)

        # Register batch
        model.eval()
        with torch.no_grad():
            warped_batch, flow_batch = model(sources, targets)

        # Store results
        for idx, batch_idx in enumerate(batch_indices):
            warped_all[batch_idx] = warped_batch[idx, 0].cpu().numpy()
            flows_all[batch_idx] = flow_batch[idx].cpu().numpy()

            if args.eval:
                metrics = compute_metrics(
                    data[batch_idx], ref,
                    warped_all[batch_idx], flows_all[batch_idx]
                )
                metrics['frame'] = batch_idx
                metrics_all.append(metrics)

        print(f"  Processed {min(end_idx, T)}/{T} frames")

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    prefix = args.prefix + '_' if args.prefix else ''

    if args.save_npy:
        np.save(os.path.join(args.output_dir, f'{prefix}warped_series.npy'), warped_all)
        np.save(os.path.join(args.output_dir, f'{prefix}flows.npy'), flows_all)
        print(f"\nSaved results to {args.output_dir}")

    # Save metrics
    if args.eval and metrics_all:
        import json
        with open(os.path.join(args.output_dir, f'{prefix}metrics_series.json'), 'w') as f:
            json.dump(metrics_all, f, indent=2)

        # Print summary
        ncc_improvements = [m['ncc_improvement'] for m in metrics_all]
        print(f"\nAverage NCC improvement: {np.mean(ncc_improvements):.4f} "
              f"± {np.std(ncc_improvements):.4f}")

    # Visualize sample frames
    if args.visualize:
        print("\nGenerating visualizations...")

        # Visualize first few frames
        vis_frames = [i for i in range(min(5, T)) if i != args.ref_idx]
        if vis_frames:
            for frame_idx in vis_frames:
                frame_prefix = f"{prefix}frame_{frame_idx:04d}_"
                save_all_visualizations(
                    data[frame_idx], ref, warped_all[frame_idx], flows_all[frame_idx],
                    output_dir=args.output_dir,
                    prefix=frame_prefix,
                    format=args.vis_format,
                    dpi=args.vis_dpi
                )

            print(f"Visualizations saved to {args.output_dir}")

    return warped_all, flows_all, metrics_all


def main():
    args = parse_args()

    print("=" * 60)
    print("fUS-VoxelMorph Registration")
    print("=" * 60)

    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load model
    print(f"\nLoading model from {args.model}...")
    model = VxmDense2D(
        in_channels=1,
        enc_channels=args.enc_channels,
        dec_channels=args.dec_channels,
        integration_steps=args.integration_steps
    ).to(device)

    checkpoint = torch.load(args.model, map_location=device)

    # Handle both full checkpoint and state_dict only
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"  Loaded from checkpoint (epoch {checkpoint.get('epoch', 'unknown')})")
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    print(f"  Model ready")

    # Run registration
    if args.source and args.target:
        register_single_pair(args, model, device)
    elif args.data_path:
        register_time_series(args, model, device)
    else:
        print("Error: Provide either (--source and --target) or --data_path")
        sys.exit(1)

    print("\nDone!")


if __name__ == '__main__':
    main()

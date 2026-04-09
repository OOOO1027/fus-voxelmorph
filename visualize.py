#!/usr/bin/env python
"""
Visualization script for fUS-VoxelMorph registration results.

Generate publication-quality figures from saved registration results.

Usage:
    # Visualize single pair
    python visualize.py --source source.npy --target target.npy \\
        --warped warped.npy --flow flow.npy --output_dir figures/

    # Visualize with metrics
    python visualize.py --source src.npy --target tgt.npy \\
        --warped warped.npy --flow flow.npy --metrics metrics.json

    # Visualize time-series results
    python visualize.py --series warped_series.npy --flows flows.npy \\
        --ref_idx 0 --output_dir figures/

    # Create specific visualizations only
    python visualize.py --source src.npy --target tgt.npy --warped warped.npy \\
        --flow flow.npy --only overlay flow --output_dir figures/
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from utils.visualization import (
    create_registration_figure,
    create_overlay_comparison,
    create_flow_visualization,
    create_jacobian_visualization,
    create_difference_map,
    create_grid_overlay,
    save_all_visualizations
)
from utils import compute_ncc, compute_mse, compute_ssim, jacobian_determinant_2d


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Visualize fUS-VoxelMorph registration results',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Input files (single pair)
    parser.add_argument('--source', '--moving', type=str, default=None,
                       help='Source/moving image (.npy)')
    parser.add_argument('--target', '--fixed', type=str, default=None,
                       help='Target/fixed image (.npy)')
    parser.add_argument('--warped', type=str, default=None,
                       help='Warped image (.npy)')
    parser.add_argument('--flow', type=str, default=None,
                       help='Displacement field (.npy)')
    parser.add_argument('--metrics', type=str, default=None,
                       help='Metrics JSON file')

    # Input files (time-series)
    parser.add_argument('--series', type=str, default=None,
                       help='Warped time-series (.npy)')
    parser.add_argument('--flows', type=str, default=None,
                       help='Flow fields time-series (.npy)')
    parser.add_argument('--ref_idx', type=int, default=0,
                       help='Reference frame index')
    parser.add_argument('--frame_idx', type=int, default=None,
                       help='Specific frame to visualize (default: middle)')

    # Output
    parser.add_argument('-o', '--output_dir', type=str, default='figures/',
                       help='Output directory')
    parser.add_argument('--prefix', type=str, default='',
                       help='Output file prefix')
    parser.add_argument('--format', type=str, default='png',
                       choices=['png', 'pdf', 'svg', 'jpg'],
                       help='Output format')
    parser.add_argument('--dpi', type=int, default=300,
                       help='Output DPI (higher = better quality)')

    # Visualization options
    parser.add_argument('--only', type=str, nargs='+',
                       choices=['overview', 'overlay', 'flow', 'jacobian',
                               'difference', 'grid', 'all'],
                       default=['all'],
                       help='Which visualizations to generate')
    parser.add_argument('--no_metrics', action='store_true',
                       help='Do not compute/show metrics')
    parser.add_argument('--title', type=str, default=None,
                       help='Figure title')

    return parser.parse_args()


def load_npy(path):
    """Load numpy file."""
    if path is None:
        return None
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return np.load(str(path))


def load_metrics(path):
    """Load metrics JSON."""
    if path is None:
        return None
    import json
    with open(path, 'r') as f:
        return json.load(f)


def compute_metrics_if_needed(source, target, warped, flow):
    """Compute metrics if not provided."""
    metrics = {}

    # Image similarity
    metrics['ncc_before'] = compute_ncc(source, target)
    metrics['ncc_after'] = compute_ncc(warped, target)
    metrics['mse_before'] = compute_mse(source, target)
    metrics['mse_after'] = compute_mse(warped, target)
    metrics['ssim_after'] = compute_ssim(warped, target)
    metrics['ncc_improvement'] = metrics['ncc_after'] - metrics['ncc_before']
    metrics['mse_improvement'] = metrics['mse_before'] - metrics['mse_after']

    # Flow statistics
    if flow is not None:
        flow_mag = np.sqrt(flow[0]**2 + flow[1]**2)
        metrics['flow_mean'] = flow_mag.mean()
        metrics['flow_max'] = flow_mag.max()
        metrics['flow_std'] = flow_mag.std()

        # Jacobian
        _, jac_stats = jacobian_determinant_2d(flow)
        metrics['jac_mean'] = jac_stats['mean']
        metrics['jac_std'] = jac_stats['std']
        metrics['jac_min'] = jac_stats['min']
        metrics['jac_pct_neg'] = jac_stats['pct_neg']

    return metrics


def visualize_single_pair(args, source, target, warped, flow, metrics):
    """Visualize single pair registration."""
    os.makedirs(args.output_dir, exist_ok=True)
    prefix = args.prefix + '_' if args.prefix else ''

    to_generate = args.only
    if 'all' in to_generate:
        to_generate = ['overview', 'overlay', 'flow', 'jacobian', 'difference', 'grid']

    print(f"Generating visualizations: {', '.join(to_generate)}")

    # 1. Overview figure
    if 'overview' in to_generate:
        print("  Creating overview...")
        fig = create_registration_figure(
            source, target, warped, flow,
            metrics=None if args.no_metrics else metrics,
            title=args.title
        )
        save_path = os.path.join(args.output_dir, f'{prefix}overview.{args.format}')
        fig.savefig(save_path, dpi=args.dpi, bbox_inches='tight')
        plt.close(fig)
        print(f"    Saved: {save_path}")

    # 2. Overlay comparison
    if 'overlay' in to_generate:
        print("  Creating overlay comparison...")
        save_path = os.path.join(args.output_dir, f'{prefix}overlay.{args.format}')
        create_overlay_comparison(source, target, warped, save_path=save_path)
        print(f"    Saved: {save_path}")

    # 3. Flow visualization
    if 'flow' in to_generate and flow is not None:
        print("  Creating flow visualization...")
        save_path = os.path.join(args.output_dir, f'{prefix}flow.{args.format}')
        create_flow_visualization(flow, source=source, save_path=save_path)
        print(f"    Saved: {save_path}")

    # 4. Jacobian visualization
    if 'jacobian' in to_generate and flow is not None:
        print("  Creating jacobian visualization...")
        save_path = os.path.join(args.output_dir, f'{prefix}jacobian.{args.format}')
        create_jacobian_visualization(flow, save_path=save_path)
        print(f"    Saved: {save_path}")

    # 5. Difference map
    if 'difference' in to_generate:
        print("  Creating difference map...")
        save_path = os.path.join(args.output_dir, f'{prefix}difference.{args.format}')
        create_difference_map(source, target, warped, save_path=save_path)
        print(f"    Saved: {save_path}")

    # 6. Grid overlay
    if 'grid' in to_generate and flow is not None:
        print("  Creating grid overlay...")
        save_path = os.path.join(args.output_dir, f'{prefix}grid.{args.format}')
        create_grid_overlay(flow, source=source, save_path=save_path)
        print(f"    Saved: {save_path}")


def visualize_timeseries(args, series, flows):
    """Visualize time-series registration results."""
    T = series.shape[0]
    ref_idx = args.ref_idx

    # Select frame to visualize
    if args.frame_idx is not None:
        frame_idx = args.frame_idx
    else:
        # Select middle frame (not reference)
        candidates = [i for i in range(T) if i != ref_idx]
        frame_idx = candidates[len(candidates) // 2]

    print(f"Time-series: {T} frames")
    print(f"Visualizing frame {frame_idx} (reference: {ref_idx})")

    source = series[frame_idx]
    target = series[ref_idx]
    warped = series[frame_idx]
    flow = flows[frame_idx] if flows is not None else None

    # Compute metrics
    if not args.no_metrics:
        print("Computing metrics...")
        metrics = compute_metrics_if_needed(source, target, warped, flow)
    else:
        metrics = None

    # Update prefix
    old_prefix = args.prefix
    args.prefix = f"{old_prefix}frame_{frame_idx:04d}" if old_prefix else f"frame_{frame_idx:04d}"

    visualize_single_pair(args, source, target, warped, flow, metrics)


def print_metrics_summary(metrics):
    """Print metrics summary."""
    print("\n" + "=" * 60)
    print("Registration Metrics")
    print("=" * 60)

    print("\nImage Similarity:")
    print(f"  NCC  - Before: {metrics.get('ncc_before', 0):.4f}  "
          f"After: {metrics.get('ncc_after', 0):.4f}  "
          f"(Δ {metrics.get('ncc_improvement', 0):+.4f})")
    print(f"  MSE  - Before: {metrics.get('mse_before', 0):.4f}  "
          f"After: {metrics.get('mse_after', 0):.4f}  "
          f"(Δ {metrics.get('mse_improvement', 0):+.4f})")

    if 'flow_mean' in metrics:
        print("\nDeformation Field:")
        print(f"  Mean displacement: {metrics['flow_mean']:.2f} px")
        print(f"  Max displacement:  {metrics['flow_max']:.2f} px")

    if 'jac_pct_neg' in metrics:
        print("\nJacobian Analysis:")
        print(f"  Mean det(J):  {metrics['jac_mean']:.4f}")
        print(f"  Folding:      {metrics['jac_pct_neg']:.2f}%")

    print("=" * 60)


def main():
    args = parse_args()

    print("=" * 60)
    print("fUS-VoxelMorph Visualization")
    print("=" * 60)

    # Determine mode
    if args.series:
        # Time-series mode
        series = load_npy(args.series)
        flows = load_npy(args.flows) if args.flows else None
        metrics = load_metrics(args.metrics) if args.metrics else None

        visualize_timeseries(args, series, flows)

    elif args.warped:
        # Single pair mode
        source = load_npy(args.source)
        target = load_npy(args.target)
        warped = load_npy(args.warped)
        flow = load_npy(args.flow) if args.flow else None

        print(f"Images: {source.shape}")
        if flow is not None:
            print(f"Flow: {flow.shape}")

        # Load or compute metrics
        if args.metrics:
            print(f"Loading metrics from {args.metrics}")
            metrics = load_metrics(args.metrics)
        elif not args.no_metrics:
            print("Computing metrics...")
            metrics = compute_metrics_if_needed(source, target, warped, flow)
        else:
            metrics = None

        if metrics:
            print_metrics_summary(metrics)

        visualize_single_pair(args, source, target, warped, flow, metrics)

    else:
        print("Error: Provide either --warped (single pair) or --series (time-series)")
        sys.exit(1)

    print(f"\nAll figures saved to: {args.output_dir}")
    print("Done!")


if __name__ == '__main__':
    main()

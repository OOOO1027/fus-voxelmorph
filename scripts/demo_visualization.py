"""
Demonstration of visualization capabilities.

This script generates synthetic registration results and creates
all types of visualizations.
"""

import os
import sys
import numpy as np
import torch

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import VxmDense2D
from data import SyntheticFUSGenerator
from utils.visualization import save_all_visualizations
from utils import compute_ncc, compute_mse, jacobian_determinant_2d


def main():
    print("=" * 70)
    print("fUS-VoxelMorph Visualization Demo")
    print("=" * 70)
    print("\nThis demo will:")
    print("  1. Generate a synthetic fUS image pair")
    print("  2. Register using a trained model (or create dummy results)")
    print("  3. Generate all visualization types")
    print("  4. Save to demo_figures/")
    print()

    input("Press Enter to continue...")
    print()

    # Create output directory
    output_dir = 'demo_figures'
    os.makedirs(output_dir, exist_ok=True)

    # Generate synthetic data
    print("Generating synthetic fUS data...")
    generator = SyntheticFUSGenerator(size=(128, 100), motion_type='mixed')
    source, target, flow_gt = generator.generate_pair(seed=42)

    print(f"  Source: {source.shape}, range: [{source.min():.3f}, {source.max():.3f}]")
    print(f"  Target: {target.shape}, range: [{target.min():.3f}, {target.max():.3f}]")
    print(f"  Flow GT: {flow_gt.shape}")

    # Try to load a model, otherwise use ground truth
    model_path = 'checkpoints/best_model.pth'
    if os.path.exists(model_path):
        print(f"\nLoading model from {model_path}...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = VxmDense2D(integration_steps=7).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        # Register
        with torch.no_grad():
            src_tensor = torch.from_numpy(source[np.newaxis, np.newaxis]).float().to(device)
            tgt_tensor = torch.from_numpy(target[np.newaxis, np.newaxis]).float().to(device)
            warped, flow = model(src_tensor, tgt_tensor)
            warped = warped[0, 0].cpu().numpy()
            flow = flow[0].cpu().numpy()
    else:
        print("\nNo trained model found. Using ground truth deformation...")
        from data.synthetic_fus import apply_deformation
        warped = apply_deformation(source, flow_gt)
        flow = flow_gt

    print(f"  Warped: {warped.shape}, range: [{warped.min():.3f}, {warped.max():.3f}]")

    # Compute metrics
    print("\nComputing metrics...")
    metrics = {
        'ncc_before': compute_ncc(source, target),
        'ncc_after': compute_ncc(warped, target),
        'mse_before': compute_mse(source, target),
        'mse_after': compute_mse(warped, target),
    }
    metrics['ncc_improvement'] = metrics['ncc_after'] - metrics['ncc_before']
    metrics['mse_improvement'] = metrics['mse_before'] - metrics['mse_after']

    flow_mag = np.sqrt(flow[0]**2 + flow[1]**2)
    metrics['flow_mean'] = flow_mag.mean()
    metrics['flow_max'] = flow_mag.max()

    _, jac_stats = jacobian_determinant_2d(flow)
    metrics['jac_pct_neg'] = jac_stats['pct_neg']

    print(f"  NCC: {metrics['ncc_before']:.4f} → {metrics['ncc_after']:.4f}")
    print(f"  MSE: {metrics['mse_before']:.4f} → {metrics['mse_after']:.4f}")
    print(f"  Flow: mean={metrics['flow_mean']:.2f}px, max={metrics['flow_max']:.2f}px")
    print(f"  Jacobian: {metrics['jac_pct_neg']:.2f}% folding")

    # Generate visualizations
    print("\nGenerating visualizations...")
    save_all_visualizations(
        source, target, warped, flow,
        output_dir=output_dir,
        prefix='demo_',
        format='png',
        dpi=150
    )

    # List generated files
    print(f"\nGenerated files in {output_dir}/:")
    for f in sorted(os.listdir(output_dir)):
        if f.startswith('demo_'):
            filepath = os.path.join(output_dir, f)
            size_kb = os.path.getsize(filepath) / 1024
            print(f"  - {f} ({size_kb:.1f} KB)")

    print("\n" + "=" * 70)
    print("Demo complete!")
    print("=" * 70)
    print("\nVisualization types:")
    print("  1. overview.png     - Comprehensive registration result")
    print("  2. overlay.png      - Green/magenta overlay (like Zhong et al.)")
    print("  3. flow.png         - Displacement field (magnitude + vectors)")
    print("  4. jacobian.png     - Jacobian determinant analysis")
    print("  5. difference.png   - Difference maps before/after")
    print("  6. grid.png         - Deformed grid overlay")
    print()
    print("To view: open demo_figures/demo_*.png")
    print("\nFor custom visualizations, use:")
    print("  python visualize.py --source src.npy --target tgt.npy \\")
    print("      --warped warped.npy --flow flow.npy --output_dir figures/")


if __name__ == '__main__':
    main()

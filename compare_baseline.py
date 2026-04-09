#!/usr/bin/env python
"""
Comparison experiment between VoxelMorph and traditional registration methods.

Compares:
- VoxelMorph (deep learning)
- Rigid (SimpleITK)
- Affine (SimpleITK)
- Demons (SimpleITK)
- B-spline (SimpleITK)

Metrics (following Zhong et al.):
- NCC (Normalized Cross-Correlation)
- MS-SSIM (Multi-Scale Structural Similarity)
- DSC (Dice Similarity Coefficient)
- Inference time
- Jacobian statistics

Usage:
    # Run on synthetic data
    python compare_baseline.py --synthetic --n_samples 20 --output_dir comparison/

    # Run on real data
    python compare_baseline.py --data_path data/test/ --output_dir comparison/

    # Run with trained VoxelMorph model
    python compare_baseline.py --model checkpoints/best_model.pth \\
        --data_path data/test/ --output_dir comparison/
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

# Import our methods
from models import VxmDense2D
from data import SyntheticFUSGenerator, FUSDataset, FUSPairDataset
from baselines import (
    RigidRegistration,
    AffineRegistration,
    DemonsRegistration,
    BSplineRegistration,
    run_comparison_experiment,
    create_comparison_figures,
    print_comparison_table,
)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Compare VoxelMorph with traditional registration methods',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Data
    data_group = parser.add_mutually_exclusive_group(required=True)
    data_group.add_argument('--data_path', type=str,
                           help='Path to test data directory or file')
    data_group.add_argument('--synthetic', action='store_true',
                           help='Use synthetic data')

    parser.add_argument('--n_samples', type=int, default=20,
                       help='Number of synthetic samples (if --synthetic)')
    parser.add_argument('--image_size', type=int, nargs=2, default=[128, 100],
                       help='Image size for synthetic data')

    # VoxelMorph model
    parser.add_argument('--model', type=str, default=None,
                       help='Path to trained VoxelMorph model')
    parser.add_argument('--integration_steps', type=int, default=7,
                       help='Integration steps for VoxelMorph')

    # Methods to compare
    parser.add_argument('--methods', type=str, nargs='+',
                       default=['rigid', 'affine', 'demons', 'bspline', 'voxelmorph'],
                       help='Methods to compare')

    # Traditional method parameters
    parser.add_argument('--demons_iterations', type=int, default=100,
                       help='Demons algorithm iterations')
    parser.add_argument('--bspline_grid', type=int, nargs=2, default=[10, 10],
                       help='B-spline control grid size')
    parser.add_argument('--bspline_iterations', type=int, default=100,
                       help='B-spline optimization iterations')

    # Output
    parser.add_argument('--output_dir', type=str, default='comparison_results',
                       help='Output directory')
    parser.add_argument('--save_visualizations', action='store_true',
                       help='Save visualization for each test case')

    # System
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device for VoxelMorph')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')

    return parser.parse_args()


def prepare_test_cases(args):
    """Prepare test cases for comparison."""
    np.random.seed(args.seed)

    if args.synthetic:
        print(f"Generating {args.n_samples} synthetic test cases...")
        generator = SyntheticFUSGenerator(
            size=tuple(args.image_size),
            motion_type='mixed'
        )

        test_cases = []
        for i in range(args.n_samples):
            source, target, _ = generator.generate_pair(seed=args.seed + i)
            test_cases.append({
                'name': f'synthetic_{i:03d}',
                'source': source,
                'target': target,
            })

    else:
        print(f"Loading data from {args.data_path}...")

        if not os.path.exists(args.data_path):
            raise FileNotFoundError(f"Data path not found: {args.data_path}")

        dataset = FUSDataset(args.data_path, target_size=tuple(args.image_size))

        # Create pairs
        pair_dataset = FUSPairDataset(dataset, mode='consecutive')

        test_cases = []
        for i in range(min(args.n_samples, len(pair_dataset))):
            source, target = pair_dataset[i]
            source = source[0].numpy()  # Remove channel dim
            target = target[0].numpy()
            test_cases.append({
                'name': f'pair_{i:03d}',
                'source': source,
                'target': target,
            })

    print(f"Prepared {len(test_cases)} test cases")
    return test_cases


def prepare_methods(args):
    """Prepare methods dictionary."""
    methods_dict = {}
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Traditional methods
    if 'rigid' in args.methods:
        print("  - Rigid registration (SimpleITK)")
        try:
            methods_dict['Rigid'] = RigidRegistration(
                metric='mean_squares',
                max_iterations=200
            )
        except ImportError as e:
            print(f"    Warning: {e}")

    if 'affine' in args.methods:
        print("  - Affine registration (SimpleITK)")
        try:
            methods_dict['Affine'] = AffineRegistration(
                metric='mean_squares',
                max_iterations=200
            )
        except ImportError as e:
            print(f"    Warning: {e}")

    if 'demons' in args.methods:
        print("  - Demons registration (SimpleITK)")
        try:
            methods_dict['Demons'] = DemonsRegistration(
                iterations=args.demons_iterations,
                smooth_sigma=1.0
            )
        except ImportError as e:
            print(f"    Warning: {e}")

    if 'bspline' in args.methods:
        print("  - B-spline registration (SimpleITK)")
        try:
            methods_dict['B-spline'] = BSplineRegistration(
                grid_size=tuple(args.bspline_grid),
                iterations=args.bspline_iterations,
                metric='mean_squares'
            )
        except ImportError as e:
            print(f"    Warning: {e}")

    # VoxelMorph
    if 'voxelmorph' in args.methods:
        print("  - VoxelMorph (deep learning)")

        if args.model and os.path.exists(args.model):
            model = VxmDense2D(
                in_channels=1,
                enc_channels=[16, 32, 32, 32],
                dec_channels=[32, 32, 32, 32, 16, 16],
                integration_steps=args.integration_steps
            ).to(device)
            model.load_state_dict(torch.load(args.model, map_location=device))
            model.eval()
            methods_dict['VoxelMorph'] = model
        else:
            print("    Warning: No trained model provided, skipping VoxelMorph")
            if not args.model:
                print("    Use --model to specify a trained model")

    return methods_dict, device


def main():
    """Main comparison function."""
    args = parse_args()

    print("=" * 70)
    print("fUS-VoxelMorph Baseline Comparison")
    print("=" * 70)
    print()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Prepare test cases
    test_cases = prepare_test_cases(args)

    # Prepare methods
    print("\nPreparing methods:")
    methods_dict, device = prepare_methods(args)

    if len(methods_dict) == 0:
        print("\nError: No methods available!")
        print("Install SimpleITK for traditional methods:")
        print("  pip install SimpleITK")
        sys.exit(1)

    print(f"\nComparing {len(methods_dict)} methods on {len(test_cases)} test cases")
    print(f"Device: {device}")
    print()

    # Run comparison
    all_results, summary_df = run_comparison_experiment(
        test_cases,
        methods_dict,
        output_dir=args.output_dir,
        device=device
    )

    # Print results
    print_comparison_table(summary_df)

    # Create figures
    print("\nGenerating comparison figures...")
    create_comparison_figures(all_results, output_dir=args.output_dir)

    # Save summary
    summary_path = os.path.join(args.output_dir, 'summary.txt')
    with open(summary_path, 'w') as f:
        f.write("fUS-VoxelMorph Baseline Comparison\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Test cases: {len(test_cases)}\n")
        f.write(f"Methods: {', '.join(methods_dict.keys())}\n")
        f.write(f"Device: {device}\n\n")
        f.write(summary_df.to_string(index=False))

    print(f"\n{'=' * 70}")
    print("Comparison complete!")
    print(f"Results saved to: {args.output_dir}/")
    print(f"  - comparison_table.csv")
    print(f"  - comparison_table.xlsx")
    print(f"  - metrics_comparison.png")
    print(f"  - raw_results.json")
    print(f"  - summary.txt")

    # Print key findings
    print("\n" + "=" * 70)
    print("Key Findings:")
    print("=" * 70)

    avg_df = summary_df[summary_df['Test Case'] == 'Average']

    if not avg_df.empty:
        # Best NCC
        best_ncc = avg_df.loc[avg_df['NCC (after)'].idxmax()]
        print(f"\nBest NCC (after):  {best_ncc['Method']} = {best_ncc['NCC (after)']:.4f}")

        # Best MS-SSIM
        best_ssim = avg_df.loc[avg_df['MS-SSIM'].idxmax()]
        print(f"Best MS-SSIM:      {best_ssim['Method']} = {best_ssim['MS-SSIM']:.4f}")

        # Best DSC
        best_dsc = avg_df.loc[avg_df['DSC'].idxmax()]
        print(f"Best DSC:          {best_dsc['Method']} = {best_dsc['DSC']:.4f}")

        # Fastest
        fastest = avg_df.loc[avg_df['Time (s)'].idxmin()]
        print(f"Fastest:           {fastest['Method']} = {fastest['Time (s)']:.4f}s")

        # Best Jacobian (lowest folding)
        deformable = avg_df[avg_df['Jac %Neg'] > 0]
        if not deformable.empty:
            best_jac = deformable.loc[deformable['Jac %Neg'].idxmin()]
            print(f"Lowest folding:    {best_jac['Method']} = {best_jac['Jac %Neg']:.2f}%")

    print("=" * 70)


if __name__ == '__main__':
    main()

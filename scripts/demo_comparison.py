"""
Demonstration of comparison between VoxelMorph and baseline methods.

This script demonstrates the comparison framework using synthetic data.
If SimpleITK is not installed, it will only compare with VoxelMorph.
"""

import os
import sys
import numpy as np
import torch

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import VxmDense2D
from data import SyntheticFUSGenerator
from utils.metrics import compute_ncc, compute_ms_ssim, compute_dsc


def create_mock_baseline(method_name):
    """Create a mock baseline method for demonstration."""

    class MockBaseline:
        def __init__(self, name):
            self.name = name

        def register(self, source, target):
            import time
            time.sleep(0.1)  # Simulate processing time

            if self.name == 'Identity':
                # Just return source
                return source, {'time': 0.001}

            elif self.name == 'Noisy':
                # Add some noise to simulate poor registration
                warped = source + np.random.normal(0, 0.05, source.shape)
                warped = np.clip(warped, 0, 1)
                return warped, {'time': 0.1}

            else:
                return source, {'time': 0.001}

    return MockBaseline(method_name)


def compare_with_baselines(source, target, model=None, device='cpu'):
    """Compare VoxelMorph with baseline methods."""
    results = {}

    print(f"\nImage shape: {source.shape}")
    print(f"Source range: [{source.min():.3f}, {source.max():.3f}]")
    print(f"Target range: [{target.min():.3f}, {target.max():.3f}]")

    # Identity baseline
    print("\n  Running Identity...")
    identity = create_mock_baseline('Identity')
    warped_id, params_id = identity.register(source, target)

    results['Identity'] = {
        'warped': warped_id,
        'ncc': compute_ncc(warped_id, target),
        'ms_ssim': compute_ms_ssim(warped_id, target),
        'dsc': compute_dsc(warped_id, target),
        'time': params_id['time'],
    }

    # VoxelMorph (if model available)
    if model is not None:
        print("  Running VoxelMorph...")
        model.eval()

        with torch.no_grad():
            src_tensor = torch.from_numpy(source[np.newaxis, np.newaxis]).float().to(device)
            tgt_tensor = torch.from_numpy(target[np.newaxis, np.newaxis]).float().to(device)

            import time
            start = time.time()
            warped_vm, flow_vm = model(src_tensor, tgt_tensor)
            vm_time = time.time() - start

            warped_vm = warped_vm[0, 0].cpu().numpy()

        results['VoxelMorph'] = {
            'warped': warped_vm,
            'ncc': compute_ncc(warped_vm, target),
            'ms_ssim': compute_ms_ssim(warped_vm, target),
            'dsc': compute_dsc(warped_vm, target),
            'time': vm_time,
        }

    return results


def print_comparison_table(results):
    """Print comparison table."""
    print("\n" + "=" * 70)
    print("Comparison Results")
    print("=" * 70)
    print(f"{'Method':<15} {'NCC':>10} {'MS-SSIM':>10} {'DSC':>10} {'Time (s)':>12}")
    print("-" * 70)

    for method_name, method_results in results.items():
        print(f"{method_name:<15} "
              f"{method_results['ncc']:>10.4f} "
              f"{method_results['ms_ssim']:>10.4f} "
              f"{method_results['dsc']:>10.4f} "
              f"{method_results['time']:>12.4f}")

    print("=" * 70)

    # Highlight best
    if len(results) > 1:
        print("\nBest by metric:")

        best_ncc = max(results.items(), key=lambda x: x[1]['ncc'])
        print(f"  NCC:      {best_ncc[0]} ({best_ncc[1]['ncc']:.4f})")

        best_ssim = max(results.items(), key=lambda x: x[1]['ms_ssim'])
        print(f"  MS-SSIM:  {best_ssim[0]} ({best_ssim[1]['ms_ssim']:.4f})")

        best_dsc = max(results.items(), key=lambda x: x[1]['dsc'])
        print(f"  DSC:      {best_dsc[0]} ({best_dsc[1]['dsc']:.4f})")

        fastest = min(results.items(), key=lambda x: x[1]['time'])
        print(f"  Speed:    {fastest[0]} ({fastest[1]['time']:.4f}s)")


def main():
    print("=" * 70)
    print("fUS-VoxelMorph Baseline Comparison Demo")
    print("=" * 70)
    print("\nThis demo compares VoxelMorph with baseline methods.")
    print()

    # Generate synthetic data
    print("Generating synthetic test data...")
    generator = SyntheticFUSGenerator(size=(128, 100), motion_type='mixed')
    source, target, flow_gt = generator.generate_pair(seed=42)
    print(f"  Generated: source {source.shape}, target {target.shape}")

    # Try to load VoxelMorph model
    model_path = 'checkpoints/best_model.pth'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = None
    if os.path.exists(model_path):
        print(f"\nLoading VoxelMorph model from {model_path}...")
        model = VxmDense2D(integration_steps=7).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        print("  Model loaded successfully")
    else:
        print(f"\nNo trained model found at {model_path}")
        print("  Run training first to include VoxelMorph in comparison")
        print("  Only Identity baseline will be shown")

    # Run comparison
    print("\nRunning comparison...")
    results = compare_with_baselines(source, target, model, device)

    # Print results
    print_comparison_table(results)

    # Save visualization
    print("\nGenerating visualization...")
    os.makedirs('demo_comparison', exist_ok=True)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(results) + 2, figsize=(4 * (len(results) + 2), 4))

    # Source
    axes[0].imshow(source, cmap='hot')
    axes[0].set_title('Source (Moving)')
    axes[0].axis('off')

    # Target
    axes[1].imshow(target, cmap='hot')
    axes[1].set_title('Target (Fixed)')
    axes[1].axis('off')

    # Methods
    for idx, (method_name, method_results) in enumerate(results.items()):
        axes[idx + 2].imshow(method_results['warped'], cmap='hot')
        axes[idx + 2].set_title(f"{method_name}\n"
                               f"NCC: {method_results['ncc']:.3f}\n"
                               f"Time: {method_results['time']:.3f}s")
        axes[idx + 2].axis('off')

    plt.tight_layout()
    plt.savefig('demo_comparison/comparison.png', dpi=150, bbox_inches='tight')
    print(f"  Saved to demo_comparison/comparison.png")

    # Try to run full comparison if SimpleITK available
    print("\n" + "=" * 70)
    print("Full Comparison")
    print("=" * 70)

    try:
        import SimpleITK as sitk
        print("\nSimpleITK is available!")
        print("Run the full comparison with:")
        print("  python compare_baseline.py --synthetic --n_samples 20 \\")
        if model is not None:
            print(f"    --model {model_path} \\")
        print("    --output_dir comparison_results/")
    except ImportError:
        print("\nSimpleITK not installed.")
        print("Install with: pip install SimpleITK")
        print("Then run: python compare_baseline.py --synthetic --n_samples 20")

    print("\n" + "=" * 70)
    print("Demo complete!")
    print("=" * 70)


if __name__ == '__main__':
    main()

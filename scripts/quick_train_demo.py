"""
Quick training demo with synthetic data.

This script runs a short training session to demonstrate
the training pipeline without requiring real data.
"""

import subprocess
import sys
import os

# Ensure we're in the project root
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("=" * 70)
print("fUS-VoxelMorph Quick Training Demo")
print("=" * 70)
print("\nThis demo runs a short training session with synthetic data.")
print("It will:")
print("  1. Generate 100 synthetic fUS image pairs")
print("  2. Train for 5 epochs")
print("  3. Save checkpoints to 'checkpoints_demo/'")
print("  4. Log to TensorBoard in 'runs/'")
print()

# Check if tensorboard is available
try:
    import tensorboard
    tensorboard_available = True
except ImportError:
    tensorboard_available = False
    print("Note: tensorboard not installed. Install with: pip install tensorboard")
    print()

input("Press Enter to continue...")
print()

# Build command
cmd = [
    sys.executable, "train.py",
    "--synthetic",
    "--synthetic_samples", "100",
    "--epochs", "5",
    "--batch_size", "4",
    "--lr", "1e-4",
    "--reg_weight", "1.0",
    "--save_dir", "checkpoints_demo",
    "--save_interval", "5",
    "--log_interval", "5",
    "--vis_interval", "10",
    "--augment",
    "--num_workers", "0",  # Single process for demo
]

if not tensorboard_available:
    cmd.append("--no_tensorboard")

print("Running command:")
print(" ".join(cmd))
print()
print("=" * 70)

# Run training
result = subprocess.run(cmd)

if result.returncode == 0:
    print("\n" + "=" * 70)
    print("Demo completed successfully!")
    print("=" * 70)
    print("\nNext steps:")
    print("  1. View TensorBoard logs:")
    print("     tensorboard --logdir=runs")
    print("  2. Use trained model for registration:")
    print("     python register.py --model checkpoints_demo/best_model.pth ...")
    print("  3. Run full training:")
    print("     python train.py --config configs/default.yaml")
else:
    print("\nDemo failed with error code:", result.returncode)
    sys.exit(1)

"""
Launch TensorBoard to view training logs.

Usage:
    python scripts/launch_tensorboard.py
    python scripts/launch_tensorboard.py --logdir custom_logs/
"""

import argparse
import subprocess
import sys
import os


def main():
    parser = argparse.ArgumentParser(description='Launch TensorBoard')
    parser.add_argument('--logdir', type=str, default='runs',
                       help='Directory containing TensorBoard logs')
    parser.add_argument('--port', type=int, default=6006,
                       help='Port to run TensorBoard on')
    parser.add_argument('--host', type=str, default='localhost',
                       help='Host to bind to')
    args = parser.parse_args()

    # Check if log directory exists
    if not os.path.exists(args.logdir):
        print(f"Error: Log directory '{args.logdir}' not found!")
        print("Run training first to generate logs.")
        sys.exit(1)

    # List available runs
    runs = [d for d in os.listdir(args.logdir) if os.path.isdir(os.path.join(args.logdir, d))]
    if runs:
        print("Available runs:")
        for run in sorted(runs):
            print(f"  - {run}")
    else:
        print("No runs found in log directory.")

    print(f"\nLaunching TensorBoard on http://{args.host}:{args.port}")
    print(f"Log directory: {args.logdir}")
    print("Press Ctrl+C to stop\n")

    # Launch TensorBoard
    try:
        subprocess.run([
            sys.executable, "-m", "tensorboard",
            "--logdir", args.logdir,
            "--port", str(args.port),
            "--host", args.host,
        ])
    except KeyboardInterrupt:
        print("\nTensorBoard stopped.")


if __name__ == '__main__':
    main()

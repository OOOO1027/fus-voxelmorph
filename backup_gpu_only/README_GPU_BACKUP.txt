=================================================================
       GPU-DRIVEN TRAINING SCRIPTS BACKUP
=================================================================

Backup Created: 2026-04-06 20:07:23

=================================================================
[GPU-DRIVEN TRAINING FILES]
=================================================================

1. train_gpu.py  (27,838 bytes)
   - GPU + Mixed Precision Training (AMP)
   - Uses torch.amp.autocast and GradScaler
   - Supports CUDA Automatic Mixed Precision for faster training
   - RECOMMENDED for modern GPUs (RTX 20 series and newer)

2. train_gpu_fp32.py  (27,852 bytes)
   - GPU FP32 Training (Full Precision)
   - No mixed precision (for GPUs without AMP support)
   - Use this if train_gpu.py causes issues

=================================================================
[EXCLUDED CPU-DRIVEN TRAINING FILES]
=================================================================

The following files are NOT included in this backup (CPU-oriented):

- train.py       - Base/generic version
- train_cpu.py   - CPU-optimized version  
- train_v2.py    - Version 2 (experimental)

=================================================================
[KEY DIFFERENCES: GPU vs CPU]
=================================================================

GPU VERSIONS (included):
- Use autocast() for mixed precision
- Use GradScaler for gradient scaling
- Force pin_memory=True for DataLoader
- Optimized CUDA memory management

CPU VERSIONS (excluded):
- No autocast/GradScaler
- Standard DataLoader settings
- Optimized for CPU computation

=================================================================
[USAGE]
=================================================================

# GPU with Mixed Precision (recommended)
python train_gpu.py --config configs/default.yaml

# GPU with FP32 only
python train_gpu_fp32.py --config configs/default.yaml

=================================================================

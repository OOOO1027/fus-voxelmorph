# Cross-Session Run Layout

`checkpoints/cross_session/` no longer stores mixed artifacts directly.

## Runs

- `runs/20260406_cpu_legacy_91k/`
  Legacy CPU run from April 6, 2026.
  Model size: 91,826 parameters.
  Historical single-scale NCC configuration.

- `runs/20260409_gpu_ms_ncc_337k/`
  Current GPU run split out from the previously mixed directory.
  Model size: 337,034 parameters.
  Multi-scale NCC + paired augmentation + reverse pairs + SWA configuration.

## Rule

- Start every fresh training run in a new dated subdirectory under `runs/`.
- Reuse an existing run directory only when resuming that exact run.
- `train_v2.py` now refuses to start a fresh run in a directory that already
  contains checkpoints or a training log.

## Notes

- The old report `reports/training_summary_cross_session_300epochs_20260406.md`
  was removed because it propagated outdated evaluation claims.
- Existing evaluation CSV files under `results/` are historical artifacts and
  should not be treated as proof for the April 9, 2026 GPU run unless re-run
  against the current checkpoints.

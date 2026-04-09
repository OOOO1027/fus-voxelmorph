# fUS-VoxelMorph DL-Prep Portable Bundle

This bundle contains the processed DL-Prep outputs needed to move the training/evaluation data to another machine.
It intentionally excludes `fus-voxelmorph/` and `fus_voxelmorph_guide.md` because those already exist on the destination computer.

Contents:
- `dl_prep/`: full copied snapshot of `Session_Level_Pairwise_Baseline/outputs/dl_prep`
- `reports/dl_prep_summary.md`: frozen DL-Prep summary report
- `portable/portable_training_assets_manifest.csv`: relative-path manifest intended for transfer/use on another machine
- `portable/*_native_npz.txt` and `portable/*_padded_npz.txt`: relative-path lists for train/val/test

Recommended entry on the other computer:
- Training data manifest: `portable/portable_training_assets_manifest.csv`
- Padded assets: `dl_prep/training_assets/npz/padded_canonical/`
- Native assets: `dl_prep/training_assets/npz/native/`
- Eval protocol: `dl_prep/eval/eval_protocol.json`
- Baseline reference: `dl_prep/eval/baseline_reference.csv`

Notes:
- `valid_mask` is included in every NPZ.
- `canvas_shape_hw` is auto-derived from the formal v1 DL pair dataset only.
- For the current dataset, the derived canonical canvas is `128 x 132`.

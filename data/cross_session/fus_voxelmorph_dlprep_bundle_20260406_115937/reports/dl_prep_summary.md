# DL-Prep Summary

## Current Scope
- Current frozen task is `cross-session anatomy-only image-to-image registration on neurovascular_map`, with optional Stage 3B rigid-prealigned moving input.
- Formal v1 DL pair dataset is the 24 Stage 3B eligible exact-match directed pairs only. This scope is frozen in `/home/zhangtaoouyang/new_work/Session_Level_Pairwise_Baseline/outputs/dl_prep/manifests/pair_dataset.csv` and `/home/zhangtaoouyang/new_work/Session_Level_Pairwise_Baseline/outputs/dl_prep/summaries/dl_prep_meta.json`.
- The auto-derived canonical canvas is `128 x 132`, and this result is explicitly derived from the formal v1 DL pair dataset only, not from Stage 1 all-session shape statistics. Evidence: `/home/zhangtaoouyang/new_work/Session_Level_Pairwise_Baseline/outputs/dl_prep/summaries/canvas_derivation_evidence.csv` and `/home/zhangtaoouyang/new_work/Session_Level_Pairwise_Baseline/outputs/dl_prep/summaries/dl_prep_meta.json`.
- Training interfaces are frozen for two anatomy-only inputs: `Training mode A = moving_raw + fixed_raw` and `Training mode B = moving_rigid + fixed_raw`.

## Current Exclusions
- `state / label / fDOP` are excluded from v1 primary training inputs and primary loss.
- Functional-primary registration is excluded.
- New classical tuning and nonrigid salvage are excluded.
- Stage 4A auxiliary results are not upgraded into the v1 training mainline.

## Current Benchmark
- Stage 3B is the unique default classical benchmark: `pure_failure_count=0`, `state_helpful_fraction=0.875`, `label_helpful_fraction=0.875`, `route_a_strengthened=True`. Evidence: `/home/zhangtaoouyang/new_work/Session_Level_Pairwise_Baseline/outputs/stage3b/summaries/route_stress_test_summary.csv`.
- Stage 3B remains a benchmark/reference, not perfect ground-truth deformation.
- Stage 4A is preserved only as an exploratory note. It does not replace the default baseline. Reference files: `/home/zhangtaoouyang/new_work/Session_Level_Pairwise_Baseline/outputs/stage4a/manifests/aux_manifest.csv` and `/home/zhangtaoouyang/new_work/Session_Level_Pairwise_Baseline/outputs/stage4a/comparisons/path_comparison.csv`.

## Dataset And Asset Contract
- Pair dataset main table is `/home/zhangtaoouyang/new_work/Session_Level_Pairwise_Baseline/outputs/dl_prep/manifests/pair_dataset.csv` with explicit `previous -> current` semantics, raw anatomy paths, rigid-prealigned anatomy path, transform paths, split membership, and Stage 4A annotations.
- Training assets are exported under `/home/zhangtaoouyang/new_work/Session_Level_Pairwise_Baseline/outputs/dl_prep/training_assets` in both `native` and `padded_canonical` NPZ forms. The master asset manifest is `/home/zhangtaoouyang/new_work/Session_Level_Pairwise_Baseline/outputs/dl_prep/training_assets/manifests/training_assets_manifest.csv`.
- `valid_mask` is a required field in both native and padded NPZ assets.
- Padding is center/symmetric with explicit `pad_top / pad_bottom / pad_left / pad_right`, no cropping, no resampling.
- For the first actual DL implementation, the recommended starting point is `Training mode B`, because it keeps the validated Stage 3B anatomy-first rigid backbone as the entry condition while still allowing an unsupervised image-to-image residual registration model.

## Split And Leakage Control
- `train`: `n_pairs=16`, `sessions=['S1', 'S14', 'S16', 'S2', 'S3', 'S5', 'S7', 'S9']`, `shape_group_counts={'(128, 127)': 8, '(128, 132)': 8}`, `pilot_count=2`, `hard_case_count=3`.
- `val`: `n_pairs=4`, `sessions=['S14', 'S16', 'S2', 'S5']`, `shape_group_counts={'(128, 127)': 2, '(128, 132)': 2}`, `pilot_count=1`, `hard_case_count=2`.
- `test`: `n_pairs=4`, `sessions=['S14', 'S2', 'S3', 'S9']`, `shape_group_counts={'(128, 127)': 2, '(128, 132)': 2}`, `pilot_count=1`, `hard_case_count=3`.
- `hard_case_subset`: `n_pairs=8`, `sessions=['S14', 'S16', 'S2', 'S3', 'S5', 'S7', 'S9']`, `shape_group_counts={'(128, 127)': 6, '(128, 132)': 2}`, `pilot_count=4`, `hard_case_count=8`.
- Reverse-direction pairs are co-located by undirected pair group, and no directed pair crosses splits. Rule source: `/home/zhangtaoouyang/new_work/Session_Level_Pairwise_Baseline/outputs/dl_prep/splits/split_meta.json`.
- Hard-case subset is `/home/zhangtaoouyang/new_work/Session_Level_Pairwise_Baseline/outputs/dl_prep/splits/hard_case_pairs.csv` and includes the 3 methodological drop pairs plus all Stage 3B tier2 diagnostic pairs with explicit `hard_case_reason`.
- A stricter session-aware holdout remains proposal-only and is documented in `/home/zhangtaoouyang/new_work/Session_Level_Pairwise_Baseline/outputs/dl_prep/splits/split_meta.json`; it is not the active v1 split.

## Future Extension Hooks
- Alternative anatomy-like anchors can be added later without changing the v1 anatomy-only contract.
- Rigid + residual hybrid models can be added later without changing the frozen Stage 3B baseline interface.
- Functional evaluation remains available only as an optional downstream hook through the evaluation protocol, not as a v1 training objective.
- Full-sequence downstream evaluation can be added later without changing the current pair-based training contract.

## Readiness
- The project now has a formal pair dataset, fixed train/val/test split, hard-case subset, exported training assets, a frozen baseline/eval protocol in `/home/zhangtaoouyang/new_work/Session_Level_Pairwise_Baseline/outputs/dl_prep/eval/eval_protocol.json`, and machine-readable meta in `/home/zhangtaoouyang/new_work/Session_Level_Pairwise_Baseline/outputs/dl_prep/summaries/dl_prep_meta.json`.
- This is sufficient to hand off to a subsequent VoxelMorph-style unsupervised model implementation phase without re-deciding pair semantics, split policy, benchmark target, or loader-facing asset format.

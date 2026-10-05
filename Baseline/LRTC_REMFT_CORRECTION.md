# LRTC-TMCP / ReMFT correction

This patch changes only LRTC-TMCP and ReMFT behavior. Project version remains 1.3.10 so completed results for unrelated baselines remain valid.

## LRTC-TMCP

- Keeps the multi-day tensor-completion formulation and 28 support days.
- Complete support slices are selected only from the frozen chronological Train interval in `splits/split_manifest.json`.
- Support uses the same within-day slot and is strictly earlier than the query.
- Validation/test hidden ground truth is never used as external support.
- The current H12 slice remains the exact incomplete cache query; observed current cells remain immutable.
- Implementation revision: `lrtc_train_archive_support28_r2`.

## ReMFT

- Restores the official notebook working scale `seq_len=96`, `pred_len=12`, `patch_len=16`, `stride=8`.
- Official PHI multi-scale mixers stay at their native H96 dimensions; the former H12 mixer reconstruction is removed.
- The last H12 is reused exactly from the shared benchmark cache.
- The extra causal H84 prefix is reconstructed from the raw timeline but masked under the same SupportBench view family/severity before exposure to the model. Natural invalid cells remain hidden.
- Formal micro-batch is restored to 1 with gradient accumulation 24; the existing 9000-update compute cap is retained as an explicit fixed optimizer-update budget rather than being attributed to the old H12 Development best epoch.
- Implementation revision: `remft_native96_sensingmatched_prefix_checkpointdepth_r4`.

## Required reruns

Only LRTC-TMCP and ReMFT must be rerun. Old checkpoints from these two implementations are intentionally rejected by `scripts/run_evaluation.py`.

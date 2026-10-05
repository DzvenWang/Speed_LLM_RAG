# Baseline Formal Protocol — v1.3.2

Dataset: `TraffiDent_Net_06`; history/horizon/stride = 12/12/12; seed = 10.
Targets are `speed` and `flow`.

Formal training and validation use severity s40. Completed Formal checkpoints are evaluated on paired s20/s40/s60/s90 evaluation caches. Generic metrics are Overall MAE/RMSE, Masked MAE/RMSE using `direct_mask`, and H3/H6/H12 metrics. Retrieval-defined support metrics are not part of the generic Baseline protocol.

Primary methods are ImputeFormer, LRTC-TMCP, GinAR, RAFT, RAST, SARAF, and ReMFT. ImputeFormer and LRTC-TMCP require the target-specific Frozen Backbone registry; when PatchSTG is selected, the previously validated PatchSTG consumer bridge is used unchanged. The other five Primary methods are independent of Frozen Backbone.

Base_SupportBench shared contract remains v1.2.1, ExperimentCache schema v3, metrics revision `traffident_metrics_v2`, and protocol version 2.1.0. Baseline v1.3.2 keeps the experiment protocol and model structures unchanged while making the Frozen Backbone registry generic; the validated PatchSTG bridge remains backward-compatible.


## LRTC-TMCP / ReMFT information-contract correction (2026-10-05)

- **LRTC-TMCP** keeps its multi-day location × day × within-day tensor. Its 28 complete support days are now selected **only from the frozen chronological Train archive**, at the same within-day slot and strictly before the query. Validation/test hidden ground truth is never used as external support. The current H12 slice remains the exact incomplete benchmark query and observed cells are immutable.
- **ReMFT** now retains the official notebook's native **H96 → P12** working scale with `patch_len=16` and `stride=8`. The final H12 is reused exactly from the shared benchmark cache. The extra causal H84 prefix is re-materialized under the same sensing view family/severity and exposes only observed cells, so native context is restored without granting complete held-out states.
- No other Baseline implementation or hyperparameter block is changed by this correction.

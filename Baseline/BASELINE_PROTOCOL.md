# Baseline Formal Protocol — v1.4.0

Dataset: `TraffiDent_Net_06`; history/horizon/stride = 12/12/12; seed = 10. Targets are `speed` and `flow`.

Formal training and validation use severity s40. Completed Formal runs are evaluated on paired s20/s40/s60/s90 evaluation caches. Generic metrics are Overall MAE/RMSE, Masked MAE/RMSE using `direct_mask`, and H3/H6/H12 metrics.

Primary methods are ImputeFormer, LRTC-TMCP, GinAR, RAFT, RAST, SARAF, and ReMFT. ImputeFormer and LRTC-TMCP use an explicitly selected trained Backbone (`--backbone <MODEL>`) for the final forecast. There is no global frozen-Backbone registry. The chosen checkpoint path and SHA256 are recorded in the Formal run and reused exactly for all subsequent evaluation severities.

Independent methods (GinAR, RAFT, RAST, SARAF, ReMFT) do not consume a reference Backbone. Matched-backbone retrieval controls are reported separately: existing PatchSTG controls remain intact, while RAST-STLAformer and RAFT-STLAformer use the exact selected STLAformer Formal checkpoint with recorded provenance.

LRTC-TMCP keeps its multi-day location × day × within-day tensor. Its 28 complete support days are selected only from the fixed chronological Train archive, at the same within-day slot and strictly before the query. Validation/test hidden ground truth is never used as external support. The current H12 slice remains the exact incomplete benchmark query and observed cells are immutable.

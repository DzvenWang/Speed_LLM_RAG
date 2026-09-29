# Baseline Formal Protocol — v1.3.2

Dataset: `TraffiDent_Net_06`; history/horizon/stride = 12/12/12; seed = 10.
Targets are `speed` and `flow`.

Formal training and validation use severity s40. Completed Formal checkpoints are evaluated on paired s20/s40/s60/s90 evaluation caches. Generic metrics are Overall MAE/RMSE, Masked MAE/RMSE using `direct_mask`, and H3/H6/H12 metrics. Retrieval-defined support metrics are not part of the generic Baseline protocol.

Primary methods are ImputeFormer, LRTC-TMCP, GinAR, RAFT, RAST, SARAF, and ReMFT. ImputeFormer and LRTC-TMCP require the target-specific Frozen Backbone registry; when PatchSTG is selected, the previously validated PatchSTG consumer bridge is used unchanged. The other five Primary methods are independent of Frozen Backbone.

Base_SupportBench shared contract remains v1.2.1, ExperimentCache schema v3, metrics revision `traffident_metrics_v2`, and protocol version 2.1.0. Baseline v1.3.2 keeps the experiment protocol and model structures unchanged while making the Frozen Backbone registry generic; the validated PatchSTG bridge remains backward-compatible.

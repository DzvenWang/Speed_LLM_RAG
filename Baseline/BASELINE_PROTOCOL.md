# Baseline Protocol v1.0.0

Baseline v1.0.0 is a clean-slate consumer of the frozen Shared Experiment Contract and the frozen PatchSTG backbone reference.

## Primary comparison

Primary methods are exactly:

`ImputeFormer, LRTC-TMCP, GinAR, RAFT, RAST, SARAF, ReMFT`.

All use the same TraffiDent Net-06 split, H=12, P=12, severity=0.4, frozen masks, metrics and prediction contract. Retrieval methods use Train-only memory. Test data are not available for checkpoint selection.

## No historical-result compatibility

Old Baseline experiment artifacts are not accepted. Formal selection requires:

- current `code_project_version == 1.0.0`;
- exact shared run identity;
- `status == COMPLETE`;
- `selection_valid == true`;
- the frozen PatchSTG checkpoint/seed when the method depends on PatchSTG.

## Reconstruction baselines

ImputeFormer and LRTC-TMCP forecast through the same frozen PatchSTG reference. Reconstruction cannot access forecast-horizon observations. LRTC-TMCP is additionally restricted to query-local H12 input in the main table.

## GinAR validity

Formal validation divergence is a hard invalidation event, not early stopping. No selectable artifact is written after divergence. A short stability probe on the full Formal training population is required before launching Formal GinAR.

## ReMFT validity

The frozen BERT runtime must match the local pretrained checkpoint in hidden size, head count and encoder depth. Runtime-only randomly initialized layers are removed before training; any remaining mismatch aborts the run.

## Secondary methods

SPRINT, SLPF, MoGERNN, LEAF, LEAF-Qwen1.5B and RAST-PatchSTG remain implemented but are reported only as secondary or controlled experiments.

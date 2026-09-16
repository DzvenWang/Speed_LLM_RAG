# v1.0.0 — Clean Primary-7 Baseline

- Locked the Primary Baseline suite to ImputeFormer, LRTC-TMCP, GinAR, RAFT, RAST, SARAF and ReMFT.
- Preserved SPRINT, SLPF, MoGERNN, LEAF, LEAF-Qwen1.5B and RAST-PatchSTG as secondary/controlled methods.
- Removed old-result version compatibility, migration and method-revision gates.
- Replaced multiple overlapping Primary PBS entry points with one clean smoke/development/formal workflow.
- Formal selection now accepts only current-version, current-identity, `selection_valid=true` results.
- GinAR divergence is now a hard Formal failure; added a full-Formal-population stability probe and more conservative Formal optimization settings.
- ReMFT now aligns frozen BERT runtime depth to the local pretrained checkpoint and hard-fails any remaining architecture mismatch.
- Retained strict causal H12 LRTC-TMCP + frozen PatchSTG main-table adaptation.

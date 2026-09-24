# Migration v0.4.2 -> v0.5.0

v0.5.0 is a scientific-model update, not a scheduler-only revision.

Changes:
- remove fixed Train-calibrated `alpha_n` from M3;
- add learned query-candidate-node scale, yielding selected `alpha_qn`;
- add M3 beneficial precision/recall/AUROC, Top-1/Top-4 hit rate and ranking accuracy;
- add M3 candidate eligibility mask for the M3->M4 interface;
- add Oracle-E@24, Oracle-E@M3-Top4 and Oracle-D@M3-Top4 diagnostics;
- add Train-only M4 node×horizon route/scale teacher pretraining;
- add Train-only M5 local intervention BCE teacher pretraining;
- retain one-V100 full-pipeline scheduling.

Old v0.4 M3/M4 checkpoints are intentionally rejected. Rerun M3 -> diagnostics -> M4 -> M5 under v0.5.

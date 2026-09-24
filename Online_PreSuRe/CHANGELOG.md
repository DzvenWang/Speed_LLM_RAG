# Changelog

## 3.3.0

- Replaced node-only composition with `node × 3-step temporal-patch` support units.
- Added `OnlineUtilityScorer`, a causal Helpful/Neutral/Harmful MLP trained only from matured downstream forecast outcomes.
- Added query-time counterfactual support probes; labels are attached only after the full future matures.
- Added support rejection/abstention and asymmetric harmful penalty.
- Added a patch-similarity control with identical candidate pool, granularity, completion and fusion to isolate the utility objective.
- Added `utility_labels.csv`, `utility_updates.csv`, `patch_supports.csv`, and `final_utility_scorer.pt`.
- Retained Perfect Completion and EntryOracle@8/16/32/64 diagnostics.
- Added an honestly named sampled single-patch downstream utility oracle over the probe set.
- Slow Online PatchSTG and fixed safe fusion are unchanged from the validated v3.2 protocol.

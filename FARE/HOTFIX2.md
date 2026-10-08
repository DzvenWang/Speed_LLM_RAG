# FARE v0.2.3 Hotfix 2

This hotfix preserves the v0.2.3 model/config/artifact schema and is intended to continue an interrupted v0.2.3 run.

Fixes:

1. Retrieval alpha-label rows now reuse rows that already have finite Retrieval candidate-utility labels. Smoke candidate assignments are sparse (160 train / 80 val); alpha supervision no longer re-samples a disjoint row set with a different seed.
2. Retrieval alpha-label selection stays balanced when the eligible pool is larger than the configured control-label budget.
3. A completed `retrieval/restorer_ranker.pt` is reused after a later-stage crash, so Stage-A Retrieval training is not repeated.
4. Existing Hotfix-1 strict-JSON handling is retained (NaN/Inf diagnostics become JSON null).

Expected continuation after the reported crash:

- REUSE_CONTEXT
- REUSE_RETRIEVAL_INDEX
- REUSE_RETRIEVAL_UTILITY
- REUSE_RETRIEVAL_RESTORER_RANKER
- RETRIEVAL_ALPHA_ELIGIBLE train=158 val=80 k=8 (numbers can vary by run/artifacts)
- RETRIEVAL_ALPHA_UTILITY ...
- RETRIEVAL_CONTROL_INIT ...
- RETRIEVAL_CONTROL_EPOCH ...

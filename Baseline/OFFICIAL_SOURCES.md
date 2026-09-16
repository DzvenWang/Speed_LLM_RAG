# Official Sources — Baseline v1.0.0

Pinned repositories and source policies are defined in `configs/official_sources.yaml` and frozen/verified by `scripts/source_lock.py`. Runtime adapters live in Baseline code; frozen vendor trees are not edited.

Primary Baseline sources include:

- `ImputeFormer` — exact pinned commit.
- `LRTC-TMCP` — official TRC 2024 repository, resolve-once from `main`, then exact commit/tree locked.
- `GinAR` — resolve-once, then locked.
- `RAFT` — exact pinned commit.
- `RAST` — exact pinned commit.
- `SARAF` — resolve-once, then locked.
- `ReMFT` — exact pinned commit `af61bef8ec872ba7b12e78675a39aeb0a20088a6`.

Secondary source trees (SPRINT, SLPF, MoGERNN, LEAF) remain in the same source-lock system because their implementations are preserved.

Run:

```bash
bash scripts/setup_official_models.sh
```

The first successful setup resolves every `resolve_once` source to a full commit SHA and writes `vendor/SOURCE_LOCK.json`; later setups reuse that resolved commit.

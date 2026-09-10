# Baseline v0.4.0 — Story-Aligned Shared-Protocol Baselines

This project is the baseline evaluation layer for **Forecast-Oriented Traffic State Inference with Support-Aware Prediction Control under Incomplete Sensing**. It consumes frozen Shared Protocol views only; it never owns Train/Val/Test sampling.

## Paper story

Main table is ordered by the alternative explanation each method represents:

1. **FrozenBackbone** — strong numerical forecasting alone.
2. **ImputeFormer + same frozen backbone** — reconstruction first, then forecast.
3. **GinAR (KDD 2024)** — end-to-end variable-missing forecasting.
4. **RAFT (ICML 2025)** — generic similarity-based retrieval augmentation.
5. **RAST (AAAI 2026)** — traffic-specific spatiotemporal retrieval augmentation.
6. **SARAF (KDD 2026)** — stationarity/reliability-aware retrieval augmentation.
7. **Ours** — query-specific predictive-support reasoning/control.

Separate comparisons:
- **LEAF (ACL Findings 2025)** — LLM-specific framework reference.
- **SLPF (TMLR 2025)** — fixed sensed/unsensed-node partial-sensing subtrack only.
- BiTGraph / PFRP are optional appendix/literature references.

## Shared Protocol

- Development: Train 12,000 + Val 5,000, 5 balanced sensing families.
- Formal: all eligible Train + exact frozen Development source Val views.
- Evaluation: Test 5,000 only after explicit authorization/freeze.
- Dynamic query signal: incomplete history.
- Retrieval memory: **Train only**, using historical full Train signal.
- Current query future is never available to retrieval or model selection.

## v0.4.0 runnable methods before final backbone freeze

`GinAR`, `RAFT`, and `SARAF` have external Development runners. Stock scripts that touch Test are never used.

```bash
qsub -v METHOD=GinAR pbs/methods/new_main_gpu_smoke.pbs
qsub -v METHOD=RAFT  pbs/methods/new_main_gpu_smoke.pbs
qsub -v METHOD=SARAF pbs/methods/new_main_gpu_smoke.pbs
```

After all three smoke logs end with `NEW MAIN GPU SMOKE PASS` and `SCRIPT_EXIT_STATUS=0`:

```bash
bash scripts/submit_prebackbone_main_development.sh
```

## Methods gated on final backbone

- ImputeFormer final result must be `ImputeFormer reconstructed history -> same frozen final backbone`.
- RAST matched formal result must use the same frozen final backbone as Ours where the interface permits.

Do not fabricate an early unmatched main-table result for either method.

## Upgrade from Baseline v0.3.1

The Development tensor materialization contract did not change. The v0.4.0 cache identity no longer depends on the list of baseline source repositories, so the existing v0.3.1 cache can be verified and migrated without rewriting arrays:

```bash
python scripts/migrate_cache_v040.py --stage development
python scripts/validate_cache.py --stage development
```

See `MIGRATION_v0.4.0.md`.

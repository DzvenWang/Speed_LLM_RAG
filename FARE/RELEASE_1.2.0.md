# FARE v1.2.0 — Strict Ablation & Target-Recovery Ranking

This is a **full standalone replacement** for v1.1.0, not a hotfix or a minimal prototype. Supports Net-01/02/03/04/06, speed/flow, the five training stages, checkpointed stage validation, OOF controller, formal multi-severity tests, Gadi auto Evaluate and independent manual Evaluate.

## Why this release

The v1.1.0 small Smoke ablations showed that `state_only` was almost identical to Full, and Flow sometimes performed better with Random Memory. On code audit, `state_only` disabled only a latent delta but left innovation-derived Cross-Attention tokens active; `no_risk` replaced the entire policy by a different confidence Gate. Thus these were not strict causal interventions.

## Main changes

1. `state_only` **never runs InnovationStudent**, passes no retrieval support tokens or latent delta, and disables innovation-based forecast residual.
2. `innovation_only` **never runs RetrievedStateRecovery**. Innovation receives observed x and retrieved historical latent tokens, with neutral state features; no state-restoration outputs are transferred.
3. `no_risk` uses the identical bounded UtilityRiskController with Risk omitted from the policy input. `no_utility` similarly omits Utility. `constant_alpha` uses the same valid mask and clamp, setting a global alpha configured in `ablation.constant_alpha` (default 0.12).
4. `no_memory` uses exactly the modified FARE-STLA input/graph/decomposition structure but no retrieval or restoration or innovation, and no retrieval candidate/utility gate. `--variant no_memory` enables **retrained** (not just inference disabled) architecture controls with phases `injection partial full`, separate artifact directories and a tiny unused memory contract. It does not run Retrieval/Recovery or generate train ranking labels.
5. `no_learned_reranker` and `no_observation_distance` split the original `no_mask_alignment` test into finer controls. `global_retrieval`/`random_memory`/`retrieval_off` remain available.
6. New training-only `fare/ranking.py`: for each **legal** retrieved candidate, compute supervised normalized restoration error only on the query's artificially missing and originally valid target cells in that candidate's own region. Cross-entropy over a soft label distribution trains Top-K Re-ranker weights; never send `x_full` or `future_y` to `FARE.forward`. This is a **state-recovery-quality proxy**, not an oracle of future forecast utility, nor a claim that retrieved candidates are optimal. Targets are masked out if there is no hidden ground-truth cell or fewer than two valid candidates.
7. Per-epoch ranking diagnostics: `ranking`, `ranking_eligible_fraction`, `ranking_selected_error`, `ranking_oracle_error`, `ranking_uniform_error`; `report_ranking.py` exports a CSV. The oracle here is **best within the legally retrieved Top-K** under recovery error, *not* an exhaustive 50k oracle.
8. `scripts/ablate.py` writes per-mode validation/test metrics to `ablation_summary.csv/json`, with distinct output subdirectories. Fixed-checkpoint ablations are explicitly labeled inference-only and are not a substitute for matched retraining.
9. Checkpoint embeds the complete v1.2.0 configuration. For formal training, v1.2.0 should train **new checkpoints**, not resume v1.1.0 optimizer states.

## Recommended initial commands

Windows:

```powershell
python -u FARE\scripts\smoke_all.py --dataset TraffiDent_Net_06 --seed 10
python -u FARE\scripts\ablate.py --checkpoint "<run_dir>\candidate_best.pt" --split val
python -u FARE\scripts\report_ranking.py --run-dir "<run_dir>"

# Matched no-memory retrained control — local smoke, separate run directory:
python -u FARE\scripts\train.py --dataset TraffiDent_Net_06 --target speed --seed 10 --variant no_memory --smoke
```

Gadi, two targets, auto Evaluate after successful training:

```bash
python -u FARE/scripts/submit_gadi.py train --dataset TraffiDent_Net_06 --targets speed flow --seed 10
# Retrained no-memory FARE-STLA comparator (also auto Evaluate):
python -u FARE/scripts/submit_gadi.py train --dataset TraffiDent_Net_06 --targets speed flow --seed 10 --variant no_memory
# Independent test of a saved checkpoint:
python -u FARE/scripts/submit_gadi.py evaluate --checkpoint /abs/path/to/candidate_best.pt --mode full
# Matched no-memory candidate must be evaluated with --mode no_memory if explicitly specified
```

## Caveats and research validity

- No efficacy is guaranteed from Smoke or synthetic tests. Do not claim TRC-level gains without formal, strict-protocol metrics for all five networks and both targets.
- `no_memory` is a **retrained control** of a matched FARE-STLA architecture; comparison against `retrieval_off` on a retrieval-trained FARE answers a different question.
- Ranking currently supervises retrieval using missing-state restoration MAE, *not* direct counterfactual forecast utility. Direct utility ranking would need leakage-safe held-out forecasts or OOF construction.
- The best candidate in a legal Top-K is not a bank-wide oracle. The Coarse Top-K is still discrete and is not fully differentiable. Query Encoder learns via representation alignment, soft candidate weights and local rank gradients.
- A mean alpha near initialization during a two-batch Smoke is not evidence that the OOF Controller is learned or calibrated.
- Gadi has a 48h queue walltime: multi-stage Formal training may require separately resumed phase jobs based on observed V100 epoch durations. This package does not claim an automatic PBS chain to resume truncated training.
- External runtime dependencies are only Base_SupportBench and Backbone's installed STLAformer vendor and checkpoint. Baseline is comparison-only. STLA_DASR, Support_Oracle, older FARE versions are NOT imported.

# PCFT-STLA v7.0.0 — Complete unified research source

This archive includes **unchanged original v6 source**, the prior propagation-identifiability tool, and a separate **complete v7.0 research implementation**. It does NOT include the external `Backbone/`, `Base_SupportBench/`, their raw training datasets, the official vendor STLA source, or pretrained `.pt` weights. Those must already exist in the `Speed_LLM_RAG` repository root.

## v7 architecture

- Native STLAformer complete forecast anchor (exact base.forward, no replacement forecast head), with a native last-attention forward hook for hidden features and a numerical parity assertion.
- C1: `ForecastSupport`, target observed availability separated from predicted source reliability and horizon-conditioned context.
- C2: `SparsePropagation`, historical valid source changes, physical directional candidate edges plus optional functional context, sparse per-horizon weights, a real null source, direct H1–H12 residual correction. Does **not** use real future source values.
- Local correction is separate from propagation correction.
- C3: `UtilityController` compares STLA / local / local+propagation candidates; predicts relative gain and harmful correction risk, supervised only on a temporally disjoint held-out TRAIN subset.
- A single model and four train stages: local, propagation, utility, joint (with disjoint-block utility refresh). Stage D can optionally unfreeze STLA end layers (default 0).
- Both fixed-s40 official input protocol, and an explicitly distinct `mixed_train_masks` mode generated from TRAIN-only raw sensor values using canonical Base_SupportBench sensing layouts. Validation and Test remain unchanged. Mixed mode is **not** claimed to be identical to the original fixed-s40 training protocol.

## Gadi install

Unzip into `/scratch/mh20/zw1165/Speed_LLM_RAG` and verify:

```bash
cd /scratch/mh20/zw1165/Speed_LLM_RAG
conda activate TFP
export SPEED_LLM_RAG_ROOT="$PWD"
export PYTHONPATH="$PWD/PCFT-STLA:$PWD/Backbone:$PWD/Base_SupportBench:${PYTHONPATH:-}"
bash PCFT-STLA/scripts/verify_v7.sh
bash PCFT-STLA/scripts/verify_v7.sh --preflight
```

The source verification uses the real target-specific SupportBench data and original official STLA checkpoint in preflight mode. Only submit a GPU smoke **after both Speed and Flow preflights pass**. This project has *not* been executed on your Gadi filesystem in this delivered environment.

### GPU smoke (submit before formal, no long training)

```bash
qsub -N PCFT7_S_smoke -v TARGET=speed,SEED=10,CHECKPOINT=Backbone_exps/speed/STLAformer/TraffiDent_Net_06/formal/20260930_215233_seed10/best.pt PCFT-STLA/pbs/smoke_v7.pbs
qsub -N PCFT7_F_smoke -v TARGET=flow,SEED=10,CHECKPOINT=Backbone_exps/flow/STLAformer/TraffiDent_Net_06/formal/20260930_220624_seed10/best.pt PCFT-STLA/pbs/smoke_v7.pbs
```

### Formal runs (ONLY after CUDA smoke passes)

```bash
qsub -N PCFT7_S -v TARGET=speed,SEED=10,CHECKPOINT=Backbone_exps/speed/STLAformer/TraffiDent_Net_06/formal/20260930_215233_seed10/best.pt PCFT-STLA/pbs/formal_v7.pbs
qsub -N PCFT7_F -v TARGET=flow,SEED=10,CHECKPOINT=Backbone_exps/flow/STLAformer/TraffiDent_Net_06/formal/20260930_220624_seed10/best.pt PCFT-STLA/pbs/formal_v7.pbs
```

The default config runs **fixed_s40** for fair comparison with v6. To run the optional multi-severity/failure-family TRAIN protocol instead, set `TRAIN_MASK_MODE=mixed_train_masks` as an additional `-v` PBS parameter, and **report it as a separate experiment**. The wrapper re-reads raw **TRAIN history only**; labels are unchanged and no test source truth enters the training inputs. `regional_outage`/`directional_outage` reuse the **existing** layout catalogs and fail if these are missing.

- Experiments: `PCFT-STLA_exps/<target>/incomplete/<dataset>/v7/formal/<timestamp>_seed10`
- `local_best.pt`, `propagation_best.pt`, `utility_best.pt`, `joint_best.pt`, `last.pt`, `epoch_history.jsonl`, `train_partition.json`, `test_results.json`.
- Restart from `last.pt` with `RESUME_CHECKPOINT=<path>`; the run folder and exact config/data/backbone identity must match. Do not change target, training masks, graph, seed or dataset on resume.
- Validation-only interventions (post-training, **not** retrained ablations) can be run via `PCFT-STLA/scripts/evaluate_v7.py`.

## Benchmarks and limitations

1. **Real-Gadi performance is not yet measured.** Synthetic CPU end-to-end pipeline and unit tests do not guarantee original vendor STLA compatibility, CUDA capacity, walltime, or performance. Preflight includes actual original-model parity; GPU smoke includes backward for all stages.
2. **v6 source retained verbatim**. v7 reuses v6 tested loading/cache/graph interfaces; it does not import its forecasting network. The v7 project does not overwrite any v6 experiment checkpoint.
3. The C3 gain target is based on counterfactual prediction *actions*, **not causal traffic propagation identification**.
4. Validation is used for stage selection; do not optimize thresholds on Test. Re-run independently trained ablations for formal causal attribution to model mechanisms; intervention outputs alone are not an independent ablation.
5. Stage D can unfreeze native STLA tail through `training.joint_unfreeze_stla_layers`; this changes the anchor from the original frozen checkpoint. Compare with same-budget STLA-only fine-tuning if enabled.
6. Full 50k TRAIN and 5k Validation windows reuse the **complete v6 cache identity** even for preflight/smoke, avoiding conflicts with existing full caches.
7. `torch.amp` requires the user's PyTorch 2.8 environment (known existing). NumPy, Pandas, PyYAML are reused from the TFP environment.

## Dedicated CPU diagnostics (integrated source)

See `diagnostics/v7_cpu/README.md` for the read-only CPU Validation diagnostics, C1/C2/C3 grouped reports, frozen-model source interventions, PBS commands and the result packaging script. Model architecture and training behavior are unchanged when diagnostics are disabled.

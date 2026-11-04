# Implementation and resource notes

## Model computation

1. Online trainable STLA prefix `E[0:4]` and first tail `E[4:6]` produce `H_pre` and `P0`.
2. Legacy C1 with fixed TRAIN Teacher Normal/CF + compatibility features computes candidate weights and independent `c1_posthoc` diagnostic.
3. Trainable C1 token head uses these same candidate weights and separately processes Normal/CF signals; None reliability suppresses unsupported tokens.
4. C2 creates horizon-specific Value and Query representations; separate observation coverage and C1 support reliability enter each graph edge.
5. Two node-local cross attentions map future support tokens to history positions, independently gated into the STLA prefix.
6. Shared original STLA tail runs again on adapted features. Original tail weights are **shared**, not cloned. Final `refined` uses the original STLA output head.

## Safety and implementation invariants

- No query future target enters `CoRSTA.forward`.
- Teacher/Replay is fixed while Student updates. C1 auxiliary utilities use only **TRAIN** targets.
- At selection time, validation targets are only consumed by metric calculations; chronological Selection and Locked Check remain separate.
- Reproducing the original STLA forward is required in preflight before allowing architectural evaluation; it is not a proof against all variants of upstream vendor code.
- Fixed teacher residuals may grow less compatible with a drifting Student; Student/Teacher output anchor and independently trained None selection mitigate this risk but do not prove it eliminated.
- C2 differentiability is checked in unit tests with and without valid neighbors. Sparse graph indexes are not trained; per-edge attention is.
- Each run identity fingerprints code/config/backbone. Same run path is never silently reused for different source.
- Original official STLA model files and weights are **not** redistributed.

## Resource estimates

Target Net_06: `N=388`, `L=P=12`, historical candidates `K=12`, feature tokens `d=32`.

FP16 candidate evidence tensor `[B,K,P,N,32]` = approx `3.41 MiB * B`: batch16 ~54.6MiB, batch64 ~218MiB. Additional C1 candidate logits, feature MLPS, STLA autograd and graph activations add memory; peak must be measured on V100.

C2 uses three graph branches up to 8+4+4 neighbors and horizon-dependent Q/K/V, rather than dense `N×N` global attention. Horizon bridge is node-local `[L,P]`, with L=P=12. Shared STLA late blocks are evaluated twice, while prefix only once. No GPU walltime claims are made without hardware profiling.

## Known limits for v0.1.0

- `direct_residual_head` is intentionally disabled. The C1 posthoc output is diagnostic, not a selectable final head.
- `batch_mode=auto` is CUDA-only; CPU smoke uses a configured simplified batch.
- In automatic selection, the GPU probe reports model allocated peak plus slack, not guaranteed end-to-end queue-level memory usage.
- Legacy bootstrap uses its own training initialization and checkpoints; C2 new dynamic graph starts from scratch while C1 weights are transferred.
- No student-aware dynamic memory rebuilding, differentiable top-k, or physical propagation delay constraint.
- Real official STLA source and Gadi Checkpoints are necessary to verify the production forward and actual V100 memory.


## CPU/GPU PBS Preflight (hotfix2)

- `pbs/preflight_cpu.pbs` requests CPU only; `pbs/preflight_gpu.pbs` remains available; both use `preflight_job.sh`.
- `--stage preflight` bypasses training-only CUDA enforcement, does not write formal run identity, and uses `CoRSTA_preflight` workspace by default.
- CPU PBS forces `--cpu`; GPU PBS forces `--require-cuda` and errors if CUDA is not accessible.
- Formal training still enforces CUDA and retains 16/32/64/128 adaptive batch sizing.
- `preflight` validates checkpoint load and baseline/split equivalence, but does not perform GPU training/memory profiling.

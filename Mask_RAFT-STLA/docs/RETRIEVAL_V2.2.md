# Mask_RAFT / Mask_RAST-STLA v2.2.0 (full research projects)

## Scope

Two independent projects, same **s40 incomplete-trained STLAformer** reference checkpoint, train-only complete historical memory. Supports target `speed`/`flow`, Net-01/02/03/04/06 (dynamic N), 12→12, four evaluation severities, persistent Gadi/PBS live logs, BEST/LAST/RESUME, automatic standard Evaluate and optional automatic Retrieval Audit. No historical-imputation decoder or reconstruction loss is introduced.

**Engineering status:** Python compile, model-level forward/backprop tests, simulated two-epoch Backbone/Cache CLI training for both projects, and paired audit on synthetic predictions have passed. **NOT** tested: real Gadi V100 end-to-end Formal, actual MAE/RMSE, multi-seed performance, online speed/memory after new regional retrieval. Existing v2.1 accuracy values are NOT v2.2 results.

## Core model changes

### RAFT: Teacher–student consistent, failure-disentangled residual evidence

- `mask_raft_stla/normal_teacher.py` **new**: persistent train-only `Y_train - STLA_teacher(X_train complete)` residual bank. Cache source keyed on checkpoint hash and Train memory metadata. Memory rows are not drawn from Validation/Test.
- Existing v2.1 masked replay bank supplies `Y_train - STLA_teacher(M_query * X_train)`. Their difference isolates **teacher's response to missingness**; the normal component captures residual prediction error even with complete history.
- Query teacher–student prediction discrepancy is provided to the candidate scorer. Per-candidate learned scales separately weight normal and failure residual components, and final correction remains evidence-anchored.
- Wrong-evidence intervention shuffles rank-to-future mappings **within each query**, independent of DataLoader Batch Size (a controlled mismatch, not a random sample from the entire memory).

### RAST: time-aligned future evolution + genuine regional retrieval

- Candidate historical anchor uses the current Query node's **last real observation time**, not always final history time; fully unobserved nodes use base-STLA-anchored future-shape transfer, without treating neighboring values as ground-truth node observations.
- `StateEncoder.region_proj`, Train-only statistical graph's deterministic connected-region assignments, `LatentMemoryIndex.region_keys` and `search_regions` build **region-specific first-stage candidate proposals** (not only rerank globally retrieved samples).
- Full model preserves global temporal+spatial candidates while also retrieving regional candidates; query/memory region projectors are kept consistent at index refresh; differentiable regional matching scores receive training gradients.
- Four named weights: Global Temporal / Global Spatial / Regional / None; `global_only` genuinely removes regional proposals while retaining trained global reranking.
- `training.base_preserve_weight` adds an auxiliary STLA base forecast loss to reduce joint-training-induced base deterioration. Defaults are a research hypothesis, NOT a proven improvement.

## Independent retrieval evaluation

New `scripts/evaluate_retrieval.py` for **both** projects. For each severity it obtains aligned Full / Base / Wrong Evidence / No Evidence / Component Ablation, plus **separately trained** No Retrieval (if user supplies its checkpoint). Outputs:

- `retrieval_audit/paired_horizon_family.csv`: each horizon and failure family, directly masked and not directly masked nodes
- `retrieval_audit/paired_summary.csv`: MAE/RMSE, negative transfer, within-checkpoint gains, real-versus-wrong gains, mechanism effect, and independent NoRetrieval gain (only if independently trained control exists)
- `retrieval_audit/audit_manifest.json`: experiment reference, checkpoint identity, explicit intervention caveats
- Standard `evaluation/s{severity}/{mode}/summary.json` also logs `useful_candidate_recall` and `weighted_candidate_utility_normalized`, as *oracle diagnostic* computed **only after inference** using labels, never as retrieval query features.

Important distinctions:

- `full` vs its own `base_prediction` shows conditional contribution, **not** causal independent Retrieval gain.
- `shuffled` is controlled rank-to-future mismatch and can be distribution-shifted; it should not replace retrained structural ablations.
- `no_retrieval` must be a **separate training job** initialized with the same STLA checkpoint (same data/seed/training budget). Do not evaluate a Full checkpoint's untrained control head and claim attribution.
- Model and loss revisions make v2.1 and v2.2 checkpoints **not compatible for resumption**. Use v2.1 projects to evaluate prior v2.1 Checkpoints, do not overwrite them.

## Deploy and run (Gadi)

Backup both old code directories first; expand combined archive in repository root. Use the exact known s40 STLA checkpoints, not the most recent complete-trained STLA.

```bash
cd /scratch/mh20/zw1165/Speed_LLM_RAG
export STLA_S40_SPEED="$(pwd)/Backbone_exps/speed/STLAformer/TraffiDent_Net_06/formal/20260930_215233_seed10/best.pt"
export STLA_S40_FLOW="$(pwd)/Backbone_exps/flow/STLAformer/TraffiDent_Net_06/formal/20260930_220624_seed10/best.pt"
test -f "$STLA_S40_SPEED" && test -f "$STLA_S40_FLOW" || exit 1

# 4 full training jobs
qsub -N MRAFT22_speed -v TARGET=speed,SEED=10,DATASET=TraffiDent_Net_06,BATCH_SIZE=64,BACKBONE_CHECKPOINT="$STLA_S40_SPEED" Mask_RAFT-STLA/pbs/formal.pbs
qsub -N MRAFT22_flow  -v TARGET=flow,SEED=10,DATASET=TraffiDent_Net_06,BATCH_SIZE=64,BACKBONE_CHECKPOINT="$STLA_S40_FLOW" Mask_RAFT-STLA/pbs/formal.pbs
qsub -N MRAST22_speed -v TARGET=speed,SEED=10,DATASET=TraffiDent_Net_06,BATCH_SIZE=64,BACKBONE_CHECKPOINT="$STLA_S40_SPEED" Mask_RAST-STLA/pbs/formal.pbs
qsub -N MRAST22_flow  -v TARGET=flow,SEED=10,DATASET=TraffiDent_Net_06,BATCH_SIZE=64,BACKBONE_CHECKPOINT="$STLA_S40_FLOW" Mask_RAST-STLA/pbs/formal.pbs
```

Training automatically evaluates all four severities, and Full training performs internal Real/Wrong/NoEvidence/Mechanism audit (without an independent No Retrieval control yet).

**Separately trained No Retrieval controls** (run 4 additional jobs for strict attribution):

```bash
qsub -N MRAFT22_NRspeed -v TARGET=speed,SEED=10,DATASET=TraffiDent_Net_06,ABLATION=no_retrieval,BATCH_SIZE=64,BACKBONE_CHECKPOINT="$STLA_S40_SPEED" Mask_RAFT-STLA/pbs/formal.pbs
qsub -N MRAFT22_NRflow  -v TARGET=flow,SEED=10,DATASET=TraffiDent_Net_06,ABLATION=no_retrieval,BATCH_SIZE=64,BACKBONE_CHECKPOINT="$STLA_S40_FLOW" Mask_RAFT-STLA/pbs/formal.pbs
qsub -N MRAST22_NRspeed -v TARGET=speed,SEED=10,DATASET=TraffiDent_Net_06,ABLATION=no_retrieval,BATCH_SIZE=64,BACKBONE_CHECKPOINT="$STLA_S40_SPEED" Mask_RAST-STLA/pbs/formal.pbs
qsub -N MRAST22_NRflow  -v TARGET=flow,SEED=10,DATASET=TraffiDent_Net_06,ABLATION=no_retrieval,BATCH_SIZE=64,BACKBONE_CHECKPOINT="$STLA_S40_FLOW" Mask_RAST-STLA/pbs/formal.pbs
```

Once a Full and its No Retrieval run have finished, merge their paired results:

```bash
python -u Mask_RAFT-STLA/scripts/evaluate_retrieval.py \
  --checkpoint /path/to/raft_full/best.pt \
  --no-retrieval-checkpoint /path/to/raft_no_retrieval/best.pt --reuse

python -u Mask_RAST-STLA/scripts/evaluate_retrieval.py \
  --checkpoint /path/to/rast_full/best.pt \
  --no-retrieval-checkpoint /path/to/rast_no_retrieval/best.pt --reuse
```

Use **same target and Backbone SHA** for each Full/NoRet pair. Existing `best.pt` is saved after every improved validation epoch, `last.pt` after each epoch, and `RESUME_CHECKPOINT` works for compatible v2.2 `last.pt` only. No change to the full 50,000 train queries, four severity Test grids, epoch schedule or candidate audit definition. Actual GPU performance and memory must be checked in logs, especially after Partial and Full Fine-tuning. The default Batch is 64; manually retry 48/32 if V100 OOM (automatic fallback is not implemented).

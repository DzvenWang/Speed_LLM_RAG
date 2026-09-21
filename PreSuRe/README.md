# PreSuRe v1.2 — Dual-Query Retrieval-Conditioned Forecasting

This is the first forecasting model after the Retrieval Information Test established two facts on TraffiDent_Net_06:

- Backbone-only retrieval is insufficient; the deployable **Backbone + sensing-evidence Dual Query** is materially better.
- Direct transfer/blending of a historical future or residual has very little value, so v1.2 does **not** copy support values into the query forecast.

## Main pipeline

```text
Incomplete query
  -> frozen PatchSTG state/forecast ------------------+
  -> sensing evidence (X, mask, transitions, graph) --+--> Dual Query
                                                        -> K0=24 leakage-safe candidates
                                                        -> learned Top-8 retriever
                                                        -> historical support episodes
                                                        -> support-set self-attention
                                                        -> node x horizon query-support cross-attention
                                                        -> standalone retrieval-conditioned forecast
PatchSTG forecast -------------------------------------+--> node x horizon adaptive forecast mixture
                                                        -> final forecast
```

No Qwen, Dual Memory, utility-sign classifier, veto, acceptance model, or direct residual transport is included.

## What changed from v1.0

1. Retrieval is fixed to the information-test winner: **Dual Query** rather than Backbone-only or raw similarity.
2. Candidate compression is `24 -> 8`, not `24 -> 4`.
3. Support episodes contain historical input/evidence, historical PatchSTG state, historical realized future, historical PatchSTG future, residual and validity.
4. Top-8 supports first communicate through support-set self-attention, then interact with the query at node x horizon resolution.
5. The retrieval branch predicts its own forecast and receives an explicit forecasting auxiliary loss.
6. The final controller mixes `PatchSTG forecast` and `retrieval-conditioned forecast`; it does not use the old near-zero safety gate.
7. The retrieval forecast head is zero-initialized, so the model still starts exactly at PatchSTG even though the mixture alpha starts at 0.25.
8. A small entropy-floor loss prevents immediate Top-8 -> Top-1 collapse without forcing uniform attention.

## Decision rule

Primary metric: **Masked MAE**.

- `GO_V13`: Development improvement >= 1.5% and Final improvement >= 1.0%.
- `CONDITIONAL_RETRIEVAL`: Development >= 0.5%, Final non-degrading.
- `STOP_RETRIEVAL_PRIMARY`: otherwise.

If v1.2 is positive, the next stage can add regime/dual-memory structure and later Qwen. If v1.2 remains below 0.5%, retrieval should be demoted from the primary performance source.

## Gadi run

```bash
cd /scratch/mh20/zw1165/Speed_LLM_RAG/PreSuRe
conda activate TFP

python tests/smoke_v12.py
python scripts/preflight.py --stage development
bash scripts/submit_v12_development.sh
```

The only main runtime log is:

```text
/scratch/mh20/zw1165/Speed_LLM_RAG/PreSuRe_logs/
PreSuReV12.development.seed10.<timestamp>.<jobid>.gadi-pbs.live.log
```

Every Live-log line is timestamped. PBS `.OU` is stored under:

```text
/scratch/mh20/zw1165/Speed_LLM_RAG/PreSuRe_logs/out/
```

No `submit.log` is created.

Follow:

```bash
tail -f /scratch/mh20/zw1165/Speed_LLM_RAG/PreSuRe_logs/PreSuReV12.development.seed10.*.live.log
```

## Outputs

```text
/scratch/mh20/zw1165/Speed_LLM_RAG/PreSuRe_exps/
DualQueryRetrievalConditionedForecasting/
  TraffiDent_Net_06/development/<timestamp>_seed10/
    best.pt
    summary.json
    DECISION.txt
    split_ids.npz
    horizon_metrics.json
    dev_pred.npy
    final_val_pred.npy
    dev_retrieval_pred.npy
    final_retrieval_pred.npy
```

Frozen intermediate artifacts and the Dual-Query retriever are under:

```text
PreSuRe_exps/_v12_artifacts/TraffiDent_Net_06/development/
```

## Key Live-log diagnostics

Each epoch reports:

- Development Masked MAE and improvement vs PatchSTG
- standalone retrieval-branch gain
- `alpha_mean/std`
- attention maximum
- normalized attention entropy
- mean absolute conditional delta
- CUDA peak allocated/reserved memory

This makes failure localization part of the normal training run rather than a separate diagnostic branch.


## v1.2.1 attention-batch fix
Support-set self-attention keeps the original node × horizon semantics but chunks the flattened `B*N*P` batch before `nn.MultiheadAttention`. This avoids the PyTorch efficient-attention 65,535 effective-batch limit (`16*388*12=74,496`) without reducing the formal training batch size.

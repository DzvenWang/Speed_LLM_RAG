# v3.0 Research Release — TRC C1/C2/C3

This source tree now defaults to **v3.0.0**, with independently trained M0–M3 contribution levels. **Read `TRC_V3_DESIGN.md` and repository-level `V3_RELEASE_NOTES.md` first.** Legacy v2 implementation details below are retained for compatibility; they do not supersede v3 training/evaluation instructions.

## v3 stable directories and formal jobs

The v3 code uses stable `PCFT-STLA_cache`, `PCFT-STLA_exps`, and
`PCFT-STLA_logs` top-level folders. The source checks cache compatibility
using versioned internal paths and checkpoint/parameter metadata.
`--smoke` is **optional** and is not required for submission to gpuvolta.
Submit `pbs/formal.pbs` directly from the Gadi login node after specifying
`TARGET`, `SEED`, `CONTRIBUTION`, `BACKBONE_MODE`, and a matching
`BACKBONE_CHECKPOINT`; this performs the preflight and full formal train on
a compute node. See `../V3_RELEASE_NOTES.md`.

---

# PCFT-STLA v2.0 — Time-coherent, Support-conditioned Future Transfer

Complete standalone research project with train-only graph and latent index building, STLAformer, global/regional/propagation retrieval, coherent lagged future transfer, action-utility model, training, evaluation, diagnostics and Gadi PBS. It does **not** load PCFT v1 or Mask_RAST Full model weights.

## Implemented mechanisms

1. **Local-regime predictive influence graph**: Train-only differenced traffic states, sparse directional lagged statistical influence with per-target local regimes and shrunk reliability (`propagation_v2.py`). This is *not* proof of physical traffic wave causation.
2. **Time-coherent historical propagation**: the original Train window ID is preserved **for each candidate and target node**. Edges connecting nodes from independently retrieved Regional windows are blocked for historical propagation transfer, although regional Anchor/Seasonal transfer remains supported (`model.py`, `propagation_v2.py`).
3. **Full-horizon support field**: observed history support plus uncertainty-discounted **Backbone-predicted** source support; never uses true query future or assumes future predictions are actual sensor measurements.
4. **Node-horizon action choice**: Legacy Anchor/Seasonal Future, coherent lagged Propagation Future, None (`action_v2.py`). The action selection is not directly multiplied by observed coverage; weights are end-to-end trainable and gain-supervised.
5. **Train-only retrieval and checkpoint index**: temporal/spatial/region/propagation keys with overlap exclusion, time validity and separate `best_memory_index.npz`.
6. **Formal metrics**: MAE primary, RMSE and MAPE secondary, four severities s20/s40/s60/s90 and failure families.

## Gadi deployment

Unzip from the `Speed_LLM_RAG` repository root. Keep external `Base_SupportBench`, `Backbone`, and respective STLA checkpoints. Use the TFP environment.

```bash
cd /scratch/mh20/zw1165/Speed_LLM_RAG
conda activate TFP
unset V23_CHECKPOINT RESUME_CHECKPOINT
python -u PCFT-STLA/scripts/preflight.py --target speed --dataset TraffiDent_Net_06 --backbone-checkpoint Backbone_exps/speed/STLAformer/TraffiDent_Net_06/formal/20260930_215233_seed10/best.pt
qsub -N PCFT2_speed -v TARGET=speed,SEED=10,BACKBONE_MODE=incomplete,ABLATION=full,BACKBONE_CHECKPOINT=Backbone_exps/speed/STLAformer/TraffiDent_Net_06/formal/20260930_215233_seed10/best.pt PCFT-STLA/pbs/formal.pbs
qsub -N PCFT2_flow -v TARGET=flow,SEED=10,BACKBONE_MODE=incomplete,ABLATION=full,BACKBONE_CHECKPOINT=Backbone_exps/flow/STLAformer/TraffiDent_Net_06/formal/20260930_220624_seed10/best.pt PCFT-STLA/pbs/formal.pbs
```

Each PBS job has 1 V100, 12 CPU, 32 GB, 48 hours, live log, automated Evaluate, versioned graph and repository index, and separate PCFT output directories. `num_workers=0`, CPU history repository and eval batch 8 protect against the previously observed SIGBUS / GPU OOM. Evaluation retries smaller batches on OOM.

## Evaluate and diagnose

```bash
python -u PCFT-STLA/scripts/evaluate.py --checkpoint /path/to/new/v2/run/best.pt --batch-size 8
python -u PCFT-STLA/scripts/diagnose.py --checkpoint /path/to/new/v2/run/best.pt --batch-size 8
```

Diagnostics: `full`, `legacy_only`, `propagation_only`, `no_lag_alignment`, `no_prop_retrieval`, `observed_support_only`, `no_action_utility`. Actions and support metrics are exposed in `model.forward()`. These are same-checkpoint tests; independent structural training ablations may be needed for causal claims.

## Tests

```bash
cd PCFT-STLA
PYTHONPATH=. python -m pytest -q
```

This is source-only, with synthetic CPU module and end-to-end tests, compilation and Bash checks. Real Gadi/Net-06 training and superiority over Mask_RAST v2.3 are **not verified**. All prediction inputs at inference come from Query history + Mask, STLA predictions and temporally legal Train-only memory.

## One-shot diagnosis (add-on)

The companion `../OneShot_Diagnostics` directory in the combined v2 release
provides a single PBS entry point for the four saved TCRR + PCFT checkpoints.
It runs full-severity, same-checkpoint mechanism ablations and traces the
Legacy / Propagation / None action weights and forecast/observed support in
one frozen forward pass per query. No model retraining or checkpoint mutation.

```bash
qsub -N TCRR_PCFT_DIA OneShot_Diagnostics/pbs/one_shot.pbs
```

See `OneShot_Diagnostics/README.md`; retain `best_memory_index.npz` on Gadi.

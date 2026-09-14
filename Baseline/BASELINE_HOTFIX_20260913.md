# Baseline v0.6.0 — RAST + LEAF TFP hotfix

Implementation revision: `20260913_rast_shim_leaf_tfp_r3`.

## RAST

`vendor/RAST` is unchanged and remains source-locked. Native RAST does not configure `pre_train_path` or `pre_train_model_name`, but upstream RAST eagerly imports optional STID/AGCRN classes. The Baseline-side compatibility shim supplies those import names only. It raises if either optional backbone is unexpectedly instantiated.

On Gadi:

```bash
cd /scratch/mh20/zw1165/Speed_LLM_RAG/Baseline
conda activate TFP
bash scripts/setup_rast_faiss.sh
python tests/smoke_rast_import_contract.py
bash scripts/resubmit_rast_development.sh
```

## LEAF

No separate LEAF environment is used. GraphBranch and HypergraphBranch run under:

`/scratch/mh20/zw1165/envs/TFP/bin/python`

Check and submit:

```bash
python scripts/check_leaf_tfp_runtime.py
bash scripts/resubmit_leaf_development.sh
```

The branch run saves `graph_val_raw.npy`, `hyper_val_raw.npy`, branch checkpoints and canonical branch metrics. The LLM selector remains post-hoc and consumes an official-style JSON with one row-major query-node record per entry:

```json
{"final_answer": 1}
```

where 1 selects GraphBranch, 2 selects HypergraphBranch, and null falls back to GraphBranch.

Submit a completed selector JSON with:

```bash
qsub -v STAGE=development,SEED=10,LEAF_SELECTOR_JSON=/path/to/selector.json pbs/methods/leaf_selector.pbs
```

No vLLM installation is performed inside TFP.

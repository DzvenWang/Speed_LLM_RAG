# Migration from PreSuRe v0.3.0 to v0.4.0

v0.4 changes the main deployable architecture. Old v0.3 M1/M3/M4/M5 checkpoints are intentionally incompatible and will be rejected by model-type guards.

After replacing the project files:

```bash
cd /scratch/mh20/zw1165/Speed_LLM_RAG/PreSuRe
conda activate TFP
python tests/smoke_static.py
python tests/smoke_oracle_math.py
python tests/smoke_node_routing.py
python tests/smoke_pipeline_v04.py
python scripts/preflight.py --stage development
bash scripts/submit_development.sh
```

The frozen PatchSTG artifacts may be reused if checkpoint/cache identity is unchanged. The retrieval build must rerun because v0.4 adds:

- `graph_adj.npy`
- `train_node_features.npy`
- `val_node_features.npy`
- `node_feature_manifest.json`

M2 regenerates v0.4 node embeddings:

- `train_node_embeddings_v04.npy`
- `val_node_embeddings_v04.npy`

Do not copy old v0.3 M1/M3/M4/M5 checkpoints into the new run directories.

`direct_mask` and `support_mask` are retained for training loss weighting and evaluation only. They are no longer deployable model inputs.

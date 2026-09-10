# Backbone v0.4.0 — recent-model extension

This project selects a strong frozen **numerical** traffic-forecasting backbone for the later predictive-support RAG/LLM system. It consumes the frozen Shared Protocol only; it never performs local view sampling and never uses Test for architecture selection.

## Candidate pool
`STAEformer`, `HimNet`, `BigST`, `STDN`, **`PatchSTG` (KDD 2025)**, **`DST-SGNN` (IJCAI 2025)**.

Stage-1A (the original four) can be reused from v0.3.1. Stage-1B adds only PatchSTG and DST-SGNN under the same Development Protocol, Train12k/Val5k, seed=10. The unchanged selection rule is applied across all six candidates. Formal then evaluates only the Top-2 with full eligible Train and seeds 10/20/30. Test stays locked until the final backbone is frozen.

## Gadi deployment
```bash
cd /scratch/mh20/zw1165/Speed_LLM_RAG
unzip -o Backbone_v0.4.0_PatchSTG_DSTSGNN.zip
cd Backbone
conda activate TFP
python -m pip install 'timm==1.0.12'
bash scripts/setup_official_models.sh
bash scripts/strip_vendor_git.sh
python scripts/verify_official_sources.py
python scripts/validate_cache.py --stage screening
```

Do **not** rebuild the current Development Protocol or screening cache merely for v0.4.0.

## GPU shape smoke before full screening
```bash
qsub -v MODEL=PatchSTG pbs/recent_adapter_smoke.pbs
qsub -v MODEL=DST-SGNN pbs/recent_adapter_smoke.pbs
```
Require `RECENT GPU SMOKE PASS` for both.

## Submit only the new candidates
```bash
bash scripts/submit_recent_screening.sh
qstat -u $USER
```

Each run goes through `pbs/backbone_select.pbs`, source-lock verification, model-aware preflight, Shared Protocol cache validation, and the common training/evaluation runner.

After both finish:
```bash
python scripts/summarize_screening.py
```
The summary now requires all six current-identity screening results and writes the new validation-only Top-2 shortlist.

## Important DST-SGNN note
The pinned official 2025 source hard-codes `cuda:0`; v0.4.0 deliberately does not edit vendor code. Therefore DST-SGNN must run on the one-V100 PBS path, not on a login/CPU node.

## Later RAG/LLM compatibility
Backbone internals remain untouched. The final model can later export frozen predictions/representations through an interface layer, preserving attribution of improvements to retrieval/reasoning/control rather than to a modified forecasting architecture.

# Deploy Baseline v1.0.0 on Gadi

Replace the Baseline project directory, then initialize official sources and the ReMFT local BERT checkpoint.

```bash
cd /scratch/mh20/zw1165/Speed_LLM_RAG/Baseline
conda activate TFP
bash scripts/setup_official_models.sh
bash scripts/setup_remft_bert.sh
python scripts/verify_official_sources.py
```

Then follow the Primary workflow in `README.md`. This release intentionally does not migrate or reuse old Baseline experiment artifacts.

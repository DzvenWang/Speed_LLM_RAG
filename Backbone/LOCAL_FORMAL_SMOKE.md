# Local Formal Smoke (v1.0.15)

From the repository root on Windows PowerShell:

```powershell
conda activate TFP
$env:SPEED_LLM_RAG_ROOT = (Get-Location).Path
$env:PYTHONUNBUFFERED = "1"
python -u Backbone\scripts\smoke_all_formal.py
```

This runs 12 sequential checks: STAEformer, HimNet, BigST, STDN, PatchSTG, STLAformer x speed/flow, seed 10.
Each check reads the real Formal cache but is bounded to 2 train batches and 1 validation batch by default. It performs forward, finite MAE loss, backward, optimizer step, validation, `smoke.pt` save/load, and target/model/seed identity validation. It does not run s20/s40/s60/s90 evaluation and does not freeze any checkpoint.

Single generic model example:
```powershell
python -u Backbone\scripts\run_model_formal.py --model STAEformer --target speed --seed 10 --smoke
```

PatchSTG example:
```powershell
python -u Backbone\scripts\run_experiment.py --target speed --stage formal --seed 10 --smoke
```

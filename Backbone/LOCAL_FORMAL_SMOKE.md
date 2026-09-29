# Local Formal Smoke

Backbone v1.1.3 supports a bounded end-to-end Formal smoke over all six models and both targets.

```powershell
cd C:\Users\zwan0652\ResearchProgram\Speed_LLM_RAG
conda activate TFP
$env:SPEED_LLM_RAG_ROOT = (Get-Location).Path
$env:PYTHONUNBUFFERED = "1"
python -u Backbone\scripts\smoke_all_formal.py --continue-on-error
```

Default coverage:
- models: STAEformer, HimNet, BigST, STDN, PatchSTG, STLAformer
- targets: speed, flow
- Formal training cache: s40
- training: 2 real batches
- validation: 1 real batch
- evaluation caches: s20, s40, s60, s90
- testing: 1 real batch per severity

The smoke path performs forward/backward/optimizer step, validation, smoke checkpoint save/reload, and bounded checkpoint evaluation on every severity. Full scenario-coverage assertions are intentionally skipped only when `--smoke-test-batches` is active; normal Formal evaluation remains unchanged.

Useful overrides:
```powershell
python -u Backbone\scripts\smoke_all_formal.py --train-batches 2 --val-batches 1 --test-batches 2 --continue-on-error
python -u Backbone\scripts\smoke_all_formal.py --training-only --continue-on-error
```

## Output layout (v1.1.4)
`Backbone_logs` is log-only. Evaluation artifacts are written under `Backbone_exps/<target>/<model>/TraffiDent_Net_06/checkpoint_evaluation/`.
The per-severity evaluator stores predictions/metrics/summary in its severity directory; the multi-severity wrapper stores only its combined result JSONs and summary CSVs under a `combined/<stamp>_seed<seed>/` directory.

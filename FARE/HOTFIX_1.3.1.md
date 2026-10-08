# FARE v1.3.1 — Real-cache gradient audit hotfix

## Failure reproduced from Net-06 Speed/Flow

`python -u FARE/scripts/diagnose_gradients.py --dataset TraffiDent_Net_06 --targets speed flow`
previously aborted on the first model forward with `RuntimeError: Memory index
not built; call retriever.refresh`. `runtime.setup()` intentionally constructs
Train-only history and associated latent tokens without its learned query/key
index; `FARETrainer` initializes that index at epoch start. The standalone audit
forgot the corresponding initialization on step one.

## Fix

- Build the Train-only retrieval index with `m.refresh(mem)` immediately after
  model/memory device placement and before any forwards.
- Keep refreshing after optimizer updates, since the retrieval encoder's key
  parameters may change between steps.
- Emit `FARE_GRADIENT_INDEX_READY` with target, memory size and key shape.
- Inspect non-finite *parameter gradients* in addition to finite loss/prediction.
- Never print a PASS when fewer than the requested audit steps execute.
- Add an end-to-end synthetic real-script regression test that begins with
  `mem.keys is None`, asserts first-forward index construction and two training
  steps for each of Speed and Flow.

## Windows PowerShell

```powershell
cd C:\Users\zwan0652\ResearchProgram\Speed_LLM_RAG
conda activate TFP
$env:SPEED_LLM_RAG_ROOT = (Get-Location).Path
python -u FARE\scripts\diagnose_gradients.py `
  --dataset TraffiDent_Net_06 `
  --targets speed flow `
  --steps 2
```

Expected: `FARE_GRADIENT_INDEX_READY`, two `FARE_GRADIENT_AUDIT` lines
and `FARE_GRADIENT_AUDIT_PASS` **for each target**. Checks that report some
zero norms on the first update can be normal with initially zero-weighted
support adapters; look at both steps. The synthetic tests do **not** replace
real Net-06 diagnostics or Gadi Pilot measurement.

## Scope and compatibility

No architecture, data contract, retrieval filtering, Ranking supervision,
Loss weighting, Gate behavior, Pilot/Formal training schedule, Gadi PBS or
checkpoint state-dict shape changed. Existing v1.3.0 weights can be evaluated
and resumed from this hotfix. No dependency on deleted experimental projects.

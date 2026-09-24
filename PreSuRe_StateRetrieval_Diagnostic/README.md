# PreSuRe State-Retrieval Diagnostic v0.9.0

This version narrows the diagnostic to the role that retrieval is intended to play under **incomplete sensing**:

> retrieve historical information that can complete missing traffic history, then compose those fragments before the frozen PatchSTG forecast.

The project no longer treats generic forecast-utility selection as the primary diagnostic question. v0.7/v0.8 scripts are retained only for reproducibility.

## v0.9 question

The diagnostic separates two problems:

1. **Fragment discovery** — does the historical Top-K pool contain useful local information for the missing region?
2. **Fragment composition** — is it better to combine pieces from multiple historical cases than to copy one whole patch/window?

The primary chain is:

```text
Incomplete history
    -> PatchSTG damaged patch
    -> existing mask-conditioned Top-K retrieval
    -> fragment the missing region
    -> select/compose historical fragments
    -> completed history
    -> frozen PatchSTG
    -> completion MAE + forecast MAE
```

## Fragment granularities

v0.9 evaluates increasingly fine completion units:

- `whole_patch`: one source for the whole damaged patch/suffix;
- `temporal1/2/3/6`: patch-wide temporal blocks;
- `node_track`: one source per node trajectory;
- `node_block3`: one source per node x three-step block;
- `cell`: per-cell oracle ceiling only.

`cell` is an upper-bound diagnostic, not a proposed final method.

## Selection/composition modes

- `oracle_completion`: uses hidden **history** truth only to identify the best completion fragment. It never uses the future forecasting target. This is a diagnostic ceiling.
- `global_top1`: uses the original patch retrieval order.
- `local_observed`: reranks each fragment using only observed local temporal/spatial context around that missing fragment.
- `local_softmax`: composes multiple candidates per fragment using observable local-context scores.

Observed cells are never overwritten.

## Recommended run from the existing v0.6-v0.8 experiment

The current experiment already contains:

```text
patchdiag_v051.20260923_205431/patch_retrieval/patch_retrieval_topk.npz
```

Run only the new v0.9 diagnostic:

```powershell
cd C:\Users\zwan0652\ResearchProgram\Speed_LLM_RAG\PreSuRe_StateRetrieval_Diagnostic

python scripts\run_v09_from_existing.py --stage development --run-tag patchdiag_v051.20260923_205431 --scopes 12 --top-k 50 --transport affine_alignment --eval-batch-size 64
```

If GPU memory is limited, use `--eval-batch-size 32`.

## Logs

The runner prints to the terminal and writes:

```text
PreSuRe_StateRetrieval_Diagnostic_logs/
  TraffiDent_Net_06/development/patchdiag_v051.20260923_205431/
    15_fragment_completion_diagnostic.log
    16_fragment_summary.log
```

Progress is printed every five queries.

## Outputs

```text
fragment_completion_diagnostic/
  per_query.csv
  fragment_selections.csv.gz
  summary_table.csv
  summary.json

FRAGMENT_DIAGNOSTIC_SUMMARY.md
fragment_diagnostic_summary.json
```

The main comparison in `summary_table.csv` is:

```text
whole-patch completion oracle
vs
structured fragment completion oracle
vs
local-observed / local-softmax composition
```

## Decision logic

The diagnostic is intended to decide the next model, not to be the final model itself.

- If structured fragment Oracle is substantially better than whole-patch Oracle, fragment discovery/composition is justified.
- If `local_observed` or `local_softmax` closes part of that gap, fragment-specific retrieval scores are already useful and can be learned/refined.
- If fragment Oracle improves completion strongly but forecasting remains weak, the next model should learn a PatchSTG-aware fragment composer instead of hard replacement.
- If even the fragment Oracle cannot improve completion/forecasting, historical retrieval is not a strong source for this missing pattern and should not be forced into PreSuRe.

## Leakage boundary

Inference-available modes use only:

- incomplete query history;
- observation mask;
- train-split historical candidates;
- time context already available at inference.

Hidden history truth is used only by `oracle_completion` for diagnostic analysis. Future forecasting targets are never used to choose fragments in v0.9.

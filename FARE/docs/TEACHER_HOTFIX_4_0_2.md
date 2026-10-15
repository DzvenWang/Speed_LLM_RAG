# FARE v4.0.2: Teacher legal-replacement sampling hotfix

## Observed error
On Gadi TFP (torch 2.8.0+cu128), `python -m pytest -q FARE/tests` failed
three tests with `IndexError` at `fare/teacher.py:44`, previously using
`choices[empty,0] = 1` on a `[B,N,K]` tensor where `empty` is `[B,N]`.

## Fix
`sample_legal_replacement(pool)` samples only valid candidate positions.
Rows with no unselected valid candidate use position 0 as a placeholder and
have `replacement_valid == False`. This is excluded from swap loss by
`replay_loss`; it does not add an invalid historical episode to the selected
set. Shapes `[B,N,K]` tested for all-empty, mixed, and singleton candidate
cases.

## Verification
`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH="$PWD/FARE" python -m pytest -q FARE/tests`

In local CPU environment (torch 2.10): 28 passed, 1 expected exploratory-
residual warning. Gadi torch 2.8/V100 remains to be confirmed by user.

## Installation
From Speed_LLM_RAG root:

```
unzip -o FARE_v4.0.2_TeacherSampling_Hotfix_Overlay.zip
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q FARE/tests
```

## Compatibility
No Backbone/Checkpoint architecture changes, no Base_SupportBench schema
changes, no new dependencies or CUDA requirements. `training.py` continues
storing checkpoint format version 4.0.1 for backward resume compatibility.

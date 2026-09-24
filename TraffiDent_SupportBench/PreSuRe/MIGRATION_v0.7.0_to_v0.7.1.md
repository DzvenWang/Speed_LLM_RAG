# PreSuRe v0.7.0 -> v0.7.1

## Purpose

v0.7.1 does **not** introduce the proposed v0.8 forecasting architecture. It adds a development-only diagnostic track to answer a prerequisite question:

> Is future PatchSTG correction need predictable from deployment-safe information under incomplete sensing?

The v0.7 Development result showed a large gap between SoftMix@M3-Top4 headroom and the learned M4 student. Before redesigning M4 again, v0.7.1 separates the problem into two probes.

## New Probe 1 — Predictive Need / Error-State

Train-only target:

`E*(q,h,n) = Y(q,h,n) - Y_base(q,h,n)`

Inference inputs remain deployment-safe node context only. The probe reports raw MAE/RMSE, sign accuracy on meaningful errors, Pearson/Spearman correlation, high-error-group performance, and improvement over the zero-error predictor.

## New Probe 2 — Support Utility A/B/C

M3 Top-4 is frozen. The same utility-probe architecture is trained three times with identical initialization:

- A: query + support only;
- B: A + predicted need from Probe 1;
- C: A + true future need, privileged diagnostic only.

The main readouts are pairwise utility-ranking accuracy, Top-1 agreement, beneficial Top-1 hit, Spearman/Pearson and beneficial AUROC.

Interpretation:

- B >> A: a learned compatibility model benefits from predicted Need.
- B_ANALYTIC strong: predicted Need itself is sufficient to rank residual supports without another learned utility student; this is the cleanest evidence for v0.8.
- C >> B but B ~= A: utility is conditionable on need, but Need prediction is the bottleneck.
- C weak: even privileged need does not make support utility sufficiently predictable; do not build v0.8 around this formulation.

## Leakage-safe split

The 12k Development Train queries are split into mutually exclusive subsets:

- Need train: 7000
- Need dev: 1000
- Utility train: 3500
- Utility dev: 500

A deterministic 1200-query subset of Validation is used only for the final diagnostic. Utility-C is explicitly non-deployable.

## New files

- `presure/probes.py`
- `scripts/run_predictive_need_probe.py`
- `scripts/probe_preflight.py`
- `scripts/submit_predictive_need_probe.sh`
- `pbs/predictive_need_probe.pbs`
- `tests/smoke_predictive_need_probe.py`

## Run

After a completed v0.7 Development run:

```bash
cd /scratch/mh20/zw1165/Speed_LLM_RAG/PreSuRe
conda activate TFP
python tests/smoke_predictive_need_probe.py
python scripts/probe_preflight.py --stage development
bash scripts/submit_predictive_need_probe.sh
```

This submits one V100 job and reuses existing v0.7 artifacts/M1/M3. It does not rerun Oracle A-F or M0-M5.

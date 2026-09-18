# PreSuRe v0.9.5 — Query Error-State Representation

v0.9.4 showed that the existing query representation is the current bottleneck: on masked hard-error cells, q75 direction AUROC was only about 0.556, q90 AUROC about 0.580, and signed-error Spearman about 0.058. v0.9.5 therefore does **not** add another retrieval or acceptance policy. It asks whether richer query-side predictive evidence can recover the missing future correction state.

The new probe augments the existing incomplete-history/mask/neighbour/time/PatchSTG-forecast representation with three auditable channels:

1. **Frozen PatchSTG predictive hidden state.** A forward-hook discovery pass chooses a deep node-aligned latent activation from the exact frozen PatchSTG checkpoint. The selected module name, raw shape and deterministic projection are recorded in `query_error_state_artifact_manifest.json`.
2. **Strictly causal recent backbone-error state.** Previous-query PatchSTG residuals are exposed only after their forecast timestamps are already realized at the current query time. If cache chronology cannot be proved, this channel is explicitly disabled and zero-filled rather than guessed.
3. **Physical transition state.** Query-history-only slope, acceleration, local shock, neighbour gradient and sensing-continuity descriptors distinguish transition regimes such as onset and recovery.

The experiment trains controlled `base`, `hidden`, `recent_error`, `physical`, `hidden_physical`, and `full` variants. **Train-dev alone selects the representation variant and direction threshold; Final Validation never selects the model.** Retrieval candidates, support residuals, Oracle choices, Qwen and Dual Memory are not inputs.

```bash
cd /scratch/mh20/zw1165/Speed_LLM_RAG/PreSuRe
conda activate TFP
python tests/smoke_query_error_state_probe.py
python scripts/prepare_query_error_state_artifacts.py --stage development
python scripts/query_error_state_preflight.py --stage development
bash scripts/submit_query_error_state_probe.sh
```

The PBS job `QErrState` automatically prepares missing/current-contract hidden/error-state artifacts before running the probe. Outputs are written to `PreSuRe_exps/QueryErrorStateProbe/TraffiDent_Net_06/development/<timestamp>_seed10/`. The main files are `summary.json`, `variant_comparison.csv`, `DECISION.txt`, and one best checkpoint per ablation. See `MIGRATION_v0.9.4_to_v0.9.5.md`.

---

# PreSuRe v0.9.4 — Query Future-Error Direction Probe

v0.9A.3 established a clean negative result: the frozen Top-1 support has positive utility only about half of the time, and the selective acceptance head cannot reliably distinguish positive from non-positive utility. v0.9.4 therefore stops adding acceptance logic and diagnoses the missing information directly.

The new **Query Future-Error Direction Probe** predicts the signed future PatchSTG error

```text
E* = Y_future - Y_base
```

from **query-side deployment-safe information only**. Positive direction means PatchSTG underprediction and a positive correction demand; negative direction means overprediction. The probe uses the existing node-local incomplete-history/mask/neighbour/time/sensing descriptors and frozen PatchSTG forecast trajectory. It does not use retrieval candidates, support residuals, Oracle choices, Qwen, or current-query future as an input.

The probe reuses the exact v0.9A train/dev/final-validation IDs. Horizon-specific q50/q75/q90 absolute-error thresholds are derived from Train only. A signed-error regression head is trained on all valid cells while the direction head is trained only where |E*| exceeds Train q50 to reduce near-zero sign noise. Primary diagnostics are Masked q75/q90 AUROC and balanced accuracy, signed-error Spearman correlation, and horizon-wise direction metrics.

```bash
cd /scratch/mh20/zw1165/Speed_LLM_RAG/PreSuRe
conda activate TFP
python tests/smoke_query_future_error_direction_probe.py
python scripts/query_future_error_direction_preflight.py --stage development
bash scripts/submit_query_future_error_direction_probe.sh
```

PBS job name: `QErrDir`. Outputs are written under `PreSuRe_exps/QueryFutureErrorDirectionProbe/TraffiDent_Net_06/development/<timestamp>_seed10/`. See `MIGRATION_v0.9.3_to_v0.9.4.md`.

---

# PreSuRe v0.9.3 — v0.9A.3 Selective Utility Acceptance

v0.9A established that node-horizon predictive utility can be ranked well on the fixed leakage-safe K=24 pool, but absolute support acceptance remained weak. v0.9A.1 showed that jointly fine-tuning ranking and safety damages the strong v0.9A ordering. v0.9A.2 then showed that an independent strong-harm veto can remove catastrophic supports, but deep fallback over-vetoes useful Top-1 supports and often lands on lower-ranked mildly harmful candidates.

v0.9A.3 therefore changes the question from **"which candidates are harmful?"** to **"should the support already selected by the frozen v0.9A ranker actually be used?"**. The v0.9A ranker, its risk beta, candidate pool and split IDs are fully frozen. A small `SelectedSupportAcceptanceHead` is trained only on the frozen Top-1 support for each node-horizon cell. Its binary target is whether that selected support has **positive predictive utility**. Near-zero utility is down-weighted rather than relabelled.

The acceptance head receives the frozen Top-1 role representation plus deployment-safe rank context: `mu`, `sigma`, beneficial logit, frozen risk score, Top1-Top2/Top1-Top3 margins, candidate-score mean/std, Top-1 z-score, normalized rank entropy, candidate availability and relative uncertainty. It cannot re-rank or choose a lower candidate.

Deployment policy is deliberately simple:

```text
Frozen v0.9A Top-1
        |
Selected-support acceptance probability
        |
        +-- p >= threshold --> USE Top-1
        |
        +-- p <  threshold --> ABSTAIN / retain PatchSTG
```

The acceptance threshold is calibrated on the original v0.9A Train-dev split by policy utility with an explicit minimum coverage of 20%. There is **no fallback**, **no re-ranking**, **no Dual Memory**, and **no Qwen** in this controlled probe.

## Recommended Gadi run

```bash
cd /scratch/mh20/zw1165/Speed_LLM_RAG/PreSuRe
conda activate TFP

python tests/smoke_predictive_utility_selective_acceptance.py
python scripts/predictive_utility_selective_acceptance_preflight.py --stage development
bash scripts/submit_predictive_utility_selective_acceptance_probe.sh
```

PBS job name: `PreUtilA3`.

Main outputs are written under:

```text
PreSuRe_exps/PredictiveUtilitySelectiveAcceptance/TraffiDent_Net_06/development/<timestamp>_seed10/
```

Important files: `selected_support_acceptance_best.pt`, `summary.json`, `policy_comparison.csv`, `threshold_grid_masked.csv`, and `DECISION.txt`.

Primary GO targets are Masked coverage >= 0.20, accepted beneficial >= 0.65, accepted harmful <= 0.30, and safe-policy regret reduction >= 10% relative to the same frozen v0.9A Top-1 policy. See `MIGRATION_v0.9.2_to_v0.9.3.md`.

---

# PreSuRe v0.9.2 — v0.9A.2 Rank-then-Veto

The completed v0.9A run learned strong node-horizon relative utility ranking, while v0.9A.1 showed that jointly fine-tuning Top-1/sign/safety objectives can damage that ranking even though strong-harm recognition improves. v0.9A.2 therefore **separates ranking from safety**.

The completed v0.9A reranker is loaded from `PreSuRe_exps/PredictiveUtilityRetrieval/.../LATEST.json` and fully frozen. A small independent `StrongHarmVetoHead` learns only whether a candidate is strongly harmful. During deployment, candidates are inspected strictly in the frozen v0.9A rank order. The veto may skip a dangerous candidate and fall back to the next-ranked candidate, but it can never re-order safe candidates. If all inspected candidates are vetoed, the policy ABSTAINs and retains PatchSTG.

The experiment reuses the exact v0.9A K=24 pool and the exact v0.9A Train/Train-dev/Final-Validation split IDs. Veto threshold and inspection depth are calibrated on Train-dev only. Dual Memory and Qwen remain disabled so any change is attributable to independent harm filtering.

## Recommended Gadi run

```bash
cd /scratch/mh20/zw1165/Speed_LLM_RAG/PreSuRe
conda activate TFP

python tests/smoke_predictive_utility_rank_then_veto.py
python scripts/predictive_utility_rank_then_veto_preflight.py --stage development
bash scripts/submit_predictive_utility_rank_then_veto_probe.sh
```

The PBS job name is `PreUtilA2`. Main outputs are `summary.json`, `policy_comparison.csv`, and `DECISION.txt` under `PreSuRe_exps/PredictiveUtilityRankThenVeto/...`. See `MIGRATION_v0.9.1_to_v0.9.2.md`.

---

# PreSuRe v0.9.1 — v0.9A.1 Safe Top-Selection

v0.9A showed that unified node-horizon predictive-utility learning can rank the fixed K=24 candidate pool much better than coarse similarity, but its completed development run remained NO_GO because top-selection safety lagged behind pairwise ranking.  v0.9A.1 keeps the same candidate pool and the same reranker architecture, warm-starts the completed v0.9A checkpoint, and changes only the objective and deployment policy.

The new training objective adds **listwise best-candidate learning**, focal beneficial/harmful sign separation, and an absolute score barrier while retaining utility regression and similar-but-harmful hard-negative ranking.  The deployment score is:

```text
score = mu + lambda * beneficial_logit - beta * sigma
```

`beta` and `lambda` are selected on Train-dev.  A numerical **ABSTAIN/no-support threshold** is then calibrated on Train-dev with an explicit minimum coverage of 35%.  If the top support does not clear this threshold, the protected PatchSTG forecast is retained.  This avoids forcing a harmful support while also preventing the degenerate solution of rejecting every query.

Dual Memory and Qwen remain intentionally disabled in v0.9A.1.  The controlled question is: **can the already strong v0.9A utility representation be converted into safer Top-1 support selection and materially lower regret?**

## Recommended Gadi run

```bash
cd /scratch/mh20/zw1165/Speed_LLM_RAG/PreSuRe
conda activate TFP

python tests/smoke_predictive_utility_safe_selection.py
python scripts/predictive_utility_safe_selection_preflight.py --stage development
bash scripts/submit_predictive_utility_safe_selection_probe.sh
```

The development run warm-starts the completed v0.9A checkpoint referenced by `PreSuRe_exps/PredictiveUtilityRetrieval/.../LATEST.json`.  Main outputs are `summary.json`, `comparison.csv`, `policy_comparison.csv`, and `DECISION.txt` under `PreSuRe_exps/PredictiveUtilitySafeSelection/...`.

The primary GO targets are Masked pairwise rank >= 0.67, coverage >= 0.35, accepted beneficial >= 0.60, accepted harmful <= 0.35, and safe-policy regret reduction >= 20% versus coarse no-abstain selection.  See `MIGRATION_v0.9.0_to_v0.9.1.md`.

---

# PreSuRe v0.9.0 — v0.9A Predictive-Utility Retrieval

v0.9A moves the main modelling effort from post-hoc routing to **support-value identification before Qwen**.  It keeps the existing leakage-safe retrieval bank and uses the same first 24 candidates, but removes the legacy M3 Top-4 compression.  One unified reranker scores every candidate at **node x horizon** granularity.

The reranker consumes only deployment-safe query/support state, sensing descriptors, historical Train residual trajectories, PatchSTG forecast relations, and coarse retrieval confidence.  Current-query future is used only to construct Train utility labels and diagnostic Validation metrics.

For each candidate it predicts:

```text
expected predictive utility mu
utility uncertainty sigma
beneficial-support logit
```

Selection uses `safe_score = mu - beta * sigma`, where `beta` is selected only on the held-out Train-dev split.  Training combines heteroscedastic utility regression, best-vs-worst pairwise ranking, clear beneficial/harmful classification, and a dedicated **high-similarity-but-harmful** ranking loss.

Predictive Need, Dual Memory, query-support alignment, and the revised Qwen reasoner are deliberately not added to this probe.  v0.9A answers one controlled question: **can better node-horizon predictive-utility learning reduce harmful support selection on the exact same candidate pool?**

## Recommended Gadi run

```bash
cd /scratch/mh20/zw1165/Speed_LLM_RAG/PreSuRe
conda activate TFP

python tests/smoke_predictive_utility_retrieval.py
python scripts/predictive_utility_reranker_preflight.py --stage development
bash scripts/submit_predictive_utility_reranker_probe.sh
```

The main outputs are `predictive_utility_reranker_best.pt`, `summary.json`, `comparison.csv`, and `DECISION.txt` under `PreSuRe_exps/PredictiveUtilityRetrieval/...`.

The configured GO targets are Masked pairwise rank >= 0.67, selected beneficial >= 0.60, selected harmful <= 0.35, and selection-regret reduction >= 20% against coarse similarity.  See `MIGRATION_v0.8.0_to_v0.9.0.md`.

---

# PreSuRe v0.8.0 — Need-Triggered Qwen Support-Value Reasoning Probe

v0.8.0 is the first LLM reasoning release. The v0.7.1--v0.7.6 diagnostic chain showed that severe forecast need is predictable and historical support opportunity exists, but deployment-safe numerical models cannot reliably identify whether a retrieved support is useful or harmful. v0.8.0 therefore introduces Qwen **only as a support-value reasoner**. It does not forecast traffic speed and does not predict the future residual.

The deployment decision is structured as `USE(P0..P3) / REJECT / ABSTAIN`. The reasoner is triggered on predicted-Need Top20 cells with disagreement among the frozen Generic, Hard-q90, Pred10, and Pred20 proposal experts. Train future creates supervision labels only; Validation future is reporting only.

Configured local models:

```text
Qwen2.5-0.5B-Instruct  /scratch/mh20/zw1165/cache/huggingface/models--Qwen--Qwen2.5-0.5B-Instruct
Qwen2.5-1.5B-Instruct  /scratch/mh20/zw1165/cache/huggingface/models--Qwen--Qwen2.5-1.5B-Instruct
```

No Internet download is used. The code resolves the local HuggingFace `snapshots/` directory automatically. Both zero-shot and LoRA-SFT probes are supported.

## Recommended Gadi run

```bash
cd /scratch/mh20/zw1165/Speed_LLM_RAG/PreSuRe
conda activate TFP
python tests/smoke_static.py
python tests/smoke_qwen_support_value_reasoning.py
python scripts/qwen_reasoning_preflight.py --stage development
bash scripts/submit_qwen_support_value_suite.sh
```

The suite runs dataset construction, zero-shot probes, LoRA SFT for 0.5B and 1.5B, post-SFT probes, and a combined summary. See `MIGRATION_v0.7.6_to_v0.8.0.md`.

---

# PreSuRe v0.7.5 — Deployment-Aligned Specialist + Expert-Preference Routing Probe

v0.7.5 is the current development diagnostic release. v0.7.4 established an important asymmetry: the Hard-q90 specialist improves utility discrimination when the true hard regime is known, but direct routing by predicted Need-q90 does not preserve that advantage. v0.7.5 therefore tests whether the remaining bottleneck is **deployment-distribution alignment and expert preference**, rather than another Need threshold.

The probe compares four deployable utility experts (`Generic`, `Hard-q90`, `Pred10`, `Pred20`) and an Expert-Preference Router. Pred10/Pred20 are trained on cells selected by Train-derived predicted-Need thresholds. The router is trained only on Train `utility_dev` to learn which expert's selected support has higher true utility; current-query future is never an inference feature. A privileged `oracle_router` is reported only as a ceiling.

## Run on Gadi

```bash
cd /scratch/mh20/zw1165/Speed_LLM_RAG/PreSuRe
conda activate TFP

python tests/smoke_static.py
python tests/smoke_deployment_aligned_utility_router_probe.py
python scripts/deployment_aligned_router_preflight.py --stage development
bash scripts/submit_deployment_aligned_utility_router_probe.sh
```

Primary outputs are `deployment_router_validation_summary.json`, `deployment_router_comparison.csv`, `train_only_predicted_need_thresholds.json`, `expert_preference_router_best.pt`, `summary.json`, and `DECISION.txt`.

The decisive deployment slice is **Masked / frozen Train-threshold PredTop10**. `true_q90` and `oracle_router` are privileged diagnostics only. See `MIGRATION_v0.7.4_to_v0.7.5.md`.

# PreSuRe v0.7.6 — Regret-Aware Support Proposal Meta-Ranking Probe

v0.7.6 is the current development diagnostic release. v0.7.5 showed that the expert pool has substantial privileged complementarity, but ordinary expert classification does not translate into better support selection. v0.7.6 therefore removes discrete expert classification as the primary objective.

For every node x horizon cell, the frozen `Generic`, `Hard-q90`, `Pred10`, and `Pred20` experts each submit one Top1 support proposal. Duplicates are merged. A new proposal-level meta-ranker estimates which proposal is actually useful using only deployment-safe query/support context, predicted Need-q90, expert score/confidence summaries, M3 confidence, historical residual summaries, and expert agreement.

The loss is explicitly utility aligned: `pairwise ranking + utility regression + beneficial/harmful classification + expected selection regret`. Unlike v0.7.5, ambiguous expert cases are retained instead of discarded. Small utility differences naturally create small regret, whereas selecting a harmful proposal when a beneficial one exists creates a large penalty.

`direct_mask` is allowed only as Train weighting and evaluation view; it is not a model feature. Current-query future is used only for Train proposal-utility supervision and final Validation reporting.

## Run on Gadi

```bash
cd /scratch/mh20/zw1165/Speed_LLM_RAG/PreSuRe
conda activate TFP

python tests/smoke_static.py
python tests/smoke_regret_aware_support_proposal_meta_probe.py
python scripts/proposal_meta_ranker_preflight.py --stage development
bash scripts/submit_regret_aware_support_proposal_meta_probe.sh
```

Primary outputs are `proposal_meta_validation_summary.json`, `proposal_meta_comparison.csv`, `proposal_meta_ranker_best.pt`, `summary.json`, and `DECISION.txt`. The decisive deployment slice is **Masked / PredTop10**. See `MIGRATION_v0.7.5_to_v0.7.6.md`.

## v0.9A.2 hotfix1 (2026-09-17)

Fixes the Train-dev policy-calibration / Final Validation rank-order path after job `179261672`: the frozen v0.9A `rscore = mu - beta*sigma` is now explicitly retained from `model_outputs()` and all policy ranking uses the shared `frozen_rank_order()` helper.  No model, target, candidate pool, split, or Rank-then-Veto policy design is changed.

## v0.9.5.1 — PatchSTG Hidden-Layer Sweep + Causal Error-State Repair

v0.9.5 showed only a small query-error direction improvement, but two intended hypotheses were not actually resolved: automatic hidden extraction selected `input_st_fc`, and recent-error evidence was disabled because row-order chronology could not be proven. v0.9.5.1 is a representation diagnostic release that fixes those two experimental gaps without changing the retrieval candidate pool, Qwen, or the protected PatchSTG forecast.

### What changes

1. **All-module PatchSTG layer catalogue.** Hidden discovery now hooks both leaf and parent modules. Every node-aligned activation is logged with module name, call ordinal, raw shape, execution index, module depth, leaf/non-leaf status, raw width, and deterministic sweep score. The sweep explicitly includes the old `input_st_fc` reference when present and compares it against deeper candidate blocks.
2. **Two-stage layer selection.** Up to 10 candidate layers are screened with seed 10 on the exact v0.9A Train/Train-dev split. The best two are then confirmed with seeds 10/20/30. Only Train-dev selects the hidden layer.
3. **Causal recent-error repair.** Chronology now uses the SupportBench `window_start` manifest coordinate when available. Train features use earlier Train forecasts only. Validation recent-error features use Train forecast outcomes only; Validation labels are never fed back as features. One deterministic historical view per anchor is used to avoid averaging counterfactual sensing views.
4. **Representation confirmation.** After selecting the best hidden layer, `base`, `physical`, `recent_repaired`, `physical_recent`, `best_hidden`, `best_hidden_physical`, and `full_repaired` are compared over seeds 10/20/30. Mean Train-dev objective selects the representation; the median-by-dev confirmation seed is frozen before Final Validation.
5. **Final Validation remains evaluation-only.** Hidden layer, representation family, and representative seed are all fixed before the selected representation is evaluated on the v0.9A Final Validation IDs.

### Run

```bash
cd /scratch/mh20/zw1165/Speed_LLM_RAG/PreSuRe
conda activate TFP
python tests/smoke_query_error_state_representation_diagnostics.py
python scripts/prepare_query_error_state_representation_diagnostics.py --stage development
python scripts/query_error_state_representation_diagnostics_preflight.py --stage development
bash scripts/submit_query_error_state_representation_diagnostics.sh
```

The PBS job is `QErrDiag`.

### Main outputs

`PreSuRe_exps/QueryErrorStateRepresentationDiagnostics/TraffiDent_Net_06/development/<stamp>_seed10/`

- `summary.json`
- `DECISION.txt`
- `layer_sweep.csv`
- `representation_comparison.csv`
- per-layer / per-seed checkpoints

The layer catalogue and repaired causal artifacts live under the existing PreSuRe artifact root:

- `patchstg_hidden_layer_catalog.json`
- `query_error_state_diagnostics_manifest.json`
- `train_recent_error_state_repaired.npy`
- `val_recent_error_state_repaired.npy`

The scientific question is now explicit: **does a deeper PatchSTG predictive state or a truly causal backbone-bias history materially improve estimation of future correction demand beyond the v0.9.4/v0.9.5 query representation?**

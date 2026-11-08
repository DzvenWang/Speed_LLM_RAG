# TCRR-STLA v9.2.0 — model and causal-scope contract

## Research claim and scientific limitation

Historical counterfactual failure response D_k(M) is the change in the forecast
of the SAME frozen predictive model, on the SAME historical traffic state, when
its *observation availability* is altered. It is NOT an identified causal effect
on real traffic dynamics. It is not an ordinary historical prediction residual.

Historical original-state residual R_k^N = Y_k - F(X_k, M_source).
Historical matched-mask response D_qk = F(X_k, M_source) - F(X_k, M_source ∧ M_q).
Their sum is the historical failure residual R_k^F. Thus explicit CF must beat
both Normal-only and a direct historical failure residual baseline to justify
its greater complexity. No accuracy improvement is guaranteed before experiments.

## Data contract

The model.forward reads ONLY: `base [B,H,N]`, `context [B,N,C]`,
`x,mask [B,L,N]`, `normal,cf [B,K,H,N]`, `score,valid [B,K]`,
`match [B,K,N,2]`. No current-query future, full input or counterfactual
teacher is accepted. TRAIN labels and full observation enter separate Loss /
Teacher and replay preparation functions. Input sequences are 5-min traffic
and normally L=H=12. N is always inferred dynamically; never hardcoded 388.

The strong v5 source retrieval (12 candidates) is preserved for Official
Forecast Evidence, subject to four fixed legal time-only anchors. Paired View
retrieves its eight remaining candidates independently using ONLY visible
traffic measurements and training-original-valid historical signatures.
Both views contain the same four time-legal anchors for matched-ID supervision.
No paired view's retrieval scores are copied to the other. TRAIN-only history
future must mature before query origin. A source-valid mask is always applied
before historical replay. Alternative fault masks are source-valid.

## Module graph

1. Frozen STLAformer and mask adapter produce `[B,H,N]` base and `[B,N,C]` context, and are precomputed offline. No teacher gradients in CF training.
2. v9.1 Normal Anchor is retained *exactly*, without unused legacy CF heads: train first, then freeze every Normal parameter in all CF stages. `nraw [B,H,N]`, `normal_prediction [B,H,N]`.
3. TemporalCFEncoder: original Normal + query-mask historical CF `[B,K,H,N]` → order-sensitive block stats and Conv1d → `[B,K,G,N,16]`.
4. MultiRelationCFGraph: sparse Functional/Upstream/Downstream messages on temporal CF embeddings. One learned linear transform per relation; separate condition gates use current masked-query representation, source observability and graph existence. `graph_cf [B,K,G,N,16]`, `gates [B,N,3]`. Physical graph channels zero when no grounded topology supplied. Graph branch emits zero for nodes with no actual neighbours.
5. CF Fusion joins query `[32]`, candidate `[32]`, temporal `[16]`, graph `[16]`, graph masses `[3]`, relation gates `[3]`; outputs `[B,K,G,N,32]`.
6. Signed Alignment: `alpha=1.5*tanh(Linear(z))`, initially +0.1. Supports positive, negative, near-zero and each horizon block separately. `candidate_cf=alpha*historical_response`; exact zero-CF invariance.
7. Incremental Single Utility and Auxiliary Mixture Marginal Utility, each `[B,K,G,N]`. Utility also reads the *aligned response*'s local statistics, not just feature z.
8. Conditional Null Logit `[B,G,N]`, plus a trainable full-observation recoverability response with teacher-based residual-gain supervision; all weights are normalized across K candidates plus Null per node×block. Candidate selection never reads current future.
9. `prediction=base+bound(nraw+sum(w*candidate_cf))`, with `bound(x)=3*tanh(x/3)`. The actual CF increment is `prediction-normal_prediction`, not the unbounded mixture.

## TRAIN-only supervisory signals

- Forecast loss computed with future validity, including already-corrected Normal anchor.
- Candidate individual utility = MAE(normal)−MAE(normal+one candidate), node×block.
- Ranking chooses the maximum *positive* candidate utility or Null.
- Discrete signed-alignment coefficient teacher from candidate-local oracle grid on TRAIN future, filtered when responses are indistinguishable.
- Leave-one-out marginal utility teacher recomputes the full bounded prediction after removing and renormalizing candidate k (including Null); only used after joint stage begins.
- Full-observation recoverability teacher estimates possible benefit versus the frozen Normal baseline. It modulates the Null logit; it cannot enter inference as a full input.
- Paired Same-ID utility difference plus Matched-vs-Mismatched historical CF contrast, using unchanged current Query/Normal and TRAIN outcomes. Correct-mask samples are NOT automatically assumed better; signed label comes from observed training improvement.

## Five stages

- Normal: only query/match/Normal Fusion/Normal Scale/Normal Null train.
- Alignment: local temporal, learned relation GNN, Fusion, signed alignment and conditional Null train. Uniform candidates but trainable Null enable reject-all.
- Utility: only utility heads, conditional Null, full-info head and recoverability_scale update. Alignment/Normal frozen.
- CF Joint: all CF parameters update together from final forecast and utility/marginal/teacher losses; Normal remains frozen.
- Paired: all CF parameters update with forecast and two pair-difference objectives, alternating which query mask supplies detached reference. Matched-ID overlap is measured and absence raises an error.

Epoch maxima default 12/20/20/12/6; disjoint chronological validation **selection** chooses best stage checkpoints. A selected Normal checkpoint explicitly means CF did not outperform Normal on Selection. Always compare Normal and Selected on Diagnostic Check and final blind Test.

## Mechanism auditing contract

`diagnostics/report.html`, `summary.json`, `ablation_results.csv`,
`query_level.csv`, `horizon_analysis.json`, `failure_groups.csv`,
`gradient_audit.json`, `recommendations.json` are generated after training.

- All checkpoint toggles on exactly the same diagnostic cohort; never compare a 32-query Wrong-mask value to a 256-query Correct value.
- Independent retraining effects are **not** inferable from one checkpoint. Use `scripts/compare_ablation_v92.py` on separately trained variant runs with identical data, Normal weights and diagnostic cohort.
- Wrong-mask replay re-executes frozen STLA under a node-rolled query mask for the same subset/candidate IDs; it is a synthetic *negative control*, not necessarily a physically plausible failure.
- Retrospective CF headroom uses VALIDATION labels to estimate an optimistic local oracle ceiling. It is NEVER deployed or used to choose test-time corrections.
- Bootstrap resamples time blocks; tiny cohorts show inconclusive rather than spurious confidence.
- No Physical Graph means Functional-only; all other graph relation contributions are not applicable.
- Candidate vs query forecast metrics are intentionally distinguished; a utility reduction does not by itself prove improved blended forecast.

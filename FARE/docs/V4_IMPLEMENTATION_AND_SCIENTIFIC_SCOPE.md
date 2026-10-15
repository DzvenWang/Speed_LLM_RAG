# FARE v4.0: implementation and scientific scope

## Main computation graph

`frozen_stla.encode_base` → `PropagationSupportEncoder` → `HierarchicalRetriever` → `PropagationTransport` → `SetConditionalSelector` → `ForecastIntegrator` → `RiskController`.

- **Propagation Support Demand:** direction-labeled sparse path coverage, temporal kernel (lag estimate), horizon and boundary view, missing component proxy, learned soft regime. Actual physical road distance/shockwave speed **not present**. Train-only legal observation-removal supervision targets backbone sensitivity rather than physical observability.
- **Demand–Capability Retrieval:** candidate historical state key (past only), historical completed-Future/residual capability, region multi-route recall, masked node rerank, and strict historical future-end exclusion. Discrete top-k ranks supported by prediction gradients on selected scores and separate teacher value labels; no unjustified claim of differentiable top-k.
- **Propagation Transport:** directional bounded paths, learned space weights and temporal 0..3 lag, target-node self residual/evolution retained, independent VALUE maps, compatibility and uncertainty proxies.
- **Set Counterfactual Utility:** 2-layer candidate Set Attention; pairwise Keep/Swap scorer over initial top-8 × unselected pool; one valid budget-constrained swap; per-evidence Horizon-specific weights and Null. Train-only exact final-forecast LOO and same-weight replacement labels on a frozen teacher snapshot.
- **Integration and Risk:** frozen STLA prefix reused; evidence-aware suffix with trainable attention; direct forecast correction; Null/Candidate/Final explicitly separate; action teacher scales unscaled Delta, and gain/harm heads get labels.

## Experimental claims that require evidence

1. Propagation support deficits must predict actual failures and influence which historical candidates are valuable *beyond mask coverage alone*.
2. Demand–Capability matching must beat state-only and random-legal retrieval, especially across regional/directional outages and severe sensing failures.
3. Dual evidence must show independent, possibly interacting value under information-level mask interventions.
4. Keep/Swap utility ranking and sign predictions must correlate with true counterfactual gains.
5. Final Risk Controller must reduce harmful corrections relative to uncalibrated alpha=1.
6. Results need original-unit Speed/Flow MAE/RMSE at s20/40/60/90, multiple seeds for final claims, canonical Base_SupportBench metrics, and valid producer provenance.

All interventions are *fixed-checkpoint* perturbations, not separately trained architecture ablations. Internal Test interventions are diagnostic-only and must not be used for hyperparameter selection.

# PCFT-STLA v3.0 — TRC research implementation

**Primary aim:** retrieve usable historical future evolution under incomplete sensing, transfer *time-consistent* traffic propagation from legal Train episodes and reliably choose when propagation actually helps.

## Three contributions

**C1 Support-Anchored Historical Future.** The base legacy future-delta path remains as a strong reference. `v3/future.py` adds node-specific observed coverage, last-observed age, neighbor coverage and recency, and **learns an evidence-only amplitude** initialized to one. No current real future enters this module. M1 adds this to M0; use group-by-missing-age and directly masked MAE to test improvement.

**C2 Provenance-Coherent Lagged Propagation.** The v2 regional candidate pool is a **per-node composite of different historical windows**. The new `v3/coherent.py` propagates only **full historical episodes**: Global episodes (same memory ID across nodes) and separate full Train episodes selected by each region. Target-region outputs are then assembled *after* within-episode source→target propagation. Lagged influence uses train-only statistical graph (`propagation_v2.py`), not claimed causal road physics. `no_directional_graph` and `no_lag_alignment` now alter graph scores and lag calculation rather than bypassing the Action Controller. M2 adds this to M1 while preserving the **same learned v2 Action Controller**, so its conditional gain cannot simply be attributed to swapping controllers.

**C3 Utility-Calibrated Evolution Action Policy.** `v3/action.py` uses predicted action gain, harm and uncertainty to score Legacy, Propagation and None. `v3/losses.py` directly trains action-level observable utility and branch-specific propagation quality; the utility head now participates in the real action. M3 replaces only the action decision in M2. A bounded correction uses only legal retrieved Train histories/futures. None remains available; propagation is **never forced** at inference.

## Independent contribution matrix

| Variant | C1 anchored future | C2 episode-coherent propagation | C3 gain–harm action |
| --- | --- | --- | --- |
| M0 | no | no | common v2 Action Controller with Propagation unavailable |
| M1 | yes | no | **same** common controller |
| M2 | yes | yes | **same** common controller |
| M3 | yes | yes | gain–harm action controller |

All four have the **same Global/Regional/Propagation candidate budgets**; disabling C2 does not silently remove the extra memory candidate pool. M0/M1 still use the identical learned Action Controller, with Propagation disabled, to preserve attribution. Independently train each variant; fix same STLAformer initialization hash, seed, Train-only repository, region graph, index refresh interval, candidate budgets, batch/epochs and loss protocol. Note that differing supervision terms are part of the named algorithm and must be separately reported.

## Training supervision and source isolation

- Stage-specific optimization is encoded by contribution configurations rather than separate partial source forks. The Backbone base is frozen for the main attribution, although learnable MaskSupport adapter is shared across variants.
- M2/M3 train a **separate propagation branch loss** so the branch receives gradients even if the final controller assigns near-zero propagation weight. This does not guarantee improvement; check branch gain on Validation and Test.
- M3 compares actual Train Query label with each executable Legacy/Propagation action, learns gain and harm, then jointly optimizes actual Forecast MAE/RMSE and harm tail.
- The source episodes use `repo.fetch(...)`, `repo.fetch_future(...)` and region indices from the **Train-only repository**. Every propagation path stays inside a single indexed history episode; the target-region result is selected only after computing intra-episode propagation.
- Observed support and inferred forecast support remain separate. `observed_support_only` recomputes uncertainty consistently; expected Future values come solely from Backbone predictions.
- Wrong-evidence audit modes shuffle/cross-replace episode futures too, not only Legacy Future correction.
- `no_risk` properly suppresses risk-based action penalties and Risk Prior; `no_action_utility` no longer replaces the learned policy with a fixed policy. `fixed_action_policy` is a separate test.

## Caveats

The current statistical graph may be useful for forecasting but does **not** prove physical or causal traffic propagation. Stage C cannot be counted as a validated TRC novelty until independent M3 > M2 and the mechanism metrics show improved action utility. An inferred-support ablation within a Full checkpoint is not a separately trained model. User has already inspected Net-06 Test extensively; hyperparameter selection should be based only on Train/Val and an independent confirmation network.

Unit tests show synthetic forward/backward and mechanism isolation only. Real Gadi V100 memory, runtime, and Speed/Flow gains remain to be demonstrated. PCFT v3 uses more full historical episode tensors than v2; default batch 64 is **not yet proven to fit V100 32 GB**. If OOM occurs, use microbatch gradient accumulation with the same effective batch before interpreting any MAE change.

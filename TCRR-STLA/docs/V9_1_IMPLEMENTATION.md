# TCRR-STLA v9.1.0: implementation, invariants and resource protocol

**Status:** Source implementation and CPU integration tests completed; not yet run on Gadi V100 with the user's actual external STLAformer checkpoint, dataset cache or roadway graph. Predictions/accuracy gains are not claimed.

## What changed from v9.0.0

1. **Local CF evidence reaches the utility head.** `TCRRV9._cf_block_features` preserves target-node CF signed mean, absolute mean, variance, within-block trend, normal/CF interaction and temporal convolution features. The old scalar H1–H12 mean is not used to predict CF utility. Both amplitude and utility consume `cf_fusion(z)` per horizon block.
2. **Missing future labels never enter a utility teacher.** `utility_targets(..., future_valid=...)` constructs validity-aware `[B,K,G,N]` targets. Both regression and node/block candidate-vs-Null ranking are masked; all-missing blocks are excluded. NaN in invalid future labels is tested.
3. **Fixed Normal anchor across training stages.** `query`, `match`, `normal_fusion`, `normal_scale`, `none_normal` are trained in the Normal stage and then frozen. CF Transfer learns `cf_local`, `temporal`, `cf_fusion`, `amplitude` with uniform CF weights. Utility and Cross stages train only `utility`, `fullinfo`, and the exact CF Null parameter `none`. We explicitly protect `none_normal` from unintended updates.
4. **Paired historical candidate IDs.** View 1 reuses the same time-legal historical candidate IDs, scores and validity as official View 0 for each TRAIN Query; the **only** intervention between those paired records is their mask and resulting frozen Teacher/replay. This is a training-only fixed-candidate intervention; independent-mask retrieval is not claimed. Cross-Failure always aligns by candidate ID and reports matched candidate ratio. Reference/optimized views alternate between epochs.
5. **Failure manifests.** Per view, store `family.npy`, `severity.npy`, `mask_hash.npy`. `manifest.json` records actual family counts, source compatibility and mask generation SHA. Physical region/directional faults require explicitly supplied road-grounded ordered region labels; absent labels lead to logged Sensor/Temporal fallback.
6. **Graph-aware local CF.** Three frozen graph sources (signed Functional; optional Upstream/Downstream) contribute block-local historical CF response, multiplied by *visible Query neighbor mass*. No usable neighbor => zero graph message; the local CF branch remains available. Functional-only is explicitly distinguished from physical graph mode.
7. **Teacher-to-student gain supervision.** Full-information gain **relative to the fixed Normal forecast** trains `fullinfo` to affect CF vs Null. A small optional positive-gain-only teacher alignment term (default 0.005) aligns train-time CF incremental forecast with Full-to-Masked teacher response only where Full Teacher improves the already-corrected TRAIN Normal anchor. No full TRAIN Query appears in `model.forward`.
8. **Diagnostics compare identical cohorts.** When Wrong-mask is requested, **every** other variant is evaluated on the same subset of Validation-Selection Query rows. `no_cf_graph` vs `no_all_graph` are separate; neither is mislabeled as the other.
9. **Independent versions:** v9.1.0 cache directory `v9_1_isolated` and experiment version `v9_1`; old v9.0.0 `last.pt` is explicitly incompatible. Existing v5 source and root legacy `VERSION` are unchanged.

## Equations

Historical normal residual `R_k^N=Y_k-B_k^N`, masked historical response `D_{qk}^{CF}=B_k^N-B_{qk}^{F}` where the **same frozen Teacher** supplies both forecasts. No current-query future is used to construct inferential evidence.

Normal anchor: `m_N = sum_k w^N_{qk} a^N_{qk} R_k^N`; `Yhat_N = B_q^F + f(m_N)`.

Per-block/target-node CF correction: `C_{qk,g,n}^{CF}=a_{qk,g,n}D_{qk,g,n}^{CF}` (the scalar amplitude is repeated for each horizon within block); `m_CF = sum_k w^CF C^CF`, with Null allowed.

`f(x)=bound*tanh(x/bound)`; `Yhat=B_q^F+f(m_N+m_CF)` and `Delta_CF=Yhat-Yhat_N` **exactly**.

Training-only candidate/block/node marginal utility:

`U_{qk,g,n} = mean_{h in g and future_valid}( |Yhat_N(h,n)-Y(h,n)| - |B_q^F(h,n)+f(m_N(h,n)+C_{qk}^{CF}(h,n))-Y(h,n)| )`.

Null target is used when all candidate improvements are non-positive. Utility predicts the signed score, and candidate-vs-Null ranking operates at node/block level, rather than collapsing all nodes into one global label.

For two masks of the same TRAIN Query, `L_cross = Huber((Uhat_a-Uhat_b), (U_a-U_b))` over the identical, legal historical candidate IDs. No candidate ID => no target; an epoch with **zero** valid matched candidates now fails, rather than silently declaring progress.

## Main files

- `tcrr_stla/v9/model.py`: Normal anchor, temporal CF-local features, graph reliability fusion, CF amplitude and utility, masked Utility Teacher, cross-failure loss.
- `tcrr_stla/v9/graph.py`: train-only signed Functional and optional directed Physical graph; sparse historical CF aggregation.
- `tcrr_stla/v9/views.py`: deterministic paired artificial failure and source-valid enforcement.
- `tcrr_stla/v9/prepare.py`: frozen Teacher, Normal/CF caches, official-fixed paired shortlist, view metadata, chunked resume.
- `tcrr_stla/v9/training.py`: stage-specific parameter freezing, GradScaler, checkpoints, alternating cross references, matched-rate monitoring.
- `scripts/train_v9.py`, `scripts/evaluate_v9.py`, `scripts/diagnose_v9.py`, `pbs/formal_v9.pbs`: launch and read-only validation/test.
- `tests/v9/test_v91_mechanisms.py`: sensitivity, future-invalid label invariance, exactly frozen Normal stage and paired-candidate identity.

## Compute and memory

- One V100 32 GiB and 12 CPUs, start Batch 32 and 4 DataLoader workers; CPU/Gadi-specific optimal size unverified.
- Teacher and historical replay precomputed under `torch.inference_mode()`; no Student backpropagation through STLA.
- Two deterministic TRAIN views, each FP32 replay `[50k,12,12,388]` ~10.4 GiB, streamed from `numpy.memmap`.
- CF neural feature integration occurs at `[B,K,3,N,d]`, not `[B,K,12,N,d]` for all hidden layers. Raw `[B,K,H,N]` response is retained to reconstruct the final forecast.
- Sequential reference/optimized view in Cross stage avoids retaining two full autograd graphs. It alternates orientation across epochs. Cross by default runs **4 epochs**, but is an ablatable stage via `CROSS_EPOCHS=0`.
- Replay cache identity includes source hashes, Teacher/Adapter hashes, graph/region references, plus v9 source fingerprints. Progress is atomic at chunk boundaries.

## Validity and research limitations

- Synthetic test pass **does not establish** improved MAE or TRC-level innovation; compare on real Speed and Flow with v5.3 and v9.0.0 under the same protocol.
- Historical NormalBank residuals from training windows are not necessarily out-of-fold. Formal methodological claims must consider OOF/chronological residual protocol.
- Same-candidate paired mask learning does **not** establish an independently queried retrieval policy for each alternative mask. It isolates the *failure intervention* from retrieval identity; a separate independent-retrieval control is necessary to study coupled policy change.
- `fullinfo` recovers a signed information-gain statistic, **not** a guarantee that a full-observation forecast is always better. Teacher alignment is applied only to positive realized TRAIN improvements and can be disabled for an independent ablation.
- Physical graph is used only if a correctly ordered and validated `.npz` and `node_ids.npy` are provided. Directional faults from approximate region labels must not be described as true road propagation.
- Null candidate and valid weighting are not a calibrated coverage guarantee. v9.1 does not claim guaranteed risk control; check CF harm and choose a later independent calibration policy if needed.

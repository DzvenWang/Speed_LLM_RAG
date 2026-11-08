# TCRR-STLA v9.0.0 — implementation architecture

## Main research hypothesis

The value of a historical counterfactual response depends on the sensing failure and the available spatiotemporal traffic evidence. A frozen backbone is the common forecasting reference; historical `Normal` residuals provide a fixed correction anchor, and historical `CF` responses provide **measurable additional** corrections.

## Definitions

For a TRAIN historical sample k with original source validity S_k:

- `B_k^N = F(X_k^{full} S_k, S_k)`.
- `R_k^N = Y_k - B_k^N`.
- For a query q mask M_q, `B_{qk}^F = F(X_k^{full}(S_k AND M_q), S_k AND M_q)`.
- `D_{qk}^{CF} = B_k^N - B_{qk}^F = R_{qk}^F - R_k^N`.

The full TRAIN query `X_full` may additionally be observed by an **offline Teacher**, never by v9's inference model. For two masks of the same TRAIN query, the teacher produces different failure responses and different full-information gains.

The v9 student uses:

`B_q^F = frozen_STLA(X_q^obs,M_q)`

`m_N = sum_k w_N(q,k) * a_N(q,k) * R_k^N`

`C_CF(q,k) = a_CF(q,k,Graph(q),Graph(D_CF)) * D_CF(q,k)`

`m_CF = sum_k w_CF(q,k) * C_CF(q,k)`

`f(z)=bound*tanh(z/bound)`

`Yhat_N = B_q^F + f(m_N)`

`Yhat = B_q^F + f(m_N + m_CF)`

`Delta_CF = Yhat - Yhat_N` exactly.

No CF candidate or no valid history produces exactly zero CF increment. Utility teacher:

`U_CF(q,k) = MAE(Yhat_N,Y_q) - MAE(B_q^F+f(m_N+C_CF(q,k)),Y_q)`.

For utility training, the labels `Y_q` are used **only** to calculate a detached train-time target. Inference does not construct utility labels. A learned signed utility and a trained Null Candidate decide whether a historical CF response is applied. Full-observation benefit is separately predicted by a query-only recoverability head using only incomplete Query features at inference.

### Graph

- Functional kNN is constructed strictly from **TRAIN only** normalized speed/flow changes; preserve correlation sign in historical CF spatial evidence.
- Upstream/downstream *physical* graphs are optional but must have node order and a confirmed edge orientation. If no physical graph is supplied, physical graph channels are zero; the system must not claim that the Functional graph is a physical road graph.
- Three graph spatial summaries are computed once per query or per retrieved candidate; historical CF propagation uses `torch.sparse.mm`, avoiding dense quadratic graph multiplication for every candidate/horizon.
- `GraphBundle.query_summary` uses only *masked observed* input, with explicit availability mass and degree/neighbor presence.

### Data/view/caching

1. `train_x_full_norm.npy` exists only in TRAIN sample API with `cf_training=True` and offline Teacher preparation. `V9EvidenceDataset` never exposes it.
2. `view 0`: original official incomplete input and existing v5.3 retrieval shortlist (filtered for strict chronology).
3. `view 1`: deterministic multi-failure Mask on the *same TRAIN Query*, with independent cached STLA prediction and CF Replay. Its shortlist is time-only to avoid using data hidden by the paired mask.
4. NormalBank stored once; view-specific `B_q^F`, Context, historical CF, Candidate IDs/Scores, Match and Mask cached as memmaps. Cache contains a TRAIN-only `teacher_cf` target not passed to model.forward.
5. Candidate historical future must mature before Query forecast starts, duplicate candidates are disabled, and candidate future labels must be complete. In a Query without history, Normal+CF correction is exactly zero.
6. Memory cache metadata includes teacher SHA, source dataset file hashes, manifest metadata, shortlist SHA and each view ID. The data-prep stage supports atomic chunk progress and restart.

### Training phases

- Normal: frozen STLA, train Normal Anchor.
- Transfer: Uniform historical CF weights + bounded CF amplitude, no untrained Ranker.
- Utility: train the signed historical CF Utility + Null Candidate with forecast MAE, utility regression, ranking and query-side full-information benefit (requires TRAIN Teacher).
- Cross (`cross_epochs>0`, **off by default**): compare candidate utility for the same candidate ID under two failures; only matched IDs receive cross-failure supervision. Uses no-grad official branch and a trainable paired branch to limit GPU memory.

### Deliberately omitted from v9

The v5 CRSI Controller, CRC, SLI, multi-stage marginal ranking and four overlapping CF intensity gates. They remain in the bundled legacy code for comparison but are not used by v9 training. V9 is a new model, **not** a continuation of v5 checkpoints; only the frozen base/adapter and dataset protocol are reused.

### Evaluation and interpretability

Evaluation reports raw-target Overall MAE/RMSE, direct-mask MAE, H12 MAE, Normal-vs-Full CF gain, CF harm rate and active rate for each severity. `teacher_cf_training_only` is **not** passed to inference. Cross-failure utility is not computed on TEST.

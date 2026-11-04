# Implementation review — v7 two-module TCRR

## 1. Standalone source compatibility and deployment

New TCRR weights are trained from scratch in `v7`. The **official STLAformer** family remains frozen, with its existing best.pt and original Cache shapes preserved. Both Speed and Flow are independent processes and have no cross-target inputs, labels, or runtime weights. There is **no checkpoint upgrade/migration facility**. Tensor dimensions are obtained from the supplied original STLAformer checkpoint and the graph metadata, never from a hard-coded Net_06 count.

## 2. C1 support matching

Let `x,m` be query history and observed mask `[B,H,N]`. Let historical TRAIN source `x_k,s_k` be `[B,K,H,N]` with original source-valid mask `s_k`. `PairSupport` contains query/source coverage, overlap, intervention exposure, recent overlap/intervention, observed agreement (masked source/query), has-overlap flag, missing age, longest run, local neighbor coverage and recent query coverage. Shape: `[B,K,N,14]`.

`O_k = m*s_k` defines comparable history coordinates. `I_k=(1-m)*s_k` defines the locations that the TRAIN source has but the query does not. Intervention exposure is **not itself proof that a support is usable**; the learned compatibility and Utility decide. For no overlap, observed agreement and has-overlap are zero. All-missing windows are supported.

The raw support tensor is frozen **data**, stored FP16 in replay; the learnable MLP embedding remains trainable. Candidate Normal/CF responses are retained until the horizon-/node-wise Utility/None mix, and are never averaged before pairing. Wrong CF response pairs are valid TRAIN negative examples only.

## 3. Multi-resolution retrieval

Index only source-valid TRAIN histories at periods `[4,2,1]`. For each query, retrieve a shortlist from each keyspace, exclude any TRAIN windows with historical/future interval overlap, then rank the union using (i) scaled history-signature score, (ii) overlap-restricted observed value agreement and (iii) normalized shape alignment of the query **Backbone prediction** to each candidate's **historical TRAIN future**. Forecast re-ranking weight decreases at low current query history coverage. Stable score sorting deduplicates IDs; final `K=12` preserves candidate-level model size.

No query future enters this retrieval function. Historical candidates and replay are regenerated under a different query Mask (no accidental reuse of stale counterfactual predictions). Each replay chunk validates SHA256 hashes and exact shape. TRAIN source histories are gathered using `np.load(..., mmap_mode='r')` and GPU query batches; chunk identity includes Backbone, graph, source Cache, code and retrieval settings.

## 4. C2 spatial transfer

The TRAIN-only functional graph uses pairwise valid moments and minimum common observations rather than filling invalid history with zeros. Road graph and directions use shared metadata and edge validation. Source reliability is estimated from C1's None mass, candidate agreement, CF reliability, compatibility, correction variance and candidate Utility.

For every horizon h, node i and valid source neighbor j, attention is:

```
score(i,j,h) = <Wq z_i, Wk z_j>/sqrt(d)
             + <E_h, W_edge e_ij>
             + R_j,h
alpha = masked_softmax_j(score)
m_i,h = sum_j alpha(i,j,h) z_j
```

Functional, Upstream and Downstream messages are separately computed. All-invalid neighbor sets return zero messages, with no NaN. The correction Head and Gate operate separately on node context, three aggregated messages, temporal observability and C1 reliability. The output is `P12=P1+g*bound(delta)`; its last correction layer is zero initialized, so Stage C2 begins at the previous C1 forecast.

## 5. Staged losses and checkpoint policy

- **S0:** original frozen STLAformer + trainable Mask Projection.
- **Normal:** learned Normal gain baseline.
- **C1:** raw-unit MAE + `0.05*RMSE` + existing candidate/CF rank and utility supervision. C1 activation parameter is zero-initialized: initial C1 exactly equals Normal output.
- **C2:** raw-unit MAE + `0.05*RMSE` + optional auxiliary `response_weight` (default `0.03`). We do **not** assume the structural response is physically causal.
- **Joint:** C1 lower LR, C2 standard LR; updates both modules subject to Validation Selection and early stopping.
- **Lock:** best Selection checkpoint fixed before TEST; the Locked Check is not used to retune parameters.

All training and metric computations keep future labels in the Trainer, not in `model.forward_progressive`; candidate ranking is constrained to TRAIN futures. For any ablation, use a fresh run directory. The code refuses to resume with differing implementation SHA/configuration or source Checkpoint identities.

## 6. Known limitations

- `source_valid` is assembled from Benchmark's original-valid/artificial-missing metadata; it must be confirmed by read-only Preflight against each specific Net/Target before use.
- Functional graph is based on statistical correlation; upstream/downstream geometry is only a direction approximation, not calibrated wave propagation.
- Replay rebuilding can dominate runtime and disk writes. Chunk-based sampling reduces random mmap churn but does not establish a guaranteed speedup.
- C1 and C2 are stronger hypotheses than a normal residual Head, not proven performance improvements until real Gadi experiments complete.
- Net_06 Seed10 TEST s90 has previously been inspected in earlier work: do not use that subset to choose new variants. Use development Selection plus independent Net/Seed or withheld periods for scientific confirmation.

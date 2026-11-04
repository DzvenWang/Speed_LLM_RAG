# v7.1.0 C1 — implementation and safety notes

## Exact regression-preserving boundaries

Unchanged relative to supplied v7.0.1: `model.py`, `counterfactual.py`, `replay.py`, `support.py`, `graph.py`, `intervention.py`, `storage.py`, and both STLAformer/data adapters. Therefore the hashing-sensitive `graph_algorithm_sha256` (currently inappropriately bound to whole `model.py`) and Replay `implementation_sha256` remain byte-identical. Graph/replay redesign is deferred.

Changed: `transport.py` (only lightweight set statistics and prediction-temperature wiring), `losses.py` (PN-relative utility, None utility, separate target temperature, NaN-safe loss), `runner.py` (wiring, safe masked loss/eval, gradient accumulation, mode-depth evaluation, report-only C1 diagnosis, resume identity), `run_v7.py` (v7.1 identity, c1 default, fail-fast), `config`, PBS and tests. `model.py` is deliberately not touched for cache identity safety.

## Full-activation C1 utility targets

Define `P0=base_prediction`, `PN=normal`, candidate transmitted residual `r_k=transport.candidate_residuals`, and `bound=b`. Candidate hypothetical prediction is `P0 + b*tanh(r_k/b)`. With target `Y`, candidate utility is `|Y-PN|-|Y-(P0+b*tanh(r_k/b))|`. None hypothetical prediction is `P0`, so its utility is `|Y-PN|-|Y-P0|`. These targets are detached from the model and computed inside TRAIN auxiliary loss, not used in inference. This is **potential** utility under full C1 activation: it is not the final mixture's exact marginal contribution.

Ranking target temperature and prediction temperature are separate. The prediction temperature defaults to 1.0 to avoid inadvertently replacing distributed Flow support with Top-1 behavior. Marginal LOO remains the original P0-referenced initial-mixture contribution target; this isolates the effect of new candidate-set statistics.

## Candidate-set features

The existing five marginal features receive four additional dimensions computed from initial candidate weights excluding None: `(abs(safe-weighted_mean), sqrt(weighted_variance+1e-8), abs(weighted_sign_sum), tanh(safe*weighted_mean))`. None mass stays independently represented by the original marginal feature. The computation creates O(B*K*P*N) activations and no K×K matrix. If no candidate is valid the denominator is clamped and the None token receives all probability mass.

## Prequeue and PBS

`pbs/submit_v7.sh` runs pytest, C1 real-dimension CPU contract, original Backbone/data CPU preflight, and two `run_identity.json` roundtrips BEFORE `qsub`. It optionally calls `scripts/reuse_v7_caches.py` to import verified old caches. `pbs/formal_v7.pbs` repeats run identity validation inside the allocation and performs a CUDA + AMP two-step C1 contract before training. Checkpoint identity includes run-identity file SHA256 and original Backbone SHA256. Child stage subprocesses inherit `--end-stage c1` so parent and children use identical identity.

For interrupted stage restart use the same immutable source, full config, epoch counts, and run directory. Never edit queued source. Code digest includes the PBS scripts, C1 preflight, and cache importer. A stale source digest fails closed rather than silently mixing model versions.

## Testable acceptance criteria

1. Unit tests cover PN-relative/None utility, temperature activation, NaN invalid labels, empty valid candidates, candidate permutation, parameter dimensions, and stages.
2. C1 real-dimension forward/backward/optimizer pass (CPU prequeue; CUDA+AMP after allocation).
3. Synthetic fake STLAformer end-to-end pipeline for Speed and Flow finishes at C1 with `best_c1.pt`, `selection_c1.json`, and no C2/TEST artifacts.
4. Verified cache import checks every Replay chunk SHA256 before stage markers are created.
5. Real official STLAformer/data preflight is required on Gadi; GPU memory-fit and accuracy are **not claimed** until actual PBS runs finish.

## Future changes deliberately not included

No C2 spatial redesign, signed-utility-head replacement, retrieval encoder retraining, rehashing functional graph algorithm identity, or new trained set-attention transformer. These are distinct later experiments and must not be mixed with initial v7.1.0 C1 attribution.

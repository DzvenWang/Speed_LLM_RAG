# PreSuRe v0.3.0 — Local Predictive Support Routing

PreSuRe is the support-aware correction framework built on the fixed PatchSTG numerical backbone under incomplete sensing. This version is the first **local-routing** implementation, motivated directly by the v0.2 Development findings:

- query-global Oracle-A/B had only small headroom;
- horizon-local Oracle-C increased headroom materially;
- node×horizon Oracle-D showed that useful residual evidence is highly local;
- deployable M2/M3 correctly chose no-op under the old global residual-transfer design.

The v0.3 design therefore does **not** try to make global sample retrieval stronger. It changes the correction unit.

## Scientific ladder

### Non-deployable Oracle diagnostics

All Oracle diagnostics retrieve from Train memory only and use Validation future **only for diagnostic selection**.

- **Oracle-A**: Top-64, one support + one global alpha per query.
- **Oracle-B**: expanded Top-512, one support + one global alpha per query.
- **Oracle-C**: one support + alpha independently for each forecast horizon.
- **Oracle-E**: one support + alpha independently for each node, shared across all horizons.
- **Oracle-F**: one support + alpha independently for each **graph-region × horizon** role.
- **Oracle-D**: one support + alpha independently for every node×horizon cell; intentionally loose ceiling.

The key diagnostic chain is:

`A/B -> C -> {E,F} -> D`

Oracle-F is the most important architecture-matched upper bound for M4. The graph regions are built from the fixed SupportBench adjacency only; no Validation/Test target is used to create them.

### Deployable M0–M5 ladder

- **M0** — frozen PatchSTG.
- **M1** — sensing-aware query representation + query-only diagnostic correction head.
- **M2** — global similarity-based retrieval residual correction. This is intentionally retained as the negative-control retrieval stage.
- **M3** — **horizon-aware predictive utility routing**. Candidate utility is predicted separately for H1…H12. The utility model sees the query/candidate representation plus deployable horizon-local descriptors of the historical residual evidence. Each horizon has an explicit no-support path and a Train-only calibrated correction scale.
- **M4** — **graph-region × horizon predictive support routing and local residual transport**. Each candidate receives a separate utility for each graph-region×horizon role. Local residual, base-forecast, sensing coverage, direct-mask, and support-mask descriptors are used. Each role can reject all support. Correction scales are calibrated using Train only.
- **M5** — **node×horizon sensing-aware safe gate** over M4 local correction. It receives the query representation, node/horizon embeddings, local base/correction values, observation fraction, direct mask and support mask. Training includes overall no-harm and masked no-harm penalties.

Core form:

`Y_final = Y_PatchSTG + G(node,horizon,query) * Delta_local_support`

## Why graph regions instead of jumping directly to 388×12 routing

Oracle-D is intentionally too flexible to be a realistic deployable target. v0.3 inserts graph-region×horizon routing between horizon-only and node×horizon routing. For Net-06 the default is 24 graph regions (~16 nodes/region on average), obtained deterministically from the fixed traffic adjacency using spectral graph partitioning with a dependency-free NumPy k-means implementation.

This gives a learnable local routing problem while preserving an interpretable connection to traffic-network structure.

## Leakage controls

- Retrieval memory: Train only.
- Same view group excluded.
- History+future interval overlap excluded.
- M3/M4 utility labels: Train future only.
- M3 horizon scales: Train calibration only.
- M4 region×horizon scales: Train calibration only.
- Validation future: model selection / reporting and explicitly labelled Oracles only.
- Test is not instantiated in Development.

## Output contract

Experiments:

`/scratch/mh20/zw1165/Speed_LLM_RAG/PreSuRe_exps`

Live logs:

`/scratch/mh20/zw1165/Speed_LLM_RAG/PreSuRe_logs/*.live.log`

PBS `.OU`:

`/scratch/mh20/zw1165/Speed_LLM_RAG/PreSuRe_logs/out/`

## Development run

```bash
cd /scratch/mh20/zw1165/Speed_LLM_RAG/PreSuRe
conda activate TFP
python tests/smoke_static.py
python tests/smoke_oracle_math.py
python tests/smoke_local_routing.py
python scripts/preflight.py --stage development
bash scripts/submit_development.sh
```

The dependency chain is:

`PatchSTG artifacts -> retrieval + graph regions -> {Oracle A/B/C/E/F/D, M0, M1 -> M2 -> M3 -> M4 -> M5}`

## Main readout

The important comparisons are:

1. `Oracle-C vs Oracle-F`: how much spatial graph-local routing adds beyond horizon routing.
2. `Oracle-F vs Oracle-D`: how much remaining headroom requires node-specific routing rather than region routing.
3. `M2 vs M3`: whether predictive horizon utility beats global similarity retrieval.
4. `M3 vs M4`: whether graph-region×horizon local routing converts the Oracle-F headroom into deployable gains.
5. `M4 vs M5`: whether local safety control lowers harmful augmentation while preserving/improving Masked and Support MAE.

The v0.3 numerical goal is not to approach the extremely loose Oracle-D. A useful first target is a reproducible **Masked/Support improvement of roughly 4–8% over M0** with a reduced harmful-augmentation rate. Only after this numerical local-routing system is validated should Qwen reasoning be added as M6/M7.

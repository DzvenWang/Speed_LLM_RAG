# Online PreSuRe v3.3 Design

## Research hypothesis

Under incomplete sensing, mature online memory already contains observations that the current PatchSTG parameters and current incomplete input do not contain. v3.2 shows that candidate availability is high, but similarity is a poor proxy for downstream value. The v3.3 hypothesis is:

> support should be selected by expected downstream forecasting utility, not by similarity alone.

## Architecture

```text
Slow Online PatchSTG θ_t
        |
        +-----------------------> Y_slow

Current X_obs, mask
        |
        v
Global Top-64 causal candidate discovery
        |
        v
Node × temporal-patch candidate features
        |
        +---------- similarity control
        |
        v
Online Utility Scorer ψ_t
 Helpful / Neutral / Harmful
        |
      abstain/reject
        |
        v
Patch-wise aligned missing completion
        |
        v
Slow Online PatchSTG θ_t
        |
        v
Fixed safe fusion with Y_slow
```

## Utility supervision protocol

For a query at time t, a limited number of candidate patch interventions are forecast immediately, while Y_t is unavailable. The model stores only their predictions and inference-visible features. When the query matures, downstream utility is computed against the now-observed future and added to the utility buffer. Thus ψ_t can only learn from past resolved cases.

## Why classification instead of direct utility regression

Previous utility regression/ranking experiments were unstable and false-positive interventions are asymmetric. v3.3 uses three classes and explicitly penalizes harmful probability. The scorer can abstain rather than force a completion.

## Visible feature groups

1. Global candidate context: frozen latent similarity, observed match, global complementarity, recency, TOD/DOW context.
2. Node compatibility: observed node-history match and overlap.
3. Temporal-patch compatibility: visible patch match and overlap.
4. Missing-information supply: node/patch complementarity and candidate patch coverage.
5. Candidate consistency: agreement with the mature candidate pool.
6. Ranking context: global rank/score and age.

No current hidden truth or future is a feature.

## Experimental falsification ladder

```text
Slow Online
  -> Global Similarity RAG
  -> Patch Similarity RAG
  -> Patch Predictive-Utility RAG
```

The decisive v3.3 test is whether utility ranking improves over patch similarity while every other retrieval/completion setting remains fixed.

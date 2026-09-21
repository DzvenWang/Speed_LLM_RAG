# PreSuRe v1.2 design rationale

## Evidence carried forward

The Retrieval Information Test found that the Dual Query materially outperformed Backbone-only retrieval on both Development and frozen Final confirmation. The additional sensing-evidence view was weak on its own but complementary to PatchSTG predictive state. This motivates a permanent two-view query rather than another Backbone-hidden probe.

The same test also showed that direct historical-future/residual blending had negligible gain. Therefore v1.2 treats historical futures as **conditional evidence**, never as values to copy.

## Dual Query

For each node, the state view contains the PatchSTG horizon trajectory and the explicitly available PatchSTG node representation. The evidence view contains incomplete history, observation mask, temporal transitions, graph-neighbor evidence, missingness duration, physical gradients and time context.

The model learns a vector gate between the two views. The global retrieval descriptor uses the same principle and is the only learned reranker used in v1.2.

## Top-8 support

The candidate pool remains leakage-safe K0=24. The frozen Dual-Query ranker compresses it to Top-8. Top-8 is intentionally larger than the earlier Top-4 so moderate rank errors do not permanently remove useful historical regimes.

## Historical episode encoding

A support episode contributes:

- support predictive state and sensing evidence;
- its already-realized historical future;
- its historical PatchSTG forecast;
- historical residual as an input feature only;
- future validity.

The residual is never transported directly.

## Multi-support interpretation

At each node x horizon, Top-8 support tokens first use self-attention. This gives the model access to support agreement/disagreement before the query attends to the set. A mild normalized entropy floor discourages immediate hard Top-1 collapse but does not require uniform weights.

## Standalone retrieval-conditioned forecast

The retrieval branch produces a full forecast. It receives a direct auxiliary forecasting loss so its gradients are not starved by the mixture controller.

The final output is

`Y = (1-alpha) * Y_base + alpha * Y_retrieval`

with node x horizon alpha. Alpha starts at 0.25. The retrieval forecast head is initialized to reproduce the base forecast exactly, so the complete system also starts exactly from PatchSTG.

## Protocol

- Current-query future is used only for Train/Development supervision of the retriever.
- Final Validation is not evaluated during model selection.
- The forecasting model never sees current-query future except as its normal supervised target.
- Support memory is Train-only for Development/Final queries.


## v1.2.1 attention-batch fix
Support-set self-attention keeps the original node × horizon semantics but chunks the flattened `B*N*P` batch before `nn.MultiheadAttention`. This avoids the PyTorch efficient-attention 65,535 effective-batch limit (`16*388*12=74,496`) without reducing the formal training batch size.

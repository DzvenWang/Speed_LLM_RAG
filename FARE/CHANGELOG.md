# Changelog

## 0.2.1
- Memory-safe learned retrieval indexing for local GPUs and V100-class accelerators.
- Automatic VRAM-aware embedding micro-batch selection; CUDA OOM halves batch and retries without changing model capacity.
- Retrieval embeddings are streamed directly to a NumPy memmap instead of accumulated in RAM.
- Candidate-assignment embedding uses the same adaptive micro-batching.
- Keeps the v0.2.0 model/checkpoint schema and `v0.2/<run_mode>` artifact paths so completed Context checkpoints remain reusable.

# Changelog

## 0.2.0
- Replaced all v0.1 heuristic placeholders with full-strength experts.
- Added trainable graph-temporal Context restoration optimized by reconstruction + frozen-STLA forecast loss.
- Added causal multi-day TMCP-style Structural restoration using 28 train-only same-slot support days.
- Added learned Context-embedding retrieval index, causal Top-K search, target-aware candidate reranking, support fusion and conservative learned retrieval gating.
- Added STLA-forecast-conditioned Backbone-Aware residual restoration.
- Added safe initialization for Retrieval and Backbone-Aware refinements.
- Added Expert Oracle and Safe Oracle diagnostics.
- Added resumable artifact lifecycle and strict dependency/leakage policies.

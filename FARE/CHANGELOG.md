# Changelog

## 0.2.3
- Kept the full-strength four expert architectures unchanged.
- Split Context, Retrieval and Backbone-Aware training into Stage A raw restoration and Stage B prediction control.
- Added formal train/val counterfactual alpha-utility labels over `[0,.25,.5,.75,1]`.
- Added explicit alpha supervision for the query mean of each cell-wise prediction-control gate.
- Gate bias is initialized from the training alpha distribution instead of a fixed conservative bias.
- Replaced Retrieval listwise softmax utility targets with pairwise utility ranking; logs pairwise loss and accuracy against random references (`ln2`, `0.5`).
- Removed the control-gate sparsity penalty that made `g=0` an attractive local optimum.
- Preserved causal/train-only support and all evaluation leakage constraints.
- Artifact namespace moved to `v0.2.3`.

## 0.2.2
- Separated dense raw restoration from forecast intervention control.
- Added baseline-relative no-harm control and candidate downstream utility labels.

## 0.2.1
- Adaptive CUDA micro-batching and streaming memmaps for Retrieval index construction.

## 0.2.0
- Full-strength Context / Structural / Retrieval / Backbone-Aware experts.

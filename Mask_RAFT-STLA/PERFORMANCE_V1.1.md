# Mask_RAFT-STLA v1.1 performance changes

Measured on Gadi V100-32GB with v1.0 formal batch=16: about 4.4GB VRAM and ~88% GPU utilisation. v1.1 raises formal batch to 64 and moves full-train future evidence aggregation to GPU when possible. Scientific architecture/losses/epochs remain unchanged.

Gadi PBS remains `ncpus=12`, `ngpus=1`, `mem=32GB`.

## v1.1.1 logging update

The Gadi `formal.pbs` now prefixes every line in the main `.live.log` with `[YYYY-MM-DD HH:MM:SS]`, including Python warnings/errors, preflight output, retrieval/store messages, batch progress, epoch summaries, and start/finish lines. GPU CSV data rows continue to contain the timestamp field returned by `nvidia-smi`.

# Mask_RAST-STLA v1.1 performance changes

Measured on Gadi V100-32GB with v1.0 formal batch=12: about 3.6GB VRAM and ~38-76% GPU utilisation. v1.1 raises formal batch to 32, enables DataLoader prefetching, preloads the full-train repository to GPU when possible, combines temporal/spatial retrieved-history encoding, and raises latent-store refresh batch size to 256. Scientific architecture/losses/epochs remain unchanged.

Gadi PBS remains `ncpus=12`, `ngpus=1`, `mem=32GB`.

## v1.1.1 logging update

The Gadi `formal.pbs` now prefixes every line in the main `.live.log` with `[YYYY-MM-DD HH:MM:SS]`, including Python warnings/errors, preflight output, retrieval/store messages, batch progress, epoch summaries, and start/finish lines. GPU CSV data rows continue to contain the timestamp field returned by `nvidia-smi`.

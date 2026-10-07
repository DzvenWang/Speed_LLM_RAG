# Mask_RAST-STLA v1.1 performance changes

Measured on Gadi V100-32GB with v1.0 formal batch=12: about 3.6GB VRAM and ~38-76% GPU utilisation. v1.1 raises formal batch to 32, enables DataLoader prefetching, preloads the full-train repository to GPU when possible, combines temporal/spatial retrieved-history encoding, and raises latent-store refresh batch size to 256. Scientific architecture/losses/epochs remain unchanged.

Gadi PBS remains `ncpus=12`, `ngpus=1`, `mem=32GB`.

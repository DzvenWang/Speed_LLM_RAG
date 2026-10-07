# Mask_RAFT-STLA v1.1 performance changes

Measured on Gadi V100-32GB with v1.0 formal batch=16: about 4.4GB VRAM and ~88% GPU utilisation. v1.1 raises formal batch to 64 and moves full-train future evidence aggregation to GPU when possible. Scientific architecture/losses/epochs remain unchanged.

Gadi PBS remains `ncpus=12`, `ngpus=1`, `mem=32GB`.

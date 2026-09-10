# Official Sources — Baseline v0.4.0

Vendor model code is not silently patched. Adaptation to Shared Protocol is performed outside the vendor tree.

## Exact pre-pinned sources

- ImputeFormer — `tongnie/ImputeFormer` — `1c3868cf499951ac43b4fc7280593fbb745f58c7`
- SLPF — `zbliu98/SLPF` — `8f135a96d57651dc19564bd3b94508c1f92c04fb`
- RAFT — `archon159/RAFT` — `bfe2320f5d719d1e2a024330de3410b370bbb2a9`
- RAST — `RWLinno/RAST` — `4b48a47a002dd8d4151afe47be8f436f0c028a67`
- LEAF — `YushengZhao/LEAF` — `650b9e915fd4cc213387e279e4e529acbe587b85`

## Resolve-once sources

- GinAR — `GestaltCogTeam/GinAR`, branch `main`
- SARAF — `ShiqiaoZhou/SARAF`, branch `main`

For GinAR/SARAF, the first successful Gadi setup resolves `main` to the exact 40-character commit and writes it to `vendor/SOURCE_LOCK.json` and `vendor/RESOLVED_SOURCES.tsv`. All later setup/verification reuses that exact commit. This avoids inventing a commit while still making the experiment immutable after first resolution.

After `SOURCE LOCK PASS`, run `scripts/strip_vendor_git.sh` if inode reduction is desired; tree SHA256 verification remains active.

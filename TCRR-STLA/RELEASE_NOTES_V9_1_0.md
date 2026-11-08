# TCRR-STLA v9.1.0 — CF mechanism and engineering corrections

Replacement for v9.0.0 gradient-zero hotfix source. New project stays inside `TCRR-STLA`, under `tcrr_stla/v9`. Legacy v5 sources and root version are retained.

**Fixes:** (1) direct target-node and horizon-sensitive CF Utility features; (2) strict future-valid utility targets and local Ranking Teacher; (3) frozen Normal Anchor across Transfer/Utility/Cross; (4) historical candidate ID alignment across masks with matched-rate checks; (5) train-only full-information positive-gain response supervision; (6) query-mask reliability-weighted three-graph CF messages; (7) paired view masks/family/severity/sha manifests; (8) same-cohort Validation negative controls and separate graph controls; (9) v9.1 cache/run namespace isolation and checkpoint version validation.

**Research cautions:** model gains versus STLAformer/v5.3 are unknown until Gadi experiments run. Physical upstream/downstream are optional and require real graph metadata. Fully OOF historical residual generation, formal risk calibration and paper-grade multiseed robustness are not established by this release.

**Files:** `docs/V9_1_IMPLEMENTATION.md` and `docs/V9_RUNBOOK.md` contain equations, interface contracts, and Speed/Flow PBS commands. The separate ZIP SHA256 file validates the delivered archive.

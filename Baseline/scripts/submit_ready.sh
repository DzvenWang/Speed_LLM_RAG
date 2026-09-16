#!/bin/bash
set -euo pipefail
cat <<'EOF'
Baseline v1.0.0 clean Primary-7 workflow

Primary: ImputeFormer, LRTC-TMCP, GinAR, RAFT, RAST, SARAF, ReMFT
Secondary: SPRINT, SLPF, MoGERNN, LEAF/LEAF-Qwen1.5B, RAST-PatchSTG

1) bash scripts/submit_primary_smoke.sh
2) bash scripts/submit_primary_development.sh
3) bash scripts/submit_ginar_stability_probe.sh 10
4) bash scripts/submit_primary_formal_seed10.sh
5) bash scripts/submit_primary_formal_remaining.sh  # seeds20/30 after seed10 sanity check
6) python scripts/primary_run_status.py
7) python scripts/freeze_formal_selection.py
8) bash scripts/submit_evaluation_all.sh
EOF

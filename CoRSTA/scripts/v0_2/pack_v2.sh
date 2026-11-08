#!/usr/bin/env bash
set -euo pipefail
ROOT="${SPEED_LLM_RAG_ROOT:-$PWD}"
cd "$ROOT"
STAMP="$(date -u +%Y%m%d_%H%M%S)"
mkdir -p CoRSTA_v022_diagnostics
OUT="CoRSTA_v022_diagnostics/CoRSTA_v022_speed_flow_${STAMP}.tar.gz"
paths=(CoRSTA_v022_exps CoRSTA_v022_logs CoRSTA/corsta_v2 CoRSTA/configs/v0_2 CoRSTA/scripts/v0_2 CoRSTA/pbs/v0_2 CoRSTA/README_V022.md)
actual=()
for x in "${paths[@]}"; do [[ ! -e "$x" ]] || actual+=("$x"); done
[[ ${#actual[@]} -gt 0 ]] || { echo 'Nothing to pack' >&2; exit 2; }
tar -czf "$OUT" --exclude='*.pt' --exclude='*.npy' --exclude='*.npz' \
  --exclude='__pycache__' --exclude='*.partial*' "${actual[@]}"
sha256sum "$OUT"
ls -lh "$OUT"

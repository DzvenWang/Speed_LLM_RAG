#!/usr/bin/env bash
set -euo pipefail
ROOT="${SPEED_LLM_RAG_ROOT:-$PWD}"
cd "$ROOT"
DIAG_ROOT="${D1_ROOT:-CoRSTA_diagnostics/D1_Net06_seed10}"
OUT="${D1_PACKAGE:-CoRSTA_diagnostics/CoRSTA_D1_speed_flow_$(date -u +%Y%m%d_%H%M%S).tar.gz}"
mkdir -p "$(dirname "$OUT")"
[[ -f "$DIAG_ROOT/speed/D1_COMPLETE.json" ]] || { echo 'Missing completed speed diagnostic' >&2; exit 2; }
[[ -f "$DIAG_ROOT/flow/D1_COMPLETE.json" ]] || { echo 'Missing completed flow diagnostic' >&2; exit 2; }
extra=()
for path in "CoRSTA_logs/diagnostics/D1/speed" "CoRSTA_logs/diagnostics/D1/flow"; do
  [[ -d "$path" ]] && extra+=("$path")
done
tar -czf "$OUT" --exclude='*.pt' --exclude='*.npy' --exclude='*.npz' \
  --exclude='*.partial' "$DIAG_ROOT/speed" "$DIAG_ROOT/flow" \
  "${extra[@]}" CoRSTA/diagnostics/*.py CoRSTA/pbs/diagnose_d1.pbs
sha256sum "$OUT" > "$OUT.sha256"
ls -lh "$OUT" "$OUT.sha256"

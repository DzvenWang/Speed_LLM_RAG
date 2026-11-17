#!/usr/bin/env bash
# Pack ReShift results/diagnostics (no checkpoints, no banks) for analysis.
# Usage: bash ReShift-STLA/scripts/reshift/pack.sh [target ...]   (default: speed flow)
set -euo pipefail
ROOT="${SPEED_LLM_RAG_ROOT:?set SPEED_LLM_RAG_ROOT}"
cd "$ROOT"
targets=("$@"); [[ ${#targets[@]} -gt 0 ]] || targets=(speed flow)
out="ReShift_pack_$(date +%Y%m%d_%H%M%S).tar.gz"
files=()
for t in "${targets[@]}"; do
  while IFS= read -r f; do files+=("$f"); done < <(find "ReShift_exps/$t" -type f \( -name 'RESULTS.md' -o -name 'summary.json' \
     -o -name 'm0.json' -o -name 'm1.json' -o -name 'COMPLETE.json' -o -name 'model_lock.json' \
     -o -name 'batch_probe.json' -o -path '*/test/s*.json' -o -path '*/audit/*.json' -o -name 'fingerprint.json' \) 2>/dev/null)
  while IFS= read -r f; do files+=("$f"); done < <(find "ReShift_logs/$t" -type f -name '*.log' 2>/dev/null)
done
[[ ${#files[@]} -gt 0 ]] || { echo 'Nothing to pack'; exit 1; }
tar -czf "$out" "${files[@]}"
echo "PACKED $out (${#files[@]} files)"

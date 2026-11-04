#!/bin/bash
# Pack only diagnostics text tables, hashes and reports. Avoid model/checkpoint/cache tensors.
set -euo pipefail
ROOT="${SPEED_LLM_RAG_ROOT:-$PWD}"
cd "$ROOT"
BASE="${1:-PCFT-STLA_diagnostics/v7_cpu}"
OUT="${2:-PCFT_v7_CPU_Diagnostic_Analysis_Small.tar.gz}"
[[ -d "$BASE" ]] || { echo "Missing diagnostic output directory: $BASE";exit 2; }
[[ "$OUT" != /* ]] || { echo 'Specify relative output archive in project root'; exit 2; }
tar -czf "$OUT" \
  --exclude='*.pt' --exclude='*.pth' --exclude='*.npz' --exclude='*.npy' \
  --exclude='*.pkl' --exclude='*.parquet' --exclude='*.safetensors' \
  "$BASE"
ls -lh "$OUT"
tar -tzf "$OUT" | tail -20

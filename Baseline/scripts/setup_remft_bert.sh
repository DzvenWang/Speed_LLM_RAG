#!/bin/bash
set -euo pipefail

ROOT=/scratch/mh20/zw1165/Speed_LLM_RAG/Baseline
SRC=${1:-${REMFT_BERT_PATH:-}}
DST="$ROOT/vendor/ReMFT/LLM/BERT"

if [[ -z "$SRC" ]]; then
  echo "Usage: REMFT_BERT_PATH=/path/to/local/bert-base-uncased $0" >&2
  echo "   or: $0 /path/to/local/bert-base-uncased" >&2
  exit 2
fi

SRC=$(python -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$SRC")
DST=$(python -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$DST")

if [[ "$SRC" == "$DST" ]]; then
  echo "ERROR: REMFT_BERT_PATH must not equal ReMFT target directory:" >&2
  echo "  $DST" >&2
  echo "Store BERT in an external model/cache directory and link it here." >&2
  exit 4
fi

if [[ ! -f "$SRC/config.json" ]]; then
  echo "Missing $SRC/config.json" >&2
  exit 3
fi

if [[ ! -f "$SRC/model.safetensors" && ! -f "$SRC/pytorch_model.bin" ]]; then
  echo "Missing BERT weights under $SRC" >&2
  exit 3
fi

mkdir -p "$(dirname "$DST")"
rm -rf "$DST"
ln -s "$SRC" "$DST"

echo "ReMFT local BERT link ready:"
echo "  $DST"
echo "  -> $SRC"

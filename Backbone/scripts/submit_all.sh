#!/bin/bash
set -euo pipefail
ROOT=/scratch/mh20/zw1165/Speed_LLM_RAG/Backbone
cd "$ROOT"
echo "Backbone v0.5.0: submitting the complete fresh five-model screening." >&2
echo "Old v0.4.0 screening/formal results are intentionally not reusable." >&2
exec "$ROOT/scripts/submit_screening.sh"

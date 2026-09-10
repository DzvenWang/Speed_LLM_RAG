#!/bin/bash
set -euo pipefail
echo "v0.4.0 migration-safe default: submit_all.sh submits Stage-1B recent models only." >&2
echo "Use scripts/submit_screening.sh explicitly only for a fresh six-model screening." >&2
exec /scratch/mh20/zw1165/Speed_LLM_RAG/Backbone/scripts/submit_recent_screening.sh

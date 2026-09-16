#!/bin/bash
set -euo pipefail
ROOT=/scratch/mh20/zw1165/Speed_LLM_RAG/Backbone
cd "$ROOT"
echo "Backbone v0.6.0: submitting the complete fresh six-model screening." >&2
echo "Existing v0.5.0 five-model screening summaries are not complete for v0.6.0 because STLAformer is now an active candidate." >&2
exec "$ROOT/scripts/submit_screening.sh"

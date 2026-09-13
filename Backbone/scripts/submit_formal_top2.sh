#!/bin/bash
set -euo pipefail
echo "Backbone v0.5.0 no longer couples downstream formal training to a screening Top-2 shortlist." >&2
echo "Submitting the configured reference_model instead." >&2
exec "$(dirname "$0")/submit_reference_formal.sh"

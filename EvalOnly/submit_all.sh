#!/bin/bash
# New default: one sequential GPU job for Backbone and one for Baseline.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python3 -u "$HERE/scripts/submit_group.py" "$@"

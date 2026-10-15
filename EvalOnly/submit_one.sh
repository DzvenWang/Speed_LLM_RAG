#!/bin/bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENVROOT="${TFP_ENV_ROOT:-${CONDA_PREFIX:-}}"
[[ -n "$ENVROOT" && -x "$ENVROOT/bin/python" ]] || { echo "ERROR: activate TFP before submitting" >&2; exit 3; }
exec "$ENVROOT/bin/python" -u "$HERE/scripts/submit_one.py" "$@"

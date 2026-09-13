#!/bin/bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")" && pwd)
cd "$ROOT"
python validate_joint_contract.py
python ExperimentCommon/tests/smoke_contract.py
python ExperimentCommon/tests/smoke_cross_project_metrics.py
python Backbone/scripts/smoke_metrics.py
python Backbone/scripts/smoke_identity_contract.py
python Backbone/scripts/smoke_time_feature_contract.py
python Backbone/scripts/smoke_direct_cache_contract.py
for f in Baseline/tests/*.py; do python "$f"; done
echo "JOINT STATIC/CONTRACT SMOKE PASS"

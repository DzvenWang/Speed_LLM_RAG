from pathlib import Path
import sys

_COMMON = Path(__file__).resolve().parents[2] / 'ExperimentCommon'
if not _COMMON.exists():
    _COMMON = Path('/scratch/mh20/zw1165/Speed_LLM_RAG/ExperimentCommon')
if str(_COMMON) not in sys.path:
    sys.path.insert(0, str(_COMMON))

from experiment_common.constants import METRICS_REVISION  # noqa: E402,F401
from experiment_common.metrics import (  # noqa: E402,F401
    _metric,
    mae_rmse,
    metric_bundle,
    per_query_metrics,
    backbone_metric_summary,
)

__all__ = [
    '_metric', 'mae_rmse', 'metric_bundle', 'per_query_metrics',
    'backbone_metric_summary', 'METRICS_REVISION'
]

from pathlib import Path
import sys

_COMMON = Path(__file__).resolve().parents[2] / 'ExperimentCommon'
if not _COMMON.exists():
    _COMMON = Path('/scratch/mh20/zw1165/Speed_LLM_RAG/ExperimentCommon')
if str(_COMMON) not in sys.path:
    sys.path.insert(0, str(_COMMON))

from experiment_common.predictions import (  # noqa: E402,F401
    as_unicode_array,
    save_prediction_npz,
    load_prediction_npz,
)

__all__ = ['as_unicode_array', 'save_prediction_npz', 'load_prediction_npz']

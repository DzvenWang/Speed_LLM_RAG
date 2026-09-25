"""Compatibility re-export of the shared Experiment Contract time-feature implementation."""
import sys
from .utils import repository_root
COMMON = repository_root() / 'ExperimentCommon'
if str(COMMON) not in sys.path:
    sys.path.insert(0, str(COMMON))
from experiment_common.time_features import *  # noqa: F401,F403,E402
from experiment_common.constants import DATA_CONTRACT_REVISION, TIME_FEATURE_REVISION  # noqa: E402,F401

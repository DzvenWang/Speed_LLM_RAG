"""Compatibility re-export of the shared Experiment Contract time-feature implementation."""
from pathlib import Path
import sys
COMMON=Path('/scratch/mh20/zw1165/Speed_LLM_RAG/ExperimentCommon')
if not COMMON.exists(): COMMON=Path(__file__).resolve().parents[2]/'ExperimentCommon'
if str(COMMON) not in sys.path: sys.path.insert(0,str(COMMON))
from experiment_common.time_features import *  # noqa: F401,F403
from experiment_common.constants import DATA_CONTRACT_REVISION, TIME_FEATURE_REVISION

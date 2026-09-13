"""Baseline metric API backed by Shared Experiment Contract v1."""
from pathlib import Path
import sys
COMMON=Path('/scratch/mh20/zw1165/Speed_LLM_RAG/ExperimentCommon')
if not COMMON.exists(): COMMON=Path(__file__).resolve().parents[2]/'ExperimentCommon'
if str(COMMON) not in sys.path: sys.path.insert(0,str(COMMON))
from experiment_common.metrics import *  # noqa: F401,F403

"""Deprecated compatibility helpers.

Data identity is owned by ExperimentCommon. New code should import
experiment_common.cache directly.
"""
from pathlib import Path
import sys
COMMON=Path('/scratch/mh20/zw1165/Speed_LLM_RAG/ExperimentCommon')
if not COMMON.exists(): COMMON=Path(__file__).resolve().parents[2]/'ExperimentCommon'
if str(COMMON) not in sys.path: sys.path.insert(0,str(COMMON))
from experiment_common.cache import contract_fingerprint as data_config_sha256  # compatibility alias

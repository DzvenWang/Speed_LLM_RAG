"""Deprecated compatibility helpers.

Data identity is owned by Base_SupportBench cache layer. New code should import
experiment_common.cache directly.
"""
from pathlib import Path
import sys
COMMON=(next((p for p in Path(__file__).resolve().parents if p.name == 'Baseline'), Path.cwd())).parent / 'Base_SupportBench'
if not COMMON.exists(): COMMON=Path(__file__).resolve().parents[2]/'Base_SupportBench'
if str(COMMON) not in sys.path: sys.path.insert(0,str(COMMON))
from experiment_common.cache import contract_fingerprint as data_config_sha256  # compatibility alias

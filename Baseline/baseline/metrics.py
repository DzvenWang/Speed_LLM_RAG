"""Baseline metric API backed by Shared Experiment Contract v1."""
from pathlib import Path
import sys
COMMON=(next((p for p in Path(__file__).resolve().parents if p.name == 'Baseline'), Path.cwd())).parent / 'Base_SupportBench'
if not COMMON.exists(): COMMON=Path(__file__).resolve().parents[2]/'Base_SupportBench'
if str(COMMON) not in sys.path: sys.path.insert(0,str(COMMON))
from experiment_common.metrics import *  # noqa: F401,F403

#!/usr/bin/env python3
from pathlib import Path
import ast
ROOT=Path(__file__).resolve().parents[1]
formal=(ROOT/'pbs/methods/formal_primary.pbs').read_text()
run_eval=(ROOT/'scripts/run_evaluation.py').read_text()
run_evals=(ROOT/'scripts/run_evaluations.py').read_text()
pre=(ROOT/'scripts/preflight.py').read_text()
cfg=(ROOT/'configs/baseline.yaml').read_text()
assert 'AUTO_EVAL="${AUTO_EVAL:-1}"' in formal
assert 'scripts/run_evaluations.py --method "$METHOD" --seed "$SEED" --formal-run-dir "$FORMAL_RUN_DIR"' in formal
assert "--formal-run-dir" in run_eval
assert 'direct_formal_run_no_baseline_freeze' in run_eval
assert "--formal-run-dir" in run_evals
assert 'no FORMAL_SELECTION/FROZEN_BASELINE_PROTOCOL required' in pre
assert 'direct_multiseverity_evaluation_after_completed_formal_run_no_baseline_freeze' in cfg
ast.parse(run_eval)
ast.parse(run_evals)
ast.parse(pre)
print('SMOKE direct Formal auto-evaluation PASS')

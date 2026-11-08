"""Reject incomplete formal artifacts before a Gadi POST job spends GPU time."""
import os
import subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def test_postprocess_rejects_missing_controller_and_summary(tmp_path):
    src=(ROOT/'pbs/postprocess_v5.pbs').read_text()
    src=src.replace('source "${PBS_O_WORKDIR:?}/TCRR-STLA/pbs/runtime.sh"',
                    'PYTHON="${FAKE_PYTHON:?}"')
    script=tmp_path/'post.pbs';script.write_text(src)
    py=tmp_path/'python';py.write_text('#!/bin/sh\necho executed >> "${CALL_LOG:?}"\n')
    py.chmod(0o755)
    run=tmp_path/'run';run.mkdir()
    (run/'best_c12.pt').write_bytes(b'x')
    log=tmp_path/'called.txt'
    env={**os.environ,'RUN_DIR':str(run),'FAKE_PYTHON':str(py),
         'CALL_LOG':str(log),'PBS_O_WORKDIR':str(ROOT.parent)}
    a=subprocess.run(['bash',str(script)],env=env,cwd=ROOT,capture_output=True,text=True)
    assert a.returncode!=0 and 'best_controller.pt' in a.stderr
    assert not log.exists()
    (run/'best_controller.pt').write_bytes(b'x')
    b=subprocess.run(['bash',str(script)],env=env,cwd=ROOT,capture_output=True,text=True)
    assert b.returncode!=0 and 'summary.json' in b.stderr
    assert not log.exists()

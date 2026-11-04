"""PBS contract tests: no queue, CUDA, or Gadi files required.

Mock Python executable makes sure a fresh PBS shell invokes the selected binary
for every step, reports errors, and does not rely on conda init from a login shell.
"""
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def fake_python(tmp_path):
    shim = tmp_path / 'fake_tfp_python'
    shim.write_text('''#!/bin/bash
set -eu
if [[ "${1:-}" == "-" ]]; then cat >/dev/null; fi
printf 'FAKE_TFP_PYTHON %s\\n' "$*"
''')
    with shim.open('a') as f: f.write('if [[ -n "${FAKE_FAIL_ON:-}" && "$*" == *"$FAKE_FAIL_ON"* ]]; then exit 86; fi\n')
    shim.chmod(0o755)
    return shim


def run_pbs(tmp_path, name, python_bin):
    # Mock successful PBS-mounted project paths. This fixture intentionally
    # exercises the extra compute-node filesystem preflight from v1.3.
    (tmp_path / 'Backbone').mkdir(exist_ok=True)
    (tmp_path / 'Base_SupportBench').mkdir(exist_ok=True)
    script = tmp_path / 'PCFT-STLA/diagnostics/propagation_identifiability/scripts/run_diagnostic.py'
    script.parent.mkdir(parents=True,exist_ok=True)
    script.touch()
    (tmp_path / 'fake_checkpoint.pt').touch()
    env = os.environ.copy()
    env.update(PBS_O_WORKDIR=str(tmp_path), PBS_JOBID='123456.mock',
               TARGET='speed', CHECKPOINT='fake_checkpoint.pt', SEED='10',
               TFP_PYTHON=str(python_bin), PYTHONPATH='')
    return subprocess.run(['bash', str(ROOT / 'pbs' / name)],
                          env=env, capture_output=True, text=True, timeout=20)


def test_gpu_pbs_uses_exact_tfp_executable_for_preflight_and_all(tmp_path):
    proc = run_pbs(tmp_path, 'propagation_diagnostic.pbs', fake_python(tmp_path))
    assert proc.returncode == 0, proc.stderr + '\n' + proc.stdout
    assert proc.stdout.count('FAKE_TFP_PYTHON') == 3
    assert '--phase preflight --preflight-gpu --confirm-val' in proc.stdout
    assert '--phase all --confirm-val' in proc.stdout
    assert 'PHYSICAL_DIAGNOSTIC_END status=success' in proc.stdout
    assert Path(tmp_path / 'PCFT-STLA_logs/propagation_identifiability').exists()


def test_cpu_pbs_uses_tfp_executable(tmp_path):
    proc = run_pbs(tmp_path, 'stats_only.pbs', fake_python(tmp_path))
    assert proc.returncode == 0, proc.stderr + '\n' + proc.stdout
    assert proc.stdout.count('FAKE_TFP_PYTHON') == 2
    assert '--phase stats' in proc.stdout


def test_both_jobs_fail_fast_if_tfp_binary_missing(tmp_path):
    missing = tmp_path / 'missing_python'
    for name in ('stats_only.pbs', 'propagation_diagnostic.pbs'):
        proc = run_pbs(tmp_path, name, missing)
        assert proc.returncode == 120
        assert 'FATAL_TFP_PYTHON_NOT_EXECUTABLE' in proc.stderr


def test_pbs_shell_parses():
    for name in ('stats_only.pbs', 'propagation_diagnostic.pbs'):
        proc = subprocess.run(['bash', '-n', str(ROOT / 'pbs' / name)],
                              capture_output=True, text=True)
        assert proc.returncode == 0, proc.stderr


def test_pipeline_preserves_python_failure_exit_code(tmp_path):
    shim = fake_python(tmp_path)
    (tmp_path/'Backbone').mkdir()
    (tmp_path/'Base_SupportBench').mkdir()
    script=tmp_path/'PCFT-STLA/diagnostics/propagation_identifiability/scripts/run_diagnostic.py'
    script.parent.mkdir(parents=True)
    script.touch()
    (tmp_path/'fake_checkpoint.pt').touch()
    env = os.environ.copy()
    env.update(PBS_O_WORKDIR=str(tmp_path), PBS_JOBID='123456.mock',
               TARGET='speed', CHECKPOINT='fake_checkpoint.pt', SEED='10',
               TFP_PYTHON=str(shim), FAKE_FAIL_ON='--phase all', PYTHONPATH='')
    proc = subprocess.run(['bash', str(ROOT / 'pbs/propagation_diagnostic.pbs')],
                          env=env, capture_output=True, text=True, timeout=20)
    assert proc.returncode == 86, proc.stderr + '\n' + proc.stdout
    assert 'PHYSICAL_DIAGNOSTIC_END status=success' not in proc.stdout


def test_both_pbs_request_gadi_storage_mount():
    # Mandatory PBS requests from the successful historical Backbone/PCFT v6.
    for name in ('stats_only.pbs','propagation_diagnostic.pbs'):
        script=(ROOT/'pbs'/name).read_text()
        assert '#PBS -l storage=scratch/mh20+gdata/mh20' in script
        assert '#PBS -P mh20' in script


def test_gpu_pbs_checks_checkpoint_before_cuda(tmp_path):
    (tmp_path/'Backbone').mkdir()
    (tmp_path/'Base_SupportBench').mkdir()
    p=(tmp_path/'PCFT-STLA/diagnostics/propagation_identifiability/scripts/run_diagnostic.py')
    p.parent.mkdir(parents=True)
    p.touch()
    shim=fake_python(tmp_path)
    env=os.environ.copy()
    env.update(PBS_O_WORKDIR=str(tmp_path),PBS_JOBID='test.mock',TARGET='speed',CHECKPOINT='not_here.pt',TFP_PYTHON=str(shim))
    proc=subprocess.run(['bash',str(ROOT/'pbs/propagation_diagnostic.pbs')],env=env,capture_output=True,text=True,timeout=20)
    assert proc.returncode==123
    assert 'FATAL_CHECKPOINT_NOT_VISIBLE' in proc.stderr
    assert 'FAKE_TFP_PYTHON' not in proc.stdout


def test_compute_node_missing_project_detected_before_python(tmp_path):
    shim=fake_python(tmp_path)
    env=os.environ.copy()
    env.update(PBS_O_WORKDIR=str(tmp_path),PBS_JOBID='test.mock',TARGET='speed',CHECKPOINT='fake_checkpoint.pt',TFP_PYTHON=str(shim))
    proc=subprocess.run(['bash',str(ROOT/'pbs/propagation_diagnostic.pbs')],env=env,capture_output=True,text=True,timeout=20)
    assert proc.returncode==122
    assert 'FATAL_PROJECT_STORAGE_UNAVAILABLE' in proc.stderr
    assert 'FAKE_TFP_PYTHON' not in proc.stdout

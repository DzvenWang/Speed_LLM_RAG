"""Reproduce Gadi's /etc/bashrc nounset failure without using a Gadi account.

The PBS wrappers MUST NOT source a noninteractive user's .bashrc, even when
that file would fail with: /etc/bashrc: BASHRCSOURCED: unbound variable.
"""
from pathlib import Path
import os
import shutil
import subprocess
import tempfile

PBS = Path(__file__).resolve().parents[1] / 'pbs'

MOCK = """import os,sys
from pathlib import Path
with Path(os.environ['FARE_PBS_BOOTSTRAP_MARKER']).open('a') as f:
    f.write(Path(sys.argv[0]).name + ' ' + ' '.join(sys.argv[1:]) + '\\n')
print('FARE_FAKE_WORKLOAD_PASS', flush=True)
"""


def run_test():
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp) / 'Speed_LLM_RAG'
        work = root / 'FARE'; (work / 'pbs').mkdir(parents=True)
        (work / 'scripts').mkdir(parents=True)
        for name in ('train_formal.pbs','evaluate.pbs','calibrate.pbs','oof_controller.pbs','_activate_tfp.sh'):
            shutil.copy2(PBS / name, work / 'pbs' / name)
        for name in ('train.py','evaluate.py','calibrate.py','build_oof.py'):
            (work / 'scripts' / name).write_text(MOCK)
        home = Path(temp) / 'home'; home.mkdir()
        rc_marker = Path(temp) / 'BAD_BASHRC_SOURCED'
        (home / '.bashrc').write_text(
            ': "${BASHRCSOURCED:?/etc/bashrc: BASHRCSOURCED: unbound variable}"\n'
            f'touch "{rc_marker}"\n'
        )
        # Auto-discover the actual directory layout used on Gadi.
        conda_sh = root.parent / 'miniconda' / 'etc' / 'profile.d' / 'conda.sh'
        conda_sh.parent.mkdir(parents=True)
        conda_sh.write_text('conda() { if [[ "$1" == activate ]]; then export CONDA_DEFAULT_ENV="$2"; fi; }\n')
        ckpt=root/'candidate_best.pt'; ckpt.write_text('test')
        marker=Path(temp)/'calls.txt'
        common=dict(os.environ,HOME=str(home),SPEED_LLM_RAG_ROOT=str(root),
                    PBS_O_WORKDIR=str(root), PBS_JOBID='123.gadi-pbs',
                    FARE_PBS_BOOTSTRAP_MARKER=str(marker),
                    RUN_DIR=str(root), DATASET='TraffiDent_Net_06', TARGET='speed',
                    CHECKPOINT=str(ckpt))
        common.pop('FARE_CONDA_SH',None)
        common.pop('BASH_ENV',None)
        for script in ('train_formal.pbs','evaluate.pbs','calibrate.pbs','oof_controller.pbs'):
            result = subprocess.run(['bash','--noprofile','--norc',str(work/'pbs'/script)],
                                  env=common,cwd=root,capture_output=True,text=True,timeout=15)
            assert result.returncode == 0, (script,result.stdout,result.stderr)
            assert 'FARE_CONDA_READY' in result.stdout, (script,result.stdout)
            assert 'FARE_FAKE_WORKLOAD_PASS' in result.stdout, (script,result.stdout)
            assert not rc_marker.exists(), f'{script} unexpectedly sourced the hostile .bashrc'
        assert len(marker.read_text().splitlines()) == 4
        # Missing Conda => precise startup failure, not an undefined bashrc var.
        conda_sh.unlink()
        bad=dict(common, FARE_CONDA_SH=str(Path(temp)/'missing'/'conda.sh'))
        result=subprocess.run(['bash','--noprofile','--norc',str(work/'pbs'/'train_formal.pbs')],
                              env=bad,cwd=root,capture_output=True,text=True,timeout=15)
        assert result.returncode != 0, (result.stdout,result.stderr)
        assert 'FARE_CONDA_INIT_ERROR' in result.stderr, (result.stdout,result.stderr)
        assert 'BASHRCSOURCED' not in result.stderr, (result.stdout,result.stderr)
        print('FARE_GADI_BASHRC_BOOTSTRAP_TEST_PASS: all PBS paths bypass .bashrc; conda autoload; explicit error')

if __name__ == '__main__':
    run_test()

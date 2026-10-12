"""Gadi qsub orchestration: same-allocation post-train Evaluate, no held job.

Only fake qsub is used; this does NOT claim execution on Gadi hardware.
"""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

RUNNER = Path(__file__).resolve().parents[1] / 'scripts' / 'submit_gadi.py'
PBS = Path(__file__).resolve().parents[1] / 'pbs'
FAKE_QSUB = r'''#!/usr/bin/env python3
import os,sys
from pathlib import Path
counter=Path(os.environ['FARE_FAKE_QSUB_COUNTER'])
n=int(counter.read_text())+1 if counter.exists() else 1
counter.write_text(str(n))
with Path(os.environ['FARE_FAKE_QSUB_LOG']).open('a') as f:
 f.write(__import__('json').dumps(sys.argv[1:])+'\n')
print(f'{10000+n}.gadi-pbs')
'''


def run_test():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        target = root / 'FARE'
        (target / 'scripts').mkdir(parents=True)
        (target / 'pbs').mkdir(parents=True)
        shutil.copy2(RUNNER, target / 'scripts' / 'submit_gadi.py')
        for pbs in PBS.glob('*.pbs'):
            shutil.copy2(pbs, target / 'pbs' / pbs.name)
        fake_bin = root / 'bin'; fake_bin.mkdir()
        cmd = fake_bin / 'qsub'; cmd.write_text(FAKE_QSUB); cmd.chmod(0o755)
        log = root/'qsub.jsonl'
        env = dict(os.environ, SPEED_LLM_RAG_ROOT=str(root),
                   PATH=f'{fake_bin}{os.pathsep}{os.environ["PATH"]}',
                   FARE_FAKE_QSUB_COUNTER=str(root/'counter'),
                   FARE_FAKE_QSUB_LOG=str(log))
        runner = target/'scripts'/'submit_gadi.py'
        run = lambda *args: subprocess.run([sys.executable,str(runner),*args],
                          capture_output=True,text=True,env=env)
        result = run('train','--dataset','TraffiDent_Net_06',
                     '--targets','speed','flow','--seed','10')
        assert result.returncode==0, (result.stdout,result.stderr)
        records = [json.loads(x) for x in log.read_text().splitlines()]
        assert len(records)==2,records
        for i,target_name in enumerate(('speed','flow')):
            tr=records[i]
            assert tr[-1].endswith('/train_formal.pbs')
            assert '-W' not in tr
            vars=tr[tr.index('-v')+1]
            assert f'TARGET={target_name}' in vars
            assert 'AUTO_EVALUATE=1' in vars
            assert 'EVAL_SPLITS=test' in vars
            assert 'SEVERITIES=20:40:60:90' in vars
        assert 'FARE_GADI_SUBMISSION_PASS targets=speed,flow' in result.stdout
        # A manually configured conda.sh override is forwarded through qsub -v.
        override = dict(env, FARE_CONDA_SH='/scratch/mh20/zw1165/miniconda/etc/profile.d/conda.sh')
        forwarded = subprocess.run([sys.executable,str(runner),'train','--dataset',
             'TraffiDent_Net_06','--targets','speed','--dry-run'],
             capture_output=True,text=True,env=override)
        assert forwarded.returncode==0,(forwarded.stdout,forwarded.stderr)
        assert 'FARE_CONDA_SH=/scratch/mh20/zw1165/miniconda/etc/profile.d/conda.sh' in forwarded.stdout
        assert result.stdout.count('FARE_GADI_IN_JOB_EVALUATE_CONFIGURED')==2
        dirs=list((root/'FARE_exps').glob('*/TraffiDent_Net_06/formal/*'))
        assert len(dirs)==2
        for d in dirs:
            job=json.loads((d/'gadi_jobs.json').read_text())
            assert job['evaluation_in_train_allocation']
            assert job['eval_splits']==['test']
            assert set(job['jobs'])=={'train'}
        # Pilot: no second PBS job, default Validation+Test.
        before=len(records)
        pilot_cfg=root/'pilot.yaml'
        pilot_cfg.write_text('experiment:\n  run_tag: pilot\n')
        result=run('train','--dataset','TraffiDent_Net_06',
                   '--targets','speed','flow','--config',str(pilot_cfg))
        assert result.returncode==0,(result.stdout,result.stderr)
        records=[json.loads(x) for x in log.read_text().splitlines()]
        assert len(records)==before+2,records
        assert all('EVAL_SPLITS=val:test' in rec[rec.index('-v')+1] for rec in records[-2:])
        # Explicit Pilot Test: still only one qsub; passes split list + severities.
        before=len(records)
        result=run('train','--dataset','TraffiDent_Net_06','--targets','speed',
                   '--config',str(pilot_cfg),'--eval-splits','val','test',
                   '--severities','20','60')
        assert result.returncode==0,(result.stdout,result.stderr)
        records=[json.loads(x) for x in log.read_text().splitlines()]
        assert len(records)==before+1
        pvars=records[-1][records[-1].index('-v')+1]
        assert 'EVAL_SPLITS=val:test' in pvars and 'SEVERITIES=20:60' in pvars
        assert 'FARE_PILOT_TEST_EXPLORATORY' in result.stdout
        assert '-W' not in records[-1]
        # Disable auto evaluation: still only one Train qsub, no Eval.
        before=len(records)
        result=run('train','--dataset','TraffiDent_Net_06','--targets','flow',
                   '--no-auto-evaluate')
        assert result.returncode==0,(result.stdout,result.stderr)
        records=[json.loads(x) for x in log.read_text().splitlines()]
        assert len(records)==before+1
        assert 'AUTO_EVALUATE=0' in records[-1][records[-1].index('-v')+1]
        # Contradictory flags must fail without creating a job.
        result=run('train','--dataset','TraffiDent_Net_06','--targets','flow',
                   '--no-auto-evaluate','--eval-splits','test')
        assert result.returncode!=0
        assert len(log.read_text().splitlines())==len(records)
        # Manual Evaluate: still works as one independent qsub, no Hold.
        checkpoint=dirs[0]/'best.pt';checkpoint.write_bytes(b'fixture checkpoint')
        before=len(records)
        result=run('evaluate','--checkpoint',str(checkpoint),'--split','test',
                   '--severities','20','40')
        assert result.returncode==0,(result.stdout,result.stderr)
        records=[json.loads(x) for x in log.read_text().splitlines()]
        assert len(records)==before+1
        e=records[-1]
        assert '-W' not in e
        evars=e[e.index('-v')+1]
        assert 'EVAL_KIND=manual' in evars and 'SEVERITIES=20:40' in evars
        result=run('evaluate','--checkpoint',str(root/'missing.pt'))
        assert result.returncode!=0
        # Explicit --finalize still uses a separately scheduled OOF/calibration
        # pipeline. Normal Train->Candidate Evaluate does not.
        log.write_text('')
        result=run('train','--dataset','TraffiDent_Net_01','--targets','speed','--finalize')
        assert result.returncode==0,(result.stdout,result.stderr)
        records=[json.loads(x) for x in log.read_text().splitlines()]
        assert [Path(x[-1]).name for x in records]==[
            'train_formal.pbs','oof_controller.pbs','calibrate.pbs','evaluate.pbs'],records
        assert '-W' not in records[0]
        assert records[1][records[1].index('-W')+1]=='depend=afterok:10008.gadi-pbs'
        assert records[2][records[2].index('-W')+1]=='depend=afterok:10009.gadi-pbs'
        assert records[3][records[3].index('-W')+1]=='depend=afterok:10010.gadi-pbs'
        print('FARE_GADI_SUBMIT_TEST_PASS: one-job per target; pilot val/test; independent manual; optional finalize chain')

if __name__=='__main__':run_test()

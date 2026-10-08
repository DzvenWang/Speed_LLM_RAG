"""Gadi queue orchestration tested using a fake qsub command.

Verifies job dependencies; does not claim real PBS/Gadi execution.
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
        result = subprocess.run([sys.executable,str(runner),'train',
                                 '--dataset','TraffiDent_Net_06',
                                 '--targets','speed','flow','--seed','10'],
                                capture_output=True,text=True,env=env)
        assert result.returncode==0, (result.stdout,result.stderr)
        records = [json.loads(x) for x in log.read_text().splitlines()]
        assert len(records)==4,records
        for i,target_name in enumerate(('speed','flow')):
            tr,ev=records[i*2:i*2+2]
            assert tr[-1].endswith('/train_formal.pbs')
            assert ev[-1].endswith('/evaluate.pbs')
            assert '-W' in ev and ev[ev.index('-W')+1]==f'depend=afterok:{10001+i*2}.gadi-pbs'
            assert 'AUTO_EVALUATE=0' in tr[tr.index('-v')+1]
            assert 'CHECKPOINT_KIND=candidate' in ev[ev.index('-v')+1]
            assert f'TARGET={target_name}' in ev[ev.index('-v')+1]
        assert 'FARE_GADI_SUBMISSION_PASS targets=speed,flow' in result.stdout
        run_dirs=list((root/'FARE_exps').glob('*/TraffiDent_Net_06/formal/*'))
        assert len(run_dirs)==2
        assert all((x/'gadi_jobs.json').exists() for x in run_dirs)
        # Existing best.pt may be evaluated manually, without modifying auto results.
        checkpoint=run_dirs[0]/'best.pt'; checkpoint.write_bytes(b'fixture checkpoint')
        before=len(records)
        result=subprocess.run([sys.executable,str(runner),'evaluate',
                              '--checkpoint',str(checkpoint),'--split','test',
                              '--severities','20','40'],env=env,capture_output=True,text=True)
        assert result.returncode==0,(result.stdout,result.stderr)
        records=[json.loads(x) for x in log.read_text().splitlines()]
        assert len(records)==before+1
        e=records[-1]
        assert '-W' not in e
        evars=e[e.index('-v')+1]
        assert 'EVAL_KIND=manual' in evars and 'SEVERITIES=20:40' in evars
        # Optional finalized flow: training -> OOF -> calibration -> final test
        log.write_text('')
        result=subprocess.run([sys.executable,str(runner),'train',
                               '--dataset','TraffiDent_Net_01',
                               '--targets','speed','--finalize'],
                              env=env,capture_output=True,text=True)
        assert result.returncode==0,(result.stdout,result.stderr)
        records=[json.loads(x) for x in log.read_text().splitlines()]
        assert [Path(x[-1]).name for x in records]==[
            'train_formal.pbs','evaluate.pbs','oof_controller.pbs',
            'calibrate.pbs','evaluate.pbs'], records
        assert records[2][records[2].index('-W')+1]=='depend=afterok:10006.gadi-pbs'
        assert records[3][records[3].index('-W')+1]=='depend=afterok:10008.gadi-pbs'
        assert records[4][records[4].index('-W')+1]=='depend=afterok:10009.gadi-pbs'
        # Refuse to queue manual evaluations without a real checkpoint.
        result=subprocess.run([sys.executable,str(runner),'evaluate',
                               '--checkpoint',str(root/'missing.pt')],
                              env=env,capture_output=True,text=True)
        assert result.returncode!=0
        print('FARE_GADI_SUBMIT_TEST_PASS: both-target afterok, manual, final chain, missing checkpoint')


if __name__=='__main__':
    run_test()

"""Execute PBS shell wrappers locally with a fake prediction CLI, not Gadi."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import shutil

PBS_DIR=Path(__file__).resolve().parents[1]/'pbs'
FAKE_CLI='''import json,os,sys
from pathlib import Path
out=Path(os.environ['FARE_FAKE_PBS_MARKER'])
with out.open('a') as f:
 f.write(json.dumps(sys.argv[1:])+'\\n')
print('FARE_EVALUATE_PASS' if Path(sys.argv[0]).name=='evaluate.py' else 'FARE_TRAIN_PASS')
'''


def test_run():
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp)
        scripts=root/'FARE'/'scripts';scripts.mkdir(parents=True)
        (root/'FARE'/'pbs').mkdir(parents=True)
        shutil.copy2(PBS_DIR/'_activate_tfp.sh',root/'FARE'/'pbs'/'_activate_tfp.sh')
        for filename in ('evaluate.py','train.py'):
            (scripts/filename).write_text(FAKE_CLI)
        home=root/'fakehome';home.mkdir()
        (home/'.bashrc').write_text(': "${BASHRCSOURCED?do not source rc under PBS strict mode}"\n')
        bootstrap=root/'mock_conda.sh'
        bootstrap.write_text('conda() { if [ "$1" = activate ]; then export CONDA_DEFAULT_ENV="$2"; fi; }\n')
        run=root/'FARE_exps'/'speed'/'TraffiDent_Net_06'/'formal'/'test_seed10'
        run.mkdir(parents=True)
        ck=run/'candidate_best.pt';ck.write_bytes(b'fake')
        (run/'best.pt').write_bytes(b'fake')
        marker=root/'calls.jsonl'
        env=dict(os.environ, HOME=str(home), SPEED_LLM_RAG_ROOT=str(root),
                 FARE_FAKE_PBS_MARKER=str(marker), FARE_CONDA_SH=str(bootstrap), PBS_JOBID='9876.gadi-pbs',
                 DATASET='TraffiDent_Net_06', TARGET='speed')
        e=dict(env,RUN_DIR=str(run),CHECKPOINT_KIND='candidate', EVAL_KIND='auto',
               SPLIT='test',SEVERITIES='20:40:60:90')
        result=subprocess.run(['bash',str(PBS_DIR/'evaluate.pbs')],env=e,capture_output=True,text=True)
        assert result.returncode==0,(result.stdout,result.stderr)
        args=json.loads(marker.read_text().splitlines()[-1])
        assert args[args.index('--checkpoint')+1]==str(ck)
        assert args[args.index('--output')+1]==str(run/'evaluation_candidate')
        assert args[args.index('--severities')+1:]==['20','40','60','90']
        assert 'FARE_GADI_EVALUATE_DONE' in result.stdout
        e=dict(env,CHECKPOINT=str(run/'best.pt'),EVAL_KIND='manual', SPLIT='val')
        result=subprocess.run(['bash',str(PBS_DIR/'evaluate.pbs')],env=e,capture_output=True,text=True)
        assert result.returncode==0,(result.stdout,result.stderr)
        args=json.loads(marker.read_text().splitlines()[-1])
        assert args[args.index('--split')+1]=='val'
        assert '--severities' not in args
        assert '/manual_evaluation/' in args[args.index('--output')+1]
        e=dict(env,CHECKPOINT=str(run/'missing.pt'))
        result=subprocess.run(['bash',str(PBS_DIR/'evaluate.pbs')],env=e,capture_output=True,text=True)
        assert result.returncode!=0
        e=dict(env,RUN_DIR=str(run),AUTO_EVALUATE='1')
        e['EVAL_SPLITS']='val:test'
        e['SEVERITIES']='20:40:60:90'
        result=subprocess.run(['bash',str(PBS_DIR/'train_formal.pbs')],env=e,capture_output=True,text=True)
        assert result.returncode==0,(result.stdout,result.stderr)
        args=json.loads(marker.read_text().splitlines()[-1])
        assert '--auto-evaluate' in args
        assert args[args.index('--eval-splits')+1:args.index('--eval-severities')]==['val','test']
        assert args[args.index('--eval-severities')+1:]==['20','40','60','90']
        e['AUTO_EVALUATE']='0'
        result=subprocess.run(['bash',str(PBS_DIR/'train_formal.pbs')],env=e,capture_output=True,text=True)
        assert result.returncode==0,(result.stdout,result.stderr)
        args=json.loads(marker.read_text().splitlines()[-1])
        assert '--auto-evaluate' not in args
        # The Gadi user may have a `conda` executable, but no activation
        # function in the noninteractive PBS shell. Use explicit conda.sh.
        bare_home=root/'bare_home';bare_home.mkdir()
        no_rc=dict(e,HOME=str(bare_home),FARE_CONDA_SH=str(bootstrap),AUTO_EVALUATE='1')
        result=subprocess.run(['bash',str(PBS_DIR/'train_formal.pbs')],env=no_rc,capture_output=True,text=True)
        assert result.returncode==0,(result.stdout,result.stderr)
        assert 'FARE_CONDA_READY' in result.stdout
        assert 'env=TFP' in result.stdout
        print('FARE_GADI_PBS_EVALUATE_TEST_PASS: separate manual eval and one-allocation train passthrough')

if __name__=='__main__':test_run()

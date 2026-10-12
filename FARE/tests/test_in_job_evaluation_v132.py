"""Actual train.py post-training orchestration with minimal fake model/data.

Runs real train.py in subprocess and real subprocess-based evaluator handoff.
No CUDA, traffic data, or Gadi account is required.
"""
import json,os,shutil,subprocess,sys,tempfile
from pathlib import Path

SOURCE=Path(__file__).resolve().parents[1]/'scripts'/'train.py'
CONFIG=r'''
import json
from pathlib import Path

def load_config(path,dataset,target,seed,smoke):
    return {'experiment':{'dataset':dataset,'target':target,'seed':seed,
            'run_tag': 'pilot' if path and 'pilot' in str(path) else 'formal'},
            'training':{'batch_size':8}}
def config_from_checkpoint(*args,**kwargs):raise RuntimeError('unused')
def json_write(path,content):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(content,indent=2))
'''
RUNTIME=r'''
from pathlib import Path
class Memory:
    def save(self,path,*args):
        Path(path).parent.mkdir(parents=True,exist_ok=True)
        Path(path).write_text('train-only-memory')
def setup(*args,**kwargs):return object(),Memory(),object(),object(),None,{}
def make_run_dir(*args,**kwargs):raise RuntimeError('Pass --run-dir in test')
'''
TRAINER=r'''
from pathlib import Path
def seed_all(seed):pass
class FARETrainer:
    def __init__(self,*args):self.run_dir=Path(args[-2])
    def train(self,*args):
        out=self.run_dir/'candidate_best.pt'
        out.write_text('mock-checkpoint')
        print('FARE_TRAIN_MOCK_DONE',flush=True)
        return out
'''
EVALUATOR=r'''
import argparse,json,os,sys
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('--checkpoint');p.add_argument('--dataset');p.add_argument('--target')
p.add_argument('--memory');p.add_argument('--split');p.add_argument('--output')
p.add_argument('--severities',nargs='+')
a=p.parse_args()
assert Path(a.checkpoint).is_file()
assert Path(a.memory).is_file()
with Path(os.environ['FARE_TEST_EVAL_MARKER']).open('a') as f:
    f.write(json.dumps({'pbs_job_id':os.environ['PBS_JOBID'],
                        'split':a.split,'severities':a.severities,
                        'output':a.output})+'\n')
# The true evaluator always writes its per-split summary files.
output=Path(a.output);output.mkdir(parents=True,exist_ok=True)
for severity in (a.severities if a.split=='test' else [None]):
    path=output/f's{int(severity):02d}' if severity is not None else output
    path.mkdir(parents=True,exist_ok=True)
    tag=f's{int(severity):02d}_full' if severity is not None else 'val_full'
    data={'final':{'MAE':9.0,'RMSE':10.0},
          'base':{'MAE':10.0,'RMSE':11.0},
          'candidate':{'MAE':11.0,'RMSE':12.0},'rows':20}
    (path/f'{tag}_summary.json').write_text(json.dumps(data))
print('FARE_EVALUATE_MOCK '+a.split,flush=True)
if os.environ.get('FARE_TEST_FAIL_SPLIT')==a.split:
    sys.exit(3)
'''


ABLATOR=r'''
import argparse,json
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('--checkpoint');p.add_argument('--split');p.add_argument('--output')
p.add_argument('--modes',nargs='+');p.add_argument('--severities',nargs='+')
a=p.parse_args()
out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
rows=[]
for severity in (a.severities if a.split=='test' else ['']):
 for i,mode in enumerate(a.modes):
  rows.append({'split':a.split,'severity':severity,'mode':mode,
               'MAE':10+i,'RMSE':15+i})
(out/'ablation_summary.json').write_text(json.dumps(rows))
print('FARE_ABLATION_MOCK '+a.split,flush=True)
'''


def test_run():
    with tempfile.TemporaryDirectory() as directory:
        root=Path(directory);scripts=root/'FARE'/'scripts';scripts.mkdir(parents=True)
        package=root/'FARE'/'fare';package.mkdir()
        (scripts/'train.py').write_bytes(SOURCE.read_bytes())
        (scripts/'evaluate.py').write_text(EVALUATOR)
        (scripts/'report_pilot_val_test.py').write_bytes((SOURCE.parent/'report_pilot_val_test.py').read_bytes())
        (scripts/'report_retrieval_attribution.py').write_bytes((SOURCE.parent/'report_retrieval_attribution.py').read_bytes())
        (scripts/'ablate.py').write_text(ABLATOR)
        (scripts/'_bootstrap.py').write_text('')
        (scripts/'torch.py').write_text('class cuda:\n    @staticmethod\n    def is_available():return False\n')
        (package/'__init__.py').write_text('')
        (package/'config.py').write_text(CONFIG)
        (package/'runtime.py').write_text(RUNTIME)
        (package/'trainer.py').write_text(TRAINER)
        marker=root/'evaluations.jsonl'
        env=dict(os.environ,FARE_TEST_EVAL_MARKER=str(marker),PBS_JOBID='onejob.gadi-pbs')
        env['PYTHONPATH']=str(root/'FARE')
        def run(run_name,extra,fail_split=None):
            rd=root/run_name
            e=dict(env)
            if fail_split:e['FARE_TEST_FAIL_SPLIT']=fail_split
            p=subprocess.run([sys.executable,str(scripts/'train.py'),
                '--dataset','TraffiDent_Net_06','--target','speed','--seed','10',
                '--run-dir',str(rd),*extra],env=e,capture_output=True,text=True)
            return p,rd
        # Explicit Pilot: Train then Validation then Test in same PBS job.
        p,rd=run('pilot_both',['--config','pilot.yaml','--auto-evaluate',
                     '--eval-splits','val','test','--eval-severities','20','60'])
        assert p.returncode==0,(p.stdout,p.stderr)
        assert 'FARE_IN_JOB_EVALUATION_PASS completed=val:test' in p.stdout
        rows=[json.loads(x) for x in marker.read_text().splitlines()]
        assert [x['split'] for x in rows]==['val','test']
        assert all(x['pbs_job_id']=='onejob.gadi-pbs' for x in rows)
        assert rows[0]['severities'] is None
        assert rows[1]['severities']==['20','60']
        m=json.loads((rd/'posttrain_evaluation.json').read_text())
        assert m['status']=='pass' and m['completed_splits']==['val','test']
        assert m['pilot_test_exploratory'] is True
        # Default Pilot: Validation AND Test without explicit --eval-splits.
        marker.write_text('')
        p,rd=run('pilot_default',['--config','pilot.yaml','--auto-evaluate'])
        assert p.returncode==0,(p.stdout,p.stderr)
        assert [json.loads(x)['split'] for x in marker.read_text().splitlines()]==['val','test']
        summary=json.loads((rd/'pilot_val_test_metrics.json').read_text())
        assert [r['severity'] for r in summary]==['training-mask validation','s20','s40','s60','s90']
        # Default Pilot Retrieval Attribution: both Val and each Test severity.
        marker.write_text('')
        p,rd=run('pilot_with_ablations',['--config','pilot.yaml','--auto-evaluate',
                           '--auto-ablations','--eval-severities','20','60'])
        assert p.returncode==0,(p.stdout,p.stderr)
        assert 'FARE_RETRIEVAL_ATTRIBUTION_PASS splits=val:test' in p.stdout
        ab=root/'pilot_with_ablations'/'retrieval_attribution'
        val_rows=json.loads((ab/'val'/'ablation_summary.json').read_text())
        test_rows=json.loads((ab/'test'/'ablation_summary.json').read_text())
        assert len(val_rows)==5 and len(test_rows)==10
        report=(ab/'retrieval_gain_val_test.md').read_text()
        assert 's20' in report and 's60' in report and 'Test is exploratory' in report
        # Explicitly restrict auto-ablations to Validation when requested.
        marker.write_text('')
        p,rd=run('pilot_val_ablations',['--config','pilot.yaml','--auto-evaluate',
                    '--auto-ablations','--ablation-splits','val'])
        assert p.returncode==0,(p.stdout,p.stderr)
        assert (rd/'retrieval_attribution'/'val'/'ablation_summary.json').exists()
        assert not (rd/'retrieval_attribution'/'test').exists()
        # Default Formal: Test ONLY.
        marker.write_text('')
        p,rd=run('formal_default',['--auto-evaluate'])
        assert p.returncode==0,(p.stdout,p.stderr)
        assert [json.loads(x)['split'] for x in marker.read_text().splitlines()]==['test']
        # Explicit test failure: nonzero process, no misleading PASS marker.
        marker.write_text('')
        p,rd=run('pilot_failed',['--config','pilot.yaml','--auto-evaluate',
                                  '--eval-splits','val','test'],fail_split='test')
        assert p.returncode!=0,(p.stdout,p.stderr)
        assert 'FARE_IN_JOB_EVALUATION_FAIL' in p.stdout
        assert 'FARE_IN_JOB_EVALUATION_PASS' not in p.stdout
        m=json.loads((rd/'posttrain_evaluation.json').read_text())
        assert m['status']=='failed' and m['completed_splits']==['val']
        print('FARE_IN_JOB_EVALUATION_TEST_PASS: pilot val/test, defaults, same PBS job, error propagation')

if __name__=='__main__':test_run()

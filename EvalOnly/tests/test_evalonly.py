"""Run: python3 -m unittest discover -s EvalOnly/tests -v"""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parents[1]
CATALOG = HERE / 'scripts/catalog.py'
sys.path.insert(0, str(CATALOG.parent))
from catalog import discover


class EvalOnlyBatchTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        for d in ('Backbone', 'Baseline', 'Base_SupportBench/experiment_common', 'EvalOnly/pbs', 'EvalOnly/scripts'):
            (self.root/d).mkdir(parents=True, exist_ok=True)
        (self.root/'Base_SupportBench/experiment_common/metrics.py').write_text('MAPE_PROTOCOL="test"\n')
        import shutil
        for name in ('submit_all.sh','submit_one.sh'):
            shutil.copy2(HERE/name, self.root/'EvalOnly'/name)
        for name in ('submit_group.py','submit_one.py','catalog.py','run_manifest.py'):
            shutil.copy2(HERE/'scripts'/name, self.root/'EvalOnly/scripts'/name)
        for name in ('backbone_eval_only.pbs','baseline_eval_only.pbs','group_eval_only.pbs'):
            shutil.copy2(HERE/'pbs'/name, self.root/'EvalOnly/pbs'/name)
        self.bb = self.root/'Backbone_exps/speed/STLAformer/TraffiDent_Net_06/formal/20260930_seed10/best.pt'
        self.bb.parent.mkdir(parents=True)
        self.bb.write_bytes(b'fake')
        self.bl = self.root/'Baseline_exps/flow/GinAR/TraffiDent_Net_06/formal/20261001_seed10/best.pt'
        self.bl.parent.mkdir(parents=True)
        self.bl.write_bytes(b'fake')
        (self.bl.parent/'summary.json').write_text('{"status":"COMPLETE"}')

    def run_cmd(self, script, *args, env=None):
        return subprocess.run(['bash', str(self.root/'EvalOnly'/script), *args, '--root',str(self.root)],
                              cwd=self.root, capture_output=True, text=True, env=env)

    def test_catalog_complete_incomplete_native(self):
        ckpt = self.root/'Backbone_exps/flow/STLAformer/TraffiDent_Net_06/formal_complete/20261009_seed10/best.pt'
        ckpt.parent.mkdir(parents=True);ckpt.write_bytes(b'fake')
        bb, _ = discover(self.root,'backbone',10)
        bl, _ = discover(self.root,'baseline',10)
        self.assertEqual({i['mode'] for i in bb}, {'incomplete','complete'})
        self.assertEqual([i['method'] for i in bl], ['GinAR'])
        self.assertEqual(bl[0]['backbone'], '')

    def test_group_submission_exactly_two_qsubs(self):
        fakebin = self.root/'bin';fakebin.mkdir()
        logfile = self.root/'qsub-args.jsonl'
        qsub = fakebin/'qsub'
        qsub.write_text('#!/usr/bin/env python3\nimport json,sys\nwith open('+repr(str(logfile))+',"a") as f:f.write(json.dumps(sys.argv[1:])+"\\n")\nprint("mock-job")\n')
        qsub.chmod(0o755)
        env = os.environ.copy()
        env['PATH']=str(fakebin)+os.pathsep+env['PATH']
        out = self.run_cmd('submit_all.sh','--submit-all',env=env)
        self.assertEqual(out.returncode, 0, out.stderr+out.stdout)
        calls=[json.loads(x) for x in logfile.read_text().splitlines()]
        self.assertEqual(len(calls),2)
        for call in calls:
            self.assertIn('group_eval_only.pbs', call[-1])
            self.assertIn('SPEED_LLM_RAG_ROOT=',call[call.index('-v')+1])
        paths=list((self.root/'EvalOnly/runs').rglob('*.jsonl'))
        self.assertEqual(len(paths),2)

    def test_single_checkpoint_both_projects(self):
        r = self.run_cmd('submit_one.sh','--project','backbone','--model','STLAformer','--target','speed','--checkpoint',str(self.bb))
        self.assertEqual(r.returncode,0,r.stderr)
        self.assertIn('DRY RUN',r.stdout)
        r = self.run_cmd('submit_one.sh','--project','baseline','--method','GinAR','--target','flow','--checkpoint',str(self.bl))
        self.assertEqual(r.returncode,0,r.stderr)
        self.assertIn('FORMAL_RUN_DIR=',r.stdout)

    def test_runner_continue_and_resume(self):
        import shutil
        pbs=self.root/'EvalOnly/pbs/backbone_eval_only.pbs'
        pbs.write_text('#!/bin/bash\nif [[ "$TARGET" == "speed" && ! -f "$SPEED_LLM_RAG_ROOT/enable_speed" ]]; then exit 4; fi\necho PASS\n')
        second=self.root/'Backbone_exps/flow/STLAformer/TraffiDent_Net_06/formal_complete/20261009_seed10/best.pt'
        second.parent.mkdir(parents=True);second.write_bytes(b'fake')
        items,_=discover(self.root,'backbone',10)
        manifest=self.root/'jobs.jsonl'
        manifest.write_text(''.join(json.dumps(x)+'\n' for x in items))
        runner=self.root/'EvalOnly/scripts/run_manifest.py'
        cmd=[sys.executable,str(runner),'--kind','backbone','--manifest',str(manifest),'--root',str(self.root)]
        r1=subprocess.run(cmd,capture_output=True,text=True)
        self.assertEqual(r1.returncode,1)
        self.assertIn('EVAL_BATCH_ITEM_PASS',r1.stdout)
        self.assertIn('EVAL_BATCH_ITEM_FAIL',r1.stdout)
        (self.root/'enable_speed').touch()
        r2=subprocess.run(cmd,capture_output=True,text=True)
        self.assertEqual(r2.returncode,0,r2.stderr)
        self.assertIn('EVAL_BATCH_ALREADY_DONE',r2.stdout)
        self.assertIn('passed_new=1 passed_previous=1',r2.stdout)

if __name__=='__main__': unittest.main()

"""Synthetic contract tests; no TraffiDent data or official vendor repo required."""
from __future__ import annotations
import json
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT.parent / 'Base_SupportBench')]
from backbone.data import CachedSplit
from backbone.training_modes import (
    mode_from_checkpoint, run_stage_directory, evaluation_mode_subdir,
)
from backbone import model_runner, runner


class TinyAdapter:
    def __init__(self):
        self.model = torch.nn.Linear(1, 1)
        self.adapter_revision = 'test_adapter'
        self.adapter_metadata = {}
    def train_forward(self, batch, step):
        return self.eval_forward(batch)
    def eval_forward(self, batch):
        x = batch['x'][:, -1:, :].float()
        n = batch['y_raw'].shape[1]
        return self.model(x.unsqueeze(-1)).squeeze(-1).repeat(1, n, 1)
    def raw(self, prediction):
        return prediction


def write_split(out, split):
    M, H, P, N = 4, 3, 2, 2
    full = np.tile(np.asarray([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]],np.float32), (M,1,1))
    obs = np.ones((M,H,N), dtype=np.uint8)
    artificial = np.zeros_like(obs)
    artificial[:, 2, 0] = 1
    obs[:, 2, 0] = 0
    # A genuinely invalid source observation must remain missing in both modes.
    obs[:, 0, 1] = 0
    full[:, 0, 1] = np.nan
    x_missing = np.where(obs, full, 0).astype(np.float32)
    data = {
        'x_obs_norm': x_missing, 'x_full_norm': full,
        'history_observed': obs, 'artificial_missing': artificial,
        'y_raw': np.ones((M,P,N),np.float32)*4.,
        'y_norm': np.ones((M,P,N),np.float32)*4.,
        'future_valid': np.ones((M,P,N),np.uint8),
        'direct_mask': np.tile([1,0],(M,1)).astype(np.uint8),
        'tod_hist': np.tile([1,2,3],(M,1)).astype(np.int16),
        'dow_hist': np.zeros((M,H),np.int8),
        'tod_future': np.tile([4,5],(M,1)).astype(np.int16),
        'dow_future': np.zeros((M,P),np.int8),
    }
    for name, a in data.items(): np.save(out/f'{split}_{name}.npy',a)
    pd.DataFrame({
        'view_id':[f'{split}_{i}' for i in range(M)],
        'view_group_id':[f'g_{split}_{i}' for i in range(M)],
        'family':['sensor_outage']*M,
        'severity':[0.4]*M,
    }).to_csv(out/f'{split}_manifest.csv',index=False)


class DualModeTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path=Path(self.tmp.name)
        for split in ('train','val','test'): write_split(self.path,split)

    def test_input_modes_preserve_targets_but_change_histories(self):
        incomplete=CachedSplit(self.path,'train')
        complete=CachedSplit(self.path,'train',input_mode='complete')
        a,b=incomplete[0],complete[0]
        self.assertEqual(len(incomplete),len(complete))
        self.assertEqual(float(a['x'][-1,0]), 0.)
        self.assertEqual(float(b['x'][-1,0]), 5.)
        self.assertEqual(float(a['x'][0,1]), 0.)
        self.assertEqual(float(b['x'][0,1]), 0.)
        self.assertTrue(torch.equal(a['y_raw'],b['y_raw']))
        self.assertTrue(torch.equal(a['future_valid'],b['future_valid']))
        self.assertEqual(float(b['direct_mask'][0]),1.)
        self.assertTrue(torch.isfinite(b['x']).all())
        with self.assertRaises(ValueError): CachedSplit(self.path,'train',input_mode='not_a_mode')

    def test_complete_detects_bad_valid_raw_input(self):
        a=np.load(self.path/'train_x_full_norm.npy')
        a[0,-1,0]=np.nan
        np.save(self.path/'train_x_full_norm.npy',a)
        with self.assertRaisesRegex(ValueError,'Nonfinite valid'):
            CachedSplit(self.path,'train',input_mode='complete')[0]

    def test_checkpoint_compatibility_and_paths(self):
        self.assertEqual(mode_from_checkpoint({'model':'STLAformer'}),'incomplete')
        self.assertEqual(mode_from_checkpoint({'train_data_mode':'complete'}),'complete')
        self.assertEqual(run_stage_directory('formal','incomplete'), 'formal')
        self.assertEqual(run_stage_directory('formal','complete'), 'formal_complete')
        self.assertEqual(run_stage_directory('formal','complete',True), 'formal_complete_smoke')
        self.assertEqual(evaluation_mode_subdir('incomplete'),())
        self.assertEqual(evaluation_mode_subdir('complete'),('complete',))

    def test_patchstg_smoke_both_modes(self):
        cfg={'version':'1.3.0','experiment':{'target':'flow'},
             'runtime':{'num_workers':0},'model':{'batch_size':2,'smoke_batch_size':2,
             'optimizer':'adam','learning_rate':0.001,'train_loss':'mae'}}
        with patch.object(runner,'load_cache_identity',return_value={'cache':'test'}), \
             patch.object(runner,'require_cache_version_compatibility'), \
             patch.object(runner,'build_patchstg',side_effect=lambda *args: TinyAdapter()):
            for mode in ('incomplete','complete'):
                run=self.path/f'patch_smoke_{mode}'
                runner.smoke_patchstg(cfg,self.path,run,self.path,None,None,0.,1.,10,
                    'formal',1,1,train_data_mode=mode)
                ck=torch.load(run/'smoke.pt',map_location='cpu',weights_only=False)
                self.assertEqual(ck['train_data_mode'],mode)
                self.assertEqual(json.loads((run/'smoke_summary.json').read_text())['train_data_mode'],mode)

    def test_complete_checkpoint_is_evaluated_on_incomplete_test_input(self):
        # Run the actual evaluation entrypoint using a synthetic cache and toy adapter.
        spec=importlib.util.spec_from_file_location('test_backbone_checkpoint_eval',
                                                    ROOT/'scripts'/'evaluate_checkpoint.py')
        module=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        ck_path=self.path/'best.pt'
        adapter=TinyAdapter()
        torch.save({'model':'STLAformer','target':'speed','seed':10,'num_nodes':2,
                    'train_data_mode':'complete','best_epoch':1,'model_config':{'batch_size':2},
                    'state_dict':adapter.model.state_dict(),'mean':0.,'std':1.},ck_path)
        cfg={'version':'1.3.0',
             'experiment':{'target':'speed','subgraph':'TinyNet','families':['sensor_outage'],
                           'history':3,'horizon':2},
             'project':{'exp_root':str(self.path/'experiments'),'benchmark_root':str(self.path)}}
        args=['evaluate_checkpoint.py','--checkpoint',str(ck_path),'--model','STLAformer',
              '--severity','40','--quiet-summary','--run-stamp','teststamp']
        fake_bench=SimpleNamespace(N=2,adj=None,nodes=None)
        with patch.object(module,'load_yaml',return_value=cfg), \
             patch.object(module,'cache_path',return_value=self.path), \
             patch.object(module,'SupportBench',return_value=fake_bench), \
             patch.object(module,'build_adapter',side_effect=lambda *args: TinyAdapter()), \
             patch.object(sys,'argv',args):
            module.main()
        result=(self.path/'experiments'/'STLAformer'/'TinyNet'/'checkpoint_evaluation'/
                'complete'/self.path.name/'s40'/'teststamp_seed10')
        self.assertTrue((result/'test_summary.json').exists(), str(result))
        summary=json.loads((result/'test_summary.json').read_text())
        self.assertEqual(summary['train_data_mode'],'complete')
        self.assertEqual(summary['evaluation_input_mode'],'incomplete')
        with np.load(result/'test_predictions.npz') as prediction:
            arr=prediction['prediction'] if 'prediction' in prediction else prediction['pred']
            # The toy adapter returns linear(last_history_value); model must see masked 0, not full 5.
            with torch.no_grad():
                expected=adapter.model(torch.tensor([[[0.]]])).item()
            self.assertAlmostEqual(float(arr[0,0,0]),expected,places=5)

    def test_generic_model_trains_both_modes_and_checkpoints_differ(self):
        cfg={'version':'1.3.0','experiment':{'target':'speed'},
             'runtime':{'num_workers':0,'print_every_batches':50}}
        recipes={'models':{'STLAformer':{
            'batch_size':2,'max_epochs':1,'patience':1,
            'optimizer':'adam','learning_rate':0.001,'train_loss':'mae',
        }}}
        bench=SimpleNamespace(N=2,adj=None,nodes=None)
        with patch.object(model_runner,'load_cache_identity',return_value={'cache':'test'}), \
             patch.object(model_runner,'require_cache_version_compatibility'), \
             patch.object(model_runner,'build_adapter',side_effect=lambda *args: TinyAdapter()):
            outputs={}
            for mode in ('incomplete','complete'):
                run=self.path/f'formal_{mode}'
                outputs[mode]=model_runner.train_model(cfg,recipes,'STLAformer',self.path,run,
                    self.path,bench,0.,1.,10,train_data_mode=mode)
                ck=torch.load(run/'best.pt',map_location='cpu',weights_only=False)
                self.assertEqual(ck['train_data_mode'],mode)
                self.assertEqual(outputs[mode]['train_data_mode'],mode)
                self.assertTrue((run/'val_metrics.csv').exists())
                self.assertTrue((run/'RUN_MANIFEST.json').exists())
            self.assertNotEqual(outputs['incomplete']['validation']['MAE'],outputs['complete']['validation']['MAE'])


if __name__=='__main__':
    unittest.main()

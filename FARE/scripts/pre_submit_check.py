#!/usr/bin/env python3
"""CPU-only, fail-closed Gadi pre-submission gate for FARE v5.

Checks the **installed** source under its active Python, runs real float16
regressions at N=388 (no mock AMP bypass), validates all evaluation caches,
strictly loads the actual STLAformer vendor and checkpoint, and performs one
real vendor forward with a real cached Net-06 observation.

No GPU is required. No training/Memory is written and no qsub is submitted.
"""
import _bootstrap
import argparse
import os
import subprocess
import sys
from pathlib import Path

import torch

from fare.config import read_config, cache_dir, backbone_checkpoint, PACKAGE_ROOT
from fare.protocol import SupportDataset
from fare.backbone import load_stla


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='FARE/configs/formal.yaml')
    parser.add_argument('--dataset', default='TraffiDent_Net_06')
    parser.add_argument('--target', choices=('speed','flow'),required=True)
    parser.add_argument('--checkpoint')
    parser.add_argument('--seed', type=int, default=10)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--skip-real-forward',action='store_true',
                        help='Only strict checkpoint load, without one real STLAformer CPU forward')
    args=parser.parse_args()
    cfg=read_config(args.config,args.dataset,args.target,args.seed,args.smoke)
    print('FARE5_PRE_SUBMIT_ENV '+str({'python':sys.executable,'env':sys.prefix,
                                         'torch':torch.__version__,'target':args.target}),flush=True)
    env=os.environ.copy();env.update({'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'})
    env['PYTHONPATH']=os.pathsep.join((str(PACKAGE_ROOT),str(PACKAGE_ROOT.parent/'Backbone'),
                                       str(PACKAGE_ROOT.parent/'Base_SupportBench'),
                                       env.get('PYTHONPATH','')))
    print('FARE5_CPU_TEST_BEGIN',flush=True)
    subprocess.run([sys.executable,'-m','pytest','-q',str(PACKAGE_ROOT/'tests')],
                   check=True,cwd=PACKAGE_ROOT,env=env)
    print('FARE5_CPU_TEST_PASS',flush=True)
    train_cache=cache_dir(cfg)
    for severity in cfg['experiment']['severities']:
        test_cache=cache_dir(cfg,'evaluation',severity)
        if not (test_cache/'test_y_norm.npy').is_file():
            raise RuntimeError(f'Missing s{severity} test data: {test_cache}')
    ds=SupportDataset(train_cache,'train',1,strategy='spread')
    source=backbone_checkpoint(cfg,args.checkpoint)
    print('FARE5_PRE_SUBMIT_VENDOR_STRICT_LOAD_BEGIN checkpoint='+str(source),flush=True)
    backbone=load_stla(source,'cpu',cfg['backbone']['inject_layers'])
    if backbone.n!=ds.n:
        raise RuntimeError(f'STLAformer checkpoint nodes {backbone.n} != cache nodes {ds.n}')
    print('FARE5_PRE_SUBMIT_VENDOR_STRICT_LOAD_PASS nodes='+str(backbone.n),flush=True)
    if not args.skip_real_forward:
        ex=ds[0]
        with torch.inference_mode():
            base=backbone(ex['x'][None],ex['mask'][None],ex['tod'][None],ex['dow'][None])
        pred=base['prediction']
        expected=(1,ds.p,ds.n)
        if tuple(pred.shape)!=expected or not torch.isfinite(pred).all():
            raise RuntimeError(f'Unexpected or nonfinite real vendor prediction: {tuple(pred.shape)}')
        print('FARE5_PRE_SUBMIT_REAL_VENDOR_FORWARD_PASS',flush=True)
    print('FARE5_PRE_SUBMIT_CHECK_PASS',flush=True)


if __name__=='__main__':
    main()

#!/usr/bin/env python3
"""v9.3 source, model and Mask preflight without running any expensive teacher."""
import sys,os,json,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,torch,yaml
from tcrr_stla.config import load_config,cache_path,resolve_stla_checkpoint
from tcrr_stla.v93.interventions import benchmark_directory,plan_digest

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--target',required=True,choices=('speed','flow'))
    p.add_argument('--dataset',default='TraffiDent_Net_06')
    p.add_argument('--backbone-checkpoint',required=True)
    p.add_argument('--config',default=str(ROOT/'configs/tcrr_stla_v93.yaml'))
    p.add_argument('--require-gpu',action='store_true')
    a=p.parse_args();settings=yaml.safe_load(Path(a.config).read_text())
    if settings['version']!='9.3.0':raise RuntimeError('Wrong v9.3 config')
    os.environ['TRAFFIDENT_TARGET']=a.target
    cfg=load_config(ROOT/'configs/tcrr_stla_v5.yaml');cfg['experiment'].update(target=a.target,dataset=a.dataset)
    ck=resolve_stla_checkpoint(cfg,a.backbone_checkpoint)
    cache=cache_path(cfg,'formal')
    needed=('train_x_full_norm.npy','train_x_obs_norm.npy','train_history_observed.npy',
            'train_artificial_missing.npy','train_y_norm.npy','train_future_valid.npy',
            'val_x_obs_norm.npy','val_history_observed.npy','val_y_norm.npy')
    missing=[v for v in needed if not (cache/v).is_file()]
    if missing:raise FileNotFoundError('v9.3 cache files missing: '+','.join(missing))
    x=np.load(cache/'train_x_full_norm.npy',mmap_mode='r')
    obs=np.load(cache/'train_history_observed.npy',mmap_mode='r')
    art=np.load(cache/'train_artificial_missing.npy',mmap_mode='r')
    y=np.load(cache/'train_y_norm.npy',mmap_mode='r')
    if x.ndim!=3 or x.shape!=obs.shape or x.shape!=art.shape or len(y)!=len(x):
        raise RuntimeError('Train/cache node and row mismatch')
    if x.shape[1]!=12 or y.shape[1]!=12 or y.shape[2]!=x.shape[2]:
        raise RuntimeError('History/horizon/nodes mismatch')
    if a.require_gpu and not torch.cuda.is_available():raise RuntimeError('CUDA required')
    vendor=Path(cfg['project']['root'])/'Backbone/vendor/STLAformer/model/STLformer.py'
    if not vendor.is_file():raise FileNotFoundError(vendor)
    model=torch.load(ck,map_location='cpu',weights_only=False)
    if model.get('model')!='STLAformer' or model.get('num_nodes')!=x.shape[-1]:
        raise RuntimeError('Backbone identity or nodes incompatible')
    if float(model['std'])<=0:raise ValueError('Backbone standard deviation invalid')
    project=cfg['project']['root']
    bench=benchmark_directory(project,a.target,a.dataset)
    plan=plan_digest(project,a.target,a.dataset,int(settings['counterfactual']['views_per_query']))
    graphmode=settings['graph']['mode']
    if graphmode!='none':
        for rel in ('core/nodes.csv','graph/adjacency.npy','graph/distance.npy'):
            if not (bench/rel).is_file():raise FileNotFoundError(bench/rel)
        c=np.load(cache/'adjacency.npy',mmap_mode='r')
        g=np.load(bench/'graph/adjacency.npy',mmap_mode='r')
        if c.shape!=(x.shape[-1],x.shape[-1]) or not np.allclose(c,g,equal_nan=True):
            raise RuntimeError('Physical graph differs from train cache')
    info={'version':'9.3.0','target':a.target,'dataset':a.dataset,
          'nodes':int(x.shape[-1]),'train_rows':int(len(x)),
          'history':int(x.shape[1]),'horizon':int(y.shape[1]),
          'graph':graphmode,'mask_plan_sha256':plan,
          'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
          'original_backbone':str(ck)}
    print('V93_PREFLIGHT_PASS '+json.dumps(info),flush=True)
if __name__=='__main__':main()

#!/usr/bin/env python3
"""Verified v7.0.1 -> v7.1.0 S0/teacher/replay cache import.

Run ONLY after v7.1 run_identity.json is created and BEFORE qsub. Does not
import old C1/C2 checkpoints. Never edits the source run, and marks a stage
complete only once validation and all file transfers have succeeded.
"""
from __future__ import annotations
import argparse,os,json,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from tcrr_stla.config import file_sha256,load_config,cache_path
from tcrr_stla.data import SupportSplit
from tcrr_stla.v7.storage import file_manifest,open_existing_teacher
from tcrr_stla.v7.version import VERSION
from tcrr_stla.v7.replay import KEYS


def read_json(p):
    if not Path(p).is_file():raise RuntimeError(f'Missing identity: {p}')
    return json.loads(Path(p).read_text())


def check_equal(lhs,rhs,label):
    if lhs!=rhs:raise RuntimeError(f'Cache import identity differs: {label}: {lhs!r} vs {rhs!r}')


def verify(source,dest,checkpoint,target,dataset,seed):
    source=Path(source).resolve();dest=Path(dest).resolve()
    if source==dest or source in dest.parents or dest in source.parents:
        raise RuntimeError('Source and destination must be distinct independent experiment directories')
    new=read_json(dest/'run_identity.json');old=read_json(source/'run_identity.json')
    check_equal(new['version'],VERSION,'new version')
    if old['version'] not in ('7.0.1','7.1.0'):
        raise RuntimeError('Only v7.0.1/v7.1.0 source caches supported')
    if old.get('smoke') or new.get('smoke'):raise RuntimeError('Never mix smoke/formal caches')
    for key,expected in (('target',target),('dataset',dataset),('seed',seed)):
        check_equal(old[key],expected,'old '+key);check_equal(new[key],expected,'new '+key)
    checkpoint_sha=file_sha256(checkpoint)
    check_equal(old['backbone_sha256'],checkpoint_sha,'old Backbone SHA')
    check_equal(new['backbone_sha256'],checkpoint_sha,'new Backbone SHA')
    for key in ('limits','groups'):
        check_equal(old[key],new[key],key)
    runtime_keys=('retrieval_periods','retrieval_shortlist','retrieval_history_weight',
                  'retrieval_forecast_weight','replay_chunk','k','road_k','graph_mode')
    for key in runtime_keys:
        check_equal(old['effective_runtime'][key],new['effective_runtime'][key],key)
    # S0 identity is byte-level immutable; this preserves all v7.0.1 teacher keys.
    s0=source/'s0.pt'
    if not (source/'STAGE_s0.done').is_file() or not s0.is_file():raise RuntimeError('Incomplete S0 source stage')
    saved=torch.load(s0,map_location='cpu',weights_only=False)
    check_equal(saved['backbone_sha256'],checkpoint_sha,'S0 checkpoint')
    if 'mask_proj' not in saved:raise RuntimeError('S0 lacks mask projection')
    ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
    n=int(ck['num_nodes']);c=ck['model_config'];dim=int(c.get('input_embedding_dim',32))+int(c.get('tow_embedding_dim',32))
    experiment_root=next((q.parent for q in dest.parents if q.name=='TCRR-STLA_exps'),None)
    if experiment_root is None:raise RuntimeError('Destination must be under TCRR-STLA_exps')
    os.environ['SPEED_LLM_RAG_ROOT']=str(experiment_root)
    os.environ['TRAFFIDENT_TARGET']=target
    cfg=load_config(ROOT/'configs/tcrr_stla_v7.yaml')
    cfg['experiment'].update(target=target,dataset=dataset)
    db=cache_path(cfg,'formal')
    datas={key:SupportSplit(db,key,limit=new['limits'].get(key)) for key in ('train','val')}
    groups=int(new['groups'])
    teacher_key=file_sha256(s0)
    for split,ds in datas.items():
        src_dir=source/'teacher_cache'/split
        ident=file_manifest(ds,split,teacher_key,groups)
        result=open_existing_teacher(split,src_dir,ident,len(ds),n,12,dim)
        if result is None:raise RuntimeError(f'Source teacher cache is incomplete: {src_dir}')
    graph=source/'train_only_neighbours.pt'
    if not graph.exists():raise RuntimeError('Missing functional graph')
    saved_graph=torch.load(graph,map_location='cpu',weights_only=False)
    expected_graph={'file_sha256':file_sha256(db/'train_x_full_norm.npy'),
        'queries':len(datas['train']),'neighbors':int(new['configuration']['model']['graph_neighbors']),
        'nodes':n,'graph_algorithm_sha256':file_sha256(ROOT/'tcrr_stla/v7/model.py')}
    check_equal(saved_graph.get('identity'),expected_graph,'functional graph metadata')
    # Road graph checked separately by graph.load_or_build_graph at stage entry.
    config={'schema':'v7_replay_v1',
            'implementation_sha256':file_sha256(ROOT/'tcrr_stla/v7/replay.py'),
            'train_source':file_manifest(datas['train'],'train',teacher_key,groups),
            'graph_sha256':file_sha256(graph),'s0_sha256':teacher_key,
            'checkpoint_sha256':checkpoint_sha,
            'k':int(new['effective_runtime']['k']),
            'periods':list(new['effective_runtime']['retrieval_periods']),
            'shortlist':int(new['effective_runtime']['retrieval_shortlist']),
            'history_weight':new['effective_runtime']['retrieval_history_weight'],
            'forecast_weight':new['effective_runtime']['retrieval_forecast_weight'],
            'chunk_queries':int(new['effective_runtime']['replay_chunk']),
            'train_rows':len(datas['train']),'dataset':dataset,'target':target,'severity':None}
    checked=0
    for split,ds in datas.items():
        root=source/'paired_replay'/split
        expected={**config,'source':file_manifest(ds,split,teacher_key,groups),'rows':len(ds)}
        # Identity equality also checks all hash-bound source filenames.
        check_equal(read_json(root/'identity.json'),expected,'replay '+split)
        chunk=int(expected['chunk_queries']); k=int(expected['k'])
        from tcrr_stla.v7.replay import _hash_obj
        h=_hash_obj(expected)
        for start in range(0,len(ds),chunk):
            end=min(start+chunk,len(ds))
            folder=root/f'chunk_{start:06d}_{end:06d}'
            done=read_json(folder/'DONE.json')
            check_equal(done.get('identity'),h,f'replay chunk {split} identity')
            for name in KEYS:
                f=folder/(name+'.npy')
                if not f.is_file():raise RuntimeError(f'Incomplete replay chunk {f}')
                arr=np.load(f,mmap_mode='r',allow_pickle=False)
                shape=((end-start,k,12,n) if name in ('normal','cf','valid') else
                       (end-start,k,n,14) if name=='support_raw' else (end-start,k))
                check_equal(arr.shape,shape,str(f)+' shape')
                check_equal(file_sha256(f),done['hashes'][name],str(f)+' hash')
            checked+=1
    for stage in ('cache','replay'):
        if not (source/f'STAGE_{stage}.done').exists():
            raise RuntimeError(f'Old stage not complete: {stage}')
    return {'source':source,'dest':dest,'checked_chunks':checked}


def link_or_copy(src,dst):
    dst.parent.mkdir(parents=True,exist_ok=True)
    if dst.exists():
        if file_sha256(src)!=file_sha256(dst):raise RuntimeError(f'Existing destination differs: {dst}')
        return
    try:os.link(src,dst)
    except OSError:shutil.copy2(src,dst)


def main():
    a=argparse.ArgumentParser()
    a.add_argument('--source',required=True)
    a.add_argument('--destination',required=True)
    a.add_argument('--target',required=True,choices=('speed','flow'))
    a.add_argument('--dataset',required=True)
    a.add_argument('--seed',type=int,required=True)
    a.add_argument('--backbone-checkpoint',required=True)
    v=a.parse_args()
    info=verify(v.source,v.destination,v.backbone_checkpoint,v.target,v.dataset,v.seed)
    src=info['source'];dst=info['dest']
    roots=('s0.pt','train_only_neighbours.pt')
    folders=('teacher_cache','paired_replay','retrieval_keys','graph')
    for name in roots:
        link_or_copy(src/name,dst/name)
    for name in folders:
        if not (src/name).is_dir():
            if name in ('graph', 'retrieval_keys'):continue
            raise RuntimeError('Required cache folder not found: '+str(src/name))
        for f in (src/name).rglob('*'):
            if f.is_file() and not f.name.endswith(('.partial','.partial.npy')):
                link_or_copy(f,dst/f.relative_to(src))
    # Stage markers come last: if interrupted, this script is safe to re-run.
    for stage in ('s0','cache','replay'):
        marker=dst/f'STAGE_{stage}.done'
        marker.write_text('verified cache import from '+str(src)+'\n')
    print('V71_CACHE_IMPORT_PASS '+json.dumps({'source':str(src),'destination':str(dst),
                            'verified_chunks':info['checked_chunks']}),flush=True)
if __name__=='__main__':main()

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from tcrr_stla.v9.prepare import prepare_normal,prepare_view,V9EvidenceDataset

class FakeTeacher(torch.nn.Module):
    model_dim=8
    def forward(self,x,m,tod,dow):
        z=(x*m).sum(1)/m.sum(1).clamp_min(1)
        return z[:,None].expand(-1,12,-1),z.unsqueeze(-1).expand(-1,-1,8)

def fixture(tmp):
    r=tmp/'source';r.mkdir()
    rng=np.random.default_rng(10);n=4;L=12;H=12
    for split,rows,offset in [('train',40,0),('val',6,100)]:
        x=rng.normal(size=(rows,L,n)).astype('float32')
        y=rng.normal(size=(rows,H,n)).astype('float32')
        m=np.ones_like(x,dtype=bool);m[:,1::4,0]=False
        original=np.logical_not(m)  # source validity = observed OR original = all 1
        fields=dict(x_full_norm=x,x_obs_norm=x*m,history_observed=m,
                    artificial_missing=original,y_norm=y,y_raw=y,
                    future_valid=np.ones_like(y,bool),direct_mask=np.ones_like(y,bool),
                    tod_hist=np.zeros((rows,L),np.int64),dow_hist=np.zeros((rows,L),np.int64))
        for key,z in fields.items():np.save(r/f'{split}_{key}.npy',z)
        pd.DataFrame({'window_start':np.arange(rows)*24+offset*24+100}).to_csv(r/f'{split}_manifest.csv',index=False)
    return r

def test_full_prepare_and_data_loader(tmp_path):
    r=fixture(tmp_path);out=tmp_path/'v9_evidence';out.mkdir()
    teacher=FakeTeacher();device=torch.device('cpu')
    prepare_normal(r,teacher,device,out,limit=40,chunk=6)
    k=2
    ids=np.zeros((40,k),dtype=np.int32)
    ids[:,0]=np.arange(40)//2
    ids[:,1]=np.maximum(0,np.arange(40)//2-1)
    # Deliberately future candidate on first query -> invalid, not used by network
    rt=tmp_path/'retrieval_train.npz'
    np.savez(rt,version=25,indices=ids,scores=np.zeros((40,k),'float32'),
             coverage=np.ones((40,k),'float32'))
    for v in (0,1):
        prepare_view(r,'train',v,teacher,device,out,rt,k=k,chunk=3,limit=12,seed=10)
    ds=V9EvidenceDataset(r,'train',(0,1),root=out)
    assert len(ds)==24
    a,b=ds[4],ds[5]
    assert int(a['query_id'])==int(b['query_id'])==2
    assert a['cf'].shape==(k,12,4)
    assert a['teacher_cf_training_only'].shape==(12,4)
    assert not torch.equal(a['mask'],b['mask'])
    assert ds[0]['valid'].sum()==0
    # Exact v9 reload from saved manifest
    assert ds[4]['candidate_ids'].shape==(k,)
    assert 'normal_train.npy' in {p.name for p in out.glob('*.npy')}

def test_paired_view_chunk_resume_without_replay_restart(tmp_path):
    import json,os
    from tcrr_stla.v9.prepare import _atomic_json
    r=fixture(tmp_path);out=tmp_path/'v9_evidence';out.mkdir()
    teacher=FakeTeacher();dev=torch.device('cpu')
    prepare_normal(r,teacher,dev,out,limit=40,chunk=8)
    rt=tmp_path/'cache.npz'
    ids=np.zeros((40,2),dtype='int32')
    np.savez(rt,version=25,indices=ids,scores=np.zeros((40,2),'float32'),
             coverage=np.ones((40,2),'float32'))
    prepare_view(r,'train',0,teacher,dev,out,rt,k=2,chunk=3,limit=12)
    prepare_view(r,'train',1,teacher,dev,out,rt,k=2,chunk=3,limit=12)
    folder=out/'train_view1';manifest=folder/'manifest.json'
    original=np.array(np.load(folder/'cf.npy',mmap_mode='r')[0],copy=True)
    shapes={}
    for name in ('base','context','cf','candidate_ids','scores','valid','match','mask','x','teacher_cf','family','severity','mask_hash'):
        path=folder/(name+'.npy')
        a=np.load(path,mmap_mode='r')
        shapes[name]=list(a.shape)
        os.replace(path,folder/(name+'.partial.npy'))
    manifest.unlink()
    _atomic_json(folder/'progress.json',dict(next_row=6,shapes=shapes))
    prepare_view(r,'train',1,teacher,dev,out,rt,k=2,chunk=3,limit=12)
    assert manifest.is_file()
    np.testing.assert_array_equal(original,np.load(folder/'cf.npy',mmap_mode='r')[0])
    assert not (folder/'progress.json').exists()

"""Train memory, Pearson index with exclusion, failure replay, banks and provider."""
from __future__ import annotations
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
import pytest
import torch
from helpers import L,P,N,FakeSTLA,QueryDataset,write_train_cache
from reshift.retrieval.memory import TrainMemory
from reshift.retrieval.index import MemoryIndex,legal_wrong_indices
from reshift.retrieval.bank import build_bank,replay,ReplayBank
from reshift.retrieval.provider import EvidenceProvider

torch.set_num_threads(1)
CPU=torch.device('cpu')
BANK={'top_k':3,'periods':[3],'min_valid_features':2,'train_embargo_steps':0,'mask_slots':1,
      'query_batch':4,'replay_chunk':8,'workers':0,'pfull_batch':4,'memory_chunk':5}


def test_memory_rows_never_read_unobserved(tmp_path):
    root,a=write_train_cache(tmp_path)
    mem=TrainMemory(root)
    r=mem.rows(np.arange(len(mem)))
    ov=a['history_observed']|a['artificial_missing']
    assert np.array_equal(r['ov'],ov)
    assert np.all(r['x'][~ov]==0) and np.isfinite(r['x']).all()
    with pytest.raises(IndexError):mem.rows([len(mem)])
    f=mem.fetch(torch.tensor([[0,-1]]),CPU,keys=('x','y'))
    assert f['x'].shape==(1,2,L,N) and torch.all(f['x'][0,1]==0) and torch.all(f['y'][0,1]==0)


def test_index_exclusion_padding_and_self_match(tmp_path):
    root,a=write_train_cache(tmp_path,count=12,spacing=12)
    mem=TrainMemory(root);index=MemoryIndex(mem,[3],CPU,chunk=5)
    r=mem.rows(np.arange(len(mem)))
    # Query identical to memory row 7 but far away in time: row 7 must rank first.
    idx,sim=index.query(torch.from_numpy(r['x'][7:8]),torch.from_numpy(r['ov'][7:8]).float(),
                        torch.tensor([10_000]),top_k=3,span=12,min_valid=2)
    assert int(idx[0,0])==7 and float(sim[0,0])>0.99
    # Query at the same time as row 5: rows 4,5,6 overlap (|ds|<24) and are excluded.
    idx,_=index.query(torch.from_numpy(r['x'][5:6]),torch.from_numpy(r['ov'][5:6]).float(),
                      torch.tensor([60]),top_k=12,span=24,min_valid=2)
    got=set(idx[0][idx[0]>=0].tolist())
    assert not got&{4,5,6} and len(got)==9 and int((idx[0]<0).sum())==3
    # Embargo widens the exclusion.
    idx,_=index.query(torch.from_numpy(r['x'][5:6]),torch.from_numpy(r['ov'][5:6]).float(),
                      torch.tensor([60]),top_k=12,span=24,embargo=24,min_valid=2)
    assert not set(idx[0][idx[0]>=0].tolist())&{3,4,5,6,7}
    w=legal_wrong_indices(np.array([60]),np.ones((1,3),bool),mem.starts,24)
    assert all(abs(int(mem.starts[j])-60)>=24 for j in w[0])


def test_replay_applies_query_mask_and_neighbour_availability(tmp_path):
    root,_=write_train_cache(tmp_path)
    mem=TrainMemory(root);stla=FakeSTLA()
    qmask=(torch.rand(2,L,N)>.5).float()
    idx=torch.tensor([[1,2],[3,-1]])
    stla.seen_masks.clear()
    p0=replay(stla,mem,idx,qmask,CPU,chunk=8)
    seen=torch.cat(stla.seen_masks,0).view(2,2,L,N)
    ov=mem.fetch(idx,CPU,keys=('ov',))['ov'].float()
    assert torch.equal(seen,qmask[:,None]*ov)
    assert p0.shape==(2,2,P,N) and torch.all(p0[1,1]==0)


def test_bank_build_reuse_fingerprint_and_label_independence(tmp_path):
    root,_=write_train_cache(tmp_path/'cache')
    mem=TrainMemory(root);index=MemoryIndex(mem,[3],CPU);stla=FakeSTLA()
    g=torch.Generator().manual_seed(3)
    x=torch.randn(5,L,N,generator=g);m=(torch.rand(5,L,N,generator=g)>.3).float()
    ds=QueryDataset(x*m,m,torch.arange(5)*7+5000,y=torch.randn(5,P,N,generator=g))
    ds_other_y=QueryDataset(x*m,m,torch.arange(5)*7+5000,y=torch.randn(5,P,N,generator=g)+50)
    fp={'k':1}
    bank=build_bank(tmp_path/'bank_a',fp,ds,mem,index,stla,CPU,3,24,0,2,query_batch=2)
    other=build_bank(tmp_path/'bank_b',fp,ds_other_y,mem,index,stla,CPU,3,24,0,2,query_batch=2)
    assert bank.idx.shape==(5,3) and bank.p0.shape==(5,3,P,N)
    assert np.array_equal(np.asarray(bank.idx),np.asarray(other.idx))
    assert np.array_equal(np.asarray(bank.p0),np.asarray(other.p0))          # query labels never used
    again=build_bank(tmp_path/'bank_a',fp,ds,mem,index,stla,CPU,3,24,0,2)
    assert np.array_equal(np.asarray(again.idx),np.asarray(bank.idx))
    with pytest.raises(RuntimeError):
        build_bank(tmp_path/'bank_a',{'k':2},ds,mem,index,stla,CPU,3,24,0,2)
    rows=bank.fetch(torch.tensor([4,0]),CPU)
    assert torch.equal(rows['idx'][0],torch.from_numpy(np.asarray(bank.idx[4],dtype=np.int64)))


def test_provider_modes(tmp_path):
    root,_=write_train_cache(tmp_path/'cache')
    mem=TrainMemory(root);stla=FakeSTLA()
    prov=EvidenceProvider(mem,tmp_path/'banks',stla,BANK,CPU,{'s0_key':'x'},log=lambda s:None)
    assert mem.p_full is not None and mem.p_full.shape==(len(mem),P,N)
    m=(torch.rand(4,L,N)>.3).float();x=torch.randn(4,L,N)*m
    ds=QueryDataset(x,m,torch.arange(4)*30+9000)
    desc={'split':'val','severity':20}
    prov.ensure('val_s20',ds,desc);prov.ensure('val_s20',ds,desc,wrong=True)
    rows=torch.tensor([0,1,2,3])
    full=prov.batch('val_s20',rows,'full')
    assert full['p0'].shape==(4,3,P,N) and full['valid_k'].dtype==torch.bool
    nr=prov.batch('val_s20',rows,'no_replay')
    assert torch.allclose(nr['p0'],nr['p_full'])
    sh=prov.batch('val_s20',rows,'shuffled')
    assert torch.allclose(sh['p0'],torch.roll(full['p0'],1,dims=1)) and torch.equal(sh['sim'],full['sim'])
    wr=prov.batch('val_s20',rows,'cross_history')
    assert torch.equal(wr['sim'],full['sim'])
    with pytest.raises(RuntimeError):prov.batch('val_s40',rows)
    with pytest.raises(ValueError):prov.batch('val_s20',rows,'bogus')
    assert prov.slot_for_epoch(1)==1 and prov.slot_for_epoch(7)==1

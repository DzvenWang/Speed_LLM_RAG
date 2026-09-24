from pathlib import Path
import tempfile, numpy as np, torch
import torch.nn as nn
from online_presure.stream import run_stream

class Emb(nn.Module):
    def __init__(self,d): super().__init__(); self.lin=nn.Linear(1,d)
    def forward(self,x,te): return self.lin(x)
class Block(nn.Module):
    def __init__(self,d): super().__init__(); self.lin=nn.Linear(d,d)
    def forward(self,x): return x+0.05*self.lin(x)
class FakePatch(nn.Module):
    def __init__(self,T,N,H,D=3):
        super().__init__(); self.node_num=N; self.embedding=Emb(D); self.spa_encoder=nn.ModuleList([Block(D),Block(D)])
        self.regression_conv=nn.Conv2d(T*D,H,kernel_size=(1,1)); self.register_buffer('reo_all_idx',torch.arange(N)); self.register_buffer('ori_parts_idx',torch.arange(N)); self.register_buffer('reo_parts_idx',torch.arange(N))
    def forward(self,x,te):
        from online_presure.model_io import forward_with_prehead
        y,_=forward_with_prehead(self,x,te); return y

class FakeDS:
    def __init__(self,n=42,T=12,N=4,H=12):
        self.stage='development'; self.split='val'; self.history=T; self.horizon=H; self.rows=[]
        rng=np.random.default_rng(10)
        for i in range(n):
            base=np.sin((np.arange(T)[:,None]+i)/8)+np.linspace(0,.4,N)[None,:]
            full=(base+rng.normal(scale=.03,size=(T,N))).astype(np.float32); obs=np.ones((T,N),bool); miss=np.zeros((T,N),bool)
            miss[9:12,1]=True
            if i%2==0: miss[6:9,2]=True
            obs[miss]=False; x=full.copy(); x[miss]=0.0
            y=np.tile(full[-1][None,:],(H,1)).astype(np.float32)+np.linspace(.01,.12,H)[:,None].astype(np.float32)
            direct=np.array([0,1,1,0],bool); support=np.array([1,1,1,0],bool)
            self.rows.append({'index':i,'start':i,'origin':i+T,'future_end':i+T+H,'x_obs':torch.from_numpy(x),'x_full':torch.from_numpy(full),
                'history_observed':torch.from_numpy(obs),'artificial_missing':torch.from_numpy(miss),'full_valid':torch.ones((T,N),dtype=torch.bool),
                'y_raw':torch.from_numpy(y),'y_norm':torch.from_numpy(y),'future_valid':torch.ones((H,N),dtype=torch.bool),
                'direct_mask':torch.from_numpy(direct),'support_mask':torch.from_numpy(support),'tod_hist':torch.arange(T)%288,'dow_hist':torch.zeros(T,dtype=torch.long)})
    def __len__(self): return len(self.rows)
    def cache_index(self,p): return int(p)
    def item(self,i):
        r=self.rows[int(i)]; return {k:(v.clone() if torch.is_tensor(v) else v) for k,v in r.items()}
    def batch(self,indices):
        rr=[self.item(i) for i in indices]; keys=['x_obs','y_raw','y_norm','future_valid','direct_mask','support_mask','tod_hist','dow_hist']
        out={k:torch.stack([r[k] for r in rr],0) for k in keys}; out['indices']=np.asarray(indices,np.int64); return out

def main():
    T,N,H=12,4,12; ds=FakeDS(); model=FakePatch(T,N,H)
    cfg={'version':'3.3.0','experiment':{'horizon':H,'history':T,'seed':10},
      'adaptation':{'trainable':'last_block_head','learning_rate':1e-4,'weight_decay':0.,'anchor_lambda':0.,'replay_memory_size':16,'min_memory_cases':4,'min_mature_cases_for_update':4,'batch_size':4,'steps_per_update':1,'update_every_queries':1,'grad_clip':1.,'save_final_checkpoint':False},
      'evidence_memory':{'capacity':32,'min_cases':4},
      'retrieval':{'prefilter_k':16,'top_k':4,'latent_weight':.35,'observed_match_weight':.25,'complementarity_weight':.2,'recency_weight':.15,'context_weight':.05,'tod_weight':.7,'dow_weight':.3,'slots_per_day':288,'recency_tau_steps':32.,'observed_match_scale':.5,'min_overlap_fraction':.1,'score_temperature':.15,'nodewise':{'local_observed_match_scale':.5}},
      'compositional':{'unit':'node_temporal_patch','candidate_pool_k':16,'temporal_patch_size':3},
      'utility':{'temporal_patch_size':3,'patch_observed_match_scale':.5,'min_patch_overlap_steps':1,'candidate_consensus_scale':.5,'age_scale_steps':32.,'top_k_per_patch':1,
        'similarity_fallback_weights':{'latent':.1,'global_observed':.1,'node_observed':.15,'patch_observed':.25,'patch_complementarity':.25,'recency':.1,'context':.05},
        'hidden_dims':[16,8],'learning_rate':1e-3,'weight_decay':0.,'buffer_size':256,'min_examples':6,'min_class_examples':1,'batch_size':6,'train_steps_per_update':1,'train_every_new_labels':3,'grad_clip':5.,
        'helpful_threshold_pct':.001,'harmful_threshold_pct':-.001,'harmful_penalty':1.5,'helpful_prob_min':0.,'harmful_prob_max':1.,'utility_score_min':-9.,'abstain_when_no_support':True,'probes_per_query':3,'probe_top_fraction':.5},
      'completion':{'target':'artificial_missing_only','alignment':'node_shift','min_node_overlap_steps':1,'min_support_cases_per_entry':1,'min_weighted_coverage':.1,'consensus_scale_norm':.5,'max_gate':.2},
      'fusion':{'scope':'direct_and_support','max_external_weight':.2},'diagnostics':{'perfect_completion_oracle':True,'retrieved_completion_oracle':True,'entry_oracle_ks':[4,8,16],'retrieval_log_top_k':2,'patch_support_log_max_per_query':8}}
    with tempfile.TemporaryDirectory() as td:
        ck=Path(td)/'fake.pt'; torch.save({'x':1},ck); out=Path(td)/'out'; out.mkdir()
        s=run_stream(cfg,ds,model,ck,{'mean':0.,'std':1.},'synthetic','rag_utility_compositional',out,torch.device('cpu'),max_queries=36)
        assert s['utility']['examples']>0 and (out/'utility_labels.csv').exists() and (out/'patch_supports.csv').exists()
        import pandas as pd
        labels=pd.read_csv(out/'utility_labels.csv'); assert len(labels)>0
        # First labels can only appear when the originating query's full 12-step future has matured.
        assert np.all(labels['mature_stream_pos'].to_numpy() >= labels['query_index'].to_numpy()+H)
        print('STREAM_V33_SMOKE PASS utility_examples=',s['utility']['examples'],'classes=',s['utility']['class_counts'],'final_gain=',s['final_gains_pct']['masked_mae'])
if __name__=='__main__': main()

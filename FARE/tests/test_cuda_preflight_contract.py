"""The GPU warmup algorithm can be validated on CPU without a GPU reservation."""
import torch
import pandas as pd
from fare.cuda_preflight import run_cuda_preflight


def test_pre_memory_amp_gate_with_realistic_chronology(fixture_data):
    from conftest import DummyBackbone
    model,memory,batch=fixture_data
    n=batch['x'].shape[-1]
    class Train:
        h=12
        def __init__(self):
            self.ix=list(range(12))
            self.meta=pd.DataFrame({'window_start':torch.arange(12).numpy()*40})
        def __len__(self):return len(self.ix)
        def __getitem__(self,i):
            sample={k: v[0].clone() if torch.is_tensor(v) and v.ndim>0 else v
                    for k,v in batch.items()}
            sample['origin']=torch.tensor(i*40+11)
            sample['row_id']=torch.tensor(i)
            return sample
    cfg={'model':{'hidden_dim':8,'horizon':12,'alpha_max':1.25},
         'retrieval':{'coarse_k':4,'shortlist_k':3,'final_k':2},
         'memory':{'purge_steps':0},'training':{'masked_weight':.2,'tail_weight':.02,'support_weight':.02}}
    report=run_cuda_preflight(DummyBackbone(n),model.graph,Train(),cfg,
                              torch.device('cpu'),force_cpu=True)
    assert report['batch']==2
    assert report['nodes']==n
    assert report['gradient_tensors']>0
    assert report['loss']>=0

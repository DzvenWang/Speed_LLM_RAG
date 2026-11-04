import json
import torch
from torch.utils.data import Dataset
from pcft_stla.v6.graph import DirectedGraph
from pcft_stla.v6.pipeline import run
from conftest import FakeSTLA
from pcft_stla.v6.model import PCFTV6

class TinyDataset(Dataset):
    def __init__(self,target='flow',n=4):self.target=target;self.n=n
    def __len__(self):return self.n
    def __getitem__(self,i):
        g=torch.Generator().manual_seed(i+9)
        x=torch.randn((12,5,3),generator=g);m=torch.ones((12,5,3));x=x*m
        y=torch.randn((12,5),generator=g)
        return {'x_multi':x,'observed_mask':m,'missing_age':torch.zeros_like(x),
                'x_target':x[:,:,1], 'y_norm':y,'y_raw':y,
                'future_valid':torch.ones((12,5),dtype=torch.bool),'direct_mask':torch.ones(5,dtype=torch.bool),
                'tod':torch.arange(12),'dow':torch.zeros(12,dtype=torch.long),
                'index':torch.tensor(i)}

def test_full_pipeline_smoke(tmp_path,monkeypatch):
    import pcft_stla.v6.pipeline as pip
    ei=torch.tensor([[0,1,2,3,4],[1,2,3,4,0]])
    fi=torch.tensor([[0,2,4],[2,4,1]])
    fa=torch.tensor([[.4,.4,.3,.8,1.,0.]]*3)
    gr=DirectedGraph(ei,torch.tensor([[1.,0.,0.5,1.,0.,0.,1.,0.,1.,1.]]*5),5,fi,fa)
    fake=tmp_path/'best.pt';fake.write_bytes(b'fake')
    ident={'target':'flow','dataset':'TraffiDent_Net_06','seed':10,'mode':'incomplete',
           'channels':['speed','flow','occupancy'],'mean':0.,'std':1.}
    monkeypatch.setattr(pip,'prepare',lambda cfg,checkpoint,smoke=False: (tmp_path,tmp_path/'out',fake,gr,
                        {k:TinyDataset() for k in ('train','val','test_s20','test_s40','test_s60','test_s90')},ident))
    monkeypatch.setattr(pip,'instantiate',lambda cfg,repo,ck,graph,details,device: PCFTV6(FakeSTLA(),graph,1,max_lag=3).to(device))
    cfg={'experiment':{'target':'flow','seed':10,'dataset':'TraffiDent_Net_06','severities':[20,40,60,90]},
         'backbone':{'mode':'incomplete'},'model':{'max_lag':3,'hidden_dim':16},
         'training':{'batch_size':2,'auto_batch_probe':False,'unfreeze_last_stla_layers':1,
           'lr_encoder':2e-5,'lr_new':5e-4,'weight_decay':1e-4,'grad_clip':1.,'amp':False,
           'initial_amp_scale':1024,'min_delta':0.0001,
           'loss':{'rmse':0.05,'masked':0.1,'evolution':0.05},
           'stages':[{'name':s,'max_epochs':1,'patience':1} for s in PCFTV6.STAGES]},
         'runtime':{'eval_batch':2,'workers':0,'print_every':100,'max_pipeline_hours':2}}
    path=run(cfg,run_name='test',smoke=True)
    output=json.loads((path/'FINAL_REPORT.json').read_text())
    assert output['status']=='complete'
    assert set(output['stage_summary'])==set(PCFTV6.STAGES)
    assert len(output['test_results'])==16
    assert (path/'last.pt').exists()
    # Pipeline fully completed: resume must remain safe and reproduce final report.
    path2=run(cfg,run_name='ignored',smoke=True,resume=path/'last.pt')
    assert path2==path

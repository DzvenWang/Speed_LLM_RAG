import json
import pandas as pd
import torch
from torch.utils.data import Dataset
from test_v7_model import fake_model,fake_batch,cfg
from pcft_stla.v7.training import train_pipeline

class SampleSet(Dataset):
    def __init__(self,count):
        self.b=fake_batch(b=1)
        self.meta=pd.DataFrame({'window_start':[i*12 for i in range(count)]})
    def __len__(self):return len(self.meta)
    def __getitem__(self,index):
        o={k:v[0].clone() for k,v in self.b.items()}
        o['index']=torch.tensor(index)
        return o

def test_full_four_stage_training_on_synthetic_cpu(tmp_path):
    model=fake_model()
    c=cfg()
    c['experiment']={'target':'speed','seed':10,'severities':[20]}
    c['training'].update({'batch_size':4,'amp':False,'candidate_train_fraction':.8,
                          'temporal_embargo_steps':24,'joint_refresh_utility':True,
                          'min_delta':1e-5,'grad_clip':1.0,
                          'stages':{stage:{'max_epochs':1,'patience':1} for stage in ('local','propagation','utility','joint')}})
    c['runtime']={'eval_batch':4,'workers':0,'print_every':500}
    d={'train':SampleSet(80),'val':SampleSet(6),'test_s20':SampleSet(6)}
    identity={'target':'speed','mean':0.,'std':1.}
    train_pipeline(model,d,c,identity,tmp_path,torch.device('cpu'),smoke=True)
    assert (tmp_path/'joint_best.pt').exists()
    assert (tmp_path/'last.pt').exists()
    assert (tmp_path/'test_results.json').exists()
    results=json.loads((tmp_path/'test_results.json').read_text())
    assert results[0]['severity']==20
    assert results[0]['mae']>=0

import sys,torch,pytest
from pathlib import Path
from torch import nn
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from fare.traffic import TrafficGraph
from fare.memory import EpisodeMemory
from fare.model import FARE

class DummyBackbone(nn.Module):
    def __init__(self,n,d=8):
        super().__init__();self.d=d;self.n=n;self.mean=0.;self.std=1.
        self.emb=nn.Linear(1,d);self.project=nn.Linear(d,1,bias=False)
    def train(self,mode=True):super().train(False);return self
    def forward(self,x,mask,tod,dow,tokens=None,token_valid=None):
        b,h,n=x.shape
        hidden=self.emb(x.mean(1).unsqueeze(-1))
        pred=x.mean(1)[:,None,:].expand(-1,12,-1).clone()
        if tokens is not None:
            mean=(tokens*token_valid[...,None]).sum(2)/token_valid.sum(2).clamp_min(1)[...,None]
            pred=pred+.01*self.project(mean).squeeze(-1)[:,None,:]
        return {'prediction':pred,'hidden':hidden}

@pytest.fixture
def fixture_data():
    torch.manual_seed(10);b,h,n=2,12,6;m=10;p=12
    c={'model':{'hidden_dim':8,'horizon':12,'alpha_max':1.25},
       'retrieval':{'coarse_k':4,'shortlist_k':3,'final_k':2}}
    adj=torch.zeros(n,n)
    for i in range(n-1):adj[i,i+1]=adj[i+1,i]=1
    graph=TrafficGraph(adj,region_size=3,max_neighbors=2)
    model=FARE(DummyBackbone(n),graph,c)
    x=torch.randn(m,h,n);mask=torch.ones(m,h,n,dtype=torch.bool)
    y=x.mean(1)[:,None,:]+.15*torch.randn(m,p,n)
    mem=EpisodeMemory(x,mask,y,torch.ones_like(y,dtype=torch.bool),torch.arange(m)*2,
                      torch.arange(m)*2+h-1,y*.9)
    batch={'x':torch.randn(b,h,n),'mask':torch.ones(b,h,n,dtype=torch.bool),
           'tod':torch.zeros(b,h,dtype=torch.long),'dow':torch.zeros(b,h,dtype=torch.long),
           'origin':torch.full((b,),100,dtype=torch.long),
           'row_id':torch.arange(b),'target':torch.randn(b,p,n),
           'future_valid':torch.ones(b,p,n,dtype=torch.bool),
           'direct_mask':torch.zeros(b,n,dtype=torch.bool)}
    return model,mem,batch

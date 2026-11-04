from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pytest,torch
from torch import nn
from pcft_stla.v6.graph import DirectedGraph
from pcft_stla.v6.model import PCFTV6

class FakeAttention(nn.Module):
    def __init__(self,d):
        super().__init__();self.ff=nn.Linear(d,d)
    def forward(self,h,node_emb,time_emb):return h+self.ff(h)*0.1

class FakeSTLA(nn.Module):
    def __init__(self,n=5,d=16):
        super().__init__();self.num_nodes=n;self.model_dim=d;self.input_dim=2;self.in_steps=12
        self.steps_per_week=2016;self.tow_embedding_dim=d//2
        self.input_proj=nn.Linear(2,d//2)
        self.tow_embedding=nn.Embedding(2016,d//2)
        self.MLP=nn.Linear(d,d)
        self.attn_layers_st=nn.ModuleList([FakeAttention(d)])
    def node_position_encoding(self,z):return torch.zeros_like(z)
    def time_position_encoding(self,z):return torch.zeros_like(z)

@pytest.fixture
def small():
    torch.manual_seed(10)
    n=5;ei=torch.tensor([[0,1,2,3,4,0],[1,2,3,4,0,3]])
    ea=torch.tensor([[1.,0.,0.5,1.,0.,0.,1.,0.,1.,1.]]*6)
    fi=torch.tensor([[0,2,4],[2,4,1]])
    fa=torch.tensor([[.4,.4,.3,.8,1.,0.]]*3)
    graph=DirectedGraph(ei,ea,n,fi,fa)
    model=PCFTV6(FakeSTLA(n),graph,target_idx=1,max_lag=3,horizon=12)
    B=2;H=12;C=3
    x=torch.randn(B,H,n,C);mask=(torch.rand(B,H,n,C)>0.2).float()
    x=x*mask
    y=torch.randn(B,12,n)
    batch=dict(x_multi=x,observed_mask=mask,missing_age=torch.zeros_like(x),
        x_target=x[...,1],y_norm=y,y_raw=y,future_valid=torch.ones_like(y,dtype=torch.bool),
        direct_mask=torch.ones(B,n,dtype=torch.bool),tod=torch.randint(0,288,(B,H)),dow=torch.randint(0,7,(B,H)))
    return model,batch

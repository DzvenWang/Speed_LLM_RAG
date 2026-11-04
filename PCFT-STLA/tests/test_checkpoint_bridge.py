import torch
from pathlib import Path
from pcft_stla.v6.model import load_original_stla,STLAEncoderBridge

VENDOR='''
import torch
from torch import nn
class FakeBlock(nn.Module):
    def __init__(self,d):
        super().__init__();self.lin=nn.Linear(d,d)
    def forward(self,h,node_emb,time_emb):return h+self.lin(h)*0.01
class STLformer(nn.Module):
    def __init__(self,num_nodes,in_steps,out_steps,steps_per_week,input_dim,output_dim,input_embedding_dim,tow_embedding_dim,feed_forward_dim,num_heads,num_layers,dropout):
        super().__init__();self.num_nodes=num_nodes;self.in_steps=in_steps;self.out_steps=out_steps
        self.input_dim=input_dim;self.output_dim=output_dim;self.model_dim=input_embedding_dim+tow_embedding_dim
        self.steps_per_week=steps_per_week;self.tow_embedding_dim=tow_embedding_dim
        self.input_proj=nn.Linear(input_dim,input_embedding_dim)
        self.tow_embedding=nn.Embedding(steps_per_week,tow_embedding_dim)
        self.MLP=nn.Linear(self.model_dim,self.model_dim)
        self.attn_layers_st=nn.ModuleList([FakeBlock(self.model_dim) for _ in range(num_layers)])
        self.output_proj=nn.Linear(in_steps*self.model_dim,out_steps*output_dim)
    def node_position_encoding(self,z):return torch.zeros_like(z)
    def time_position_encoding(self,z):return torch.zeros_like(z)
'''

def test_original_vendor_checkpoint_and_hidden_bridge(tmp_path):
    path=tmp_path/'Backbone/vendor/STLAformer/model'
    path.mkdir(parents=True)
    (path/'STLformer.py').write_text(VENDOR)
    import importlib.util
    sp=importlib.util.spec_from_file_location('fake_vendor',path/'STLformer.py')
    mod=importlib.util.module_from_spec(sp);sp.loader.exec_module(mod)
    mc={'steps_per_week':2016,'input_dim':2,'input_embedding_dim':8,'tow_embedding_dim':8,
        'feed_forward_dim':32,'num_heads':2,'num_layers':2,'dropout':0.}
    model=mod.STLformer(5,12,12,mc['steps_per_week'],2,1,8,8,32,2,2,0.)
    ck={'model':'STLAformer','num_nodes':5,'model_config':mc,'state_dict':model.state_dict()}
    cp=tmp_path/'best.pt';torch.save(ck,cp)
    new,raw=load_original_stla(tmp_path,cp)
    assert raw['model']=='STLAformer'
    for k in model.state_dict():assert torch.equal(model.state_dict()[k],new.state_dict()[k])
    bridge=STLAEncoderBridge(new,3);bridge.set_target_index(0)
    x=torch.randn(2,12,5);m=torch.ones(2,12,5,3);xm=torch.zeros(2,12,5,3);age=torch.zeros_like(xm)
    tod=torch.randint(0,288,(2,12));dow=torch.randint(0,7,(2,12))
    with torch.no_grad():
        out=bridge(x,xm,m,age,tod,dow)
        t=bridge.base
        tow=torch.nextafter((dow.long()*288+tod.long()).float()/t.steps_per_week,torch.full((2,12),float('inf')))
        tow=tow[:,:,None].expand(2,12,5)
        z=torch.cat([t.input_proj(torch.stack([x,tow],-1)),t.tow_embedding((tow*t.steps_per_week).long())],-1)
        ne=t.node_position_encoding(z);te=t.time_position_encoding(z)
        bx=t.MLP(z);h=z-bx
        for a in t.attn_layers_st:h=a(h,ne,te)
        h=h+bx
    assert torch.allclose(out,h,atol=1e-6)

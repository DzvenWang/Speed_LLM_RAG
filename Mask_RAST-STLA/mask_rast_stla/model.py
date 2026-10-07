from __future__ import annotations
import copy, torch
from torch import nn
from .encoders import StateEncoder
from .stla import MaskSupportSTLAformer

class MaskRASTSTLA(nn.Module):
    def __init__(self,stla:MaskSupportSTLAformer,num_nodes,history,horizon,cfg):
        super().__init__();ec=cfg['encoder'];mc=cfg['model'];self.H=history;self.P=horizon;self.top_k=int(cfg['retrieval']['top_k']);self.temperature=float(cfg['retrieval']['temperature']);rd=int(ec['retrieval_dim']);qd=int(ec['query_dim']);td=int(ec['temporal_dim']);sd=int(ec['spatial_dim'])
        self.query_encoder=StateEncoder(num_nodes,history,td,sd,qd,rd,int(ec['spatial_heads']),float(ec['dropout']));self.memory_encoder=copy.deepcopy(self.query_encoder)
        self.temp_support=nn.Linear(td,rd);self.spat_support=nn.Linear(sd,rd)
        self.failure_router=nn.Sequential(nn.Linear(11,64),nn.GELU(),nn.Dropout(float(mc['dropout'])),nn.Linear(64,2))
        self.retrieval_head=nn.Sequential(nn.Linear(rd,96),nn.GELU(),nn.Dropout(float(mc['dropout'])),nn.Linear(96,horizon))
        self.latent_summary=nn.Sequential(nn.Linear(stla.model_dim,int(mc['latent_gate_dim'])),nn.GELU());self.query_summary=nn.Sequential(nn.Linear(qd,int(mc['latent_gate_dim'])),nn.GELU())
        fin=3+6+2*int(mc['latent_gate_dim']);self.delta=nn.Sequential(nn.Linear(fin,int(mc['fusion_hidden'])),nn.GELU(),nn.Dropout(float(mc['dropout'])),nn.Linear(int(mc['fusion_hidden']),1));self.gate=nn.Sequential(nn.Linear(fin,int(mc['fusion_hidden'])),nn.GELU(),nn.Dropout(float(mc['dropout'])),nn.Linear(int(mc['fusion_hidden']),1));nn.init.zeros_(self.delta[-1].weight);nn.init.zeros_(self.delta[-1].bias);nn.init.constant_(self.gate[-1].bias,float(mc['forecast_gate_bias']))
        self.stla=stla
    def set_train_phase(self,phase,last_layers=2): self.stla.set_train_phase(phase,last_layers)
    def _encode_retrieved_pair(self,repo,ti,si,device):
        # Encode temporal + spatial retrieved histories in one memory-encoder call.
        B,K=ti.shape; idx=torch.cat([ti,si],dim=1); x,m,tod,dow=repo.fetch(idx,device); flat=lambda z:z.reshape(B*(2*K),*z.shape[2:]); e=self.memory_encoder(flat(x),flat(m),flat(tod),flat(dow));
        tn=e['temporal_node'].reshape(B,2*K,e['temporal_node'].shape[1],e['temporal_node'].shape[2])[:,:K]
        sn=e['spatial_node'].reshape(B,2*K,e['spatial_node'].shape[1],e['spatial_node'].shape[2])[:,K:]
        return tn,sn
    def forward(self,batch,memory_index,repo):
        x=batch['x'].float();mask=batch['mask'].float();tod=batch['tod'].long();dow=batch['dow'].long();starts=batch['window_start'].long();q=self.query_encoder(x,mask,tod,dow); memory_index.to(x.device)
        ti,ts,tw,tc=memory_index.search(q['temporal_key'].detach(),'temporal',self.top_k,starts,self.H+self.P,self.temperature);si,ss,sw,sc=memory_index.search(q['spatial_key'].detach(),'spatial',self.top_k,starts,self.H+self.P,self.temperature)
        tnodes,snodes=self._encode_retrieved_pair(repo,ti,si,x.device);t_sup=(tnodes*tw[:,:,None,None]).sum(1);s_sup=(snodes*sw[:,:,None,None]).sum(1);t_sup=self.temp_support(t_sup);s_sup=self.spat_support(s_sup)
        node_cov=mask.mean(1);global_cov=mask.mean((1,2));time_cov=mask.mean(2);min_time=time_cov.min(1).values;std_time=time_cov.std(1,unbiased=False);last_age=(1-mask).flip(1).cumprod(1).sum(1)/max(self.H,1)
        B,N=node_cov.shape;rf=torch.cat([node_cov[...,None],last_age[...,None],global_cov[:,None,None].expand(B,N,1),min_time[:,None,None].expand(B,N,1),std_time[:,None,None].expand(B,N,1),tc[:,None,:].expand(B,N,3),sc[:,None,:].expand(B,N,3)],-1);rw=torch.softmax(self.failure_router(rf),-1);support=rw[...,0:1]*t_sup+rw[...,1:2]*s_sup;reliability=(rw[...,0:1]*tc[:,None,2:3]+rw[...,1:2]*sc[:,None,2:3]).clamp(0,1)
        base,z,latent_gate=self.stla(x,mask,tod,dow,support,reliability);retr=self.retrieval_head(support).transpose(1,2)
        lat=self.latent_summary(z)[:,None,:,:].expand(B,self.P,N,-1);qq=self.query_summary(q['node_query'])[:,None,:,:].expand(B,self.P,N,-1);feat=torch.cat([base[...,None],retr[...,None],(retr-base)[...,None],node_cov[:,None,:,None].expand(B,self.P,N,1),global_cov[:,None,None,None].expand(B,self.P,N,1),tc[:,0][:,None,None,None].expand(B,self.P,N,1),tc[:,2][:,None,None,None].expand(B,self.P,N,1),sc[:,0][:,None,None,None].expand(B,self.P,N,1),sc[:,2][:,None,None,None].expand(B,self.P,N,1),lat,qq],-1);delta=self.delta(feat).squeeze(-1);gate=torch.sigmoid(self.gate(feat)).squeeze(-1);pred=base+gate*delta
        # Positive complete state for multi-positive incomplete-to-full alignment.
        ones=torch.ones_like(batch['x_full']);pos=self.memory_encoder(batch['x_full'].float(),ones,tod,dow)
        return {'prediction':pred,'base_prediction':base,'retrieval_prediction':retr,'forecast_gate':gate,'latent_gate':latent_gate,'router':rw,'q_temporal':q['temporal_key'],'q_spatial':q['spatial_key'],'k_temporal':pos['temporal_key'],'k_spatial':pos['spatial_key'],'temporal_conf':tc,'spatial_conf':sc}

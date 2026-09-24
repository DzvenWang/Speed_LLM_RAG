from __future__ import annotations
import torch
from torch import nn


def make_patchstg_inputs(batch, device):
    """Build the exact inputs expected by the pinned PatchSTG adapter."""
    x=batch['x'].to(device, non_blocking=True).float().unsqueeze(-1)  # [B,T,N,1]
    B,T,N,_=x.shape
    tod=batch['tod_hist'].to(device, non_blocking=True).long().unsqueeze(-1).expand(B,T,N)
    dow=batch['dow_hist'].to(device, non_blocking=True).long().unsqueeze(-1).expand(B,T,N)
    te=torch.stack([tod,dow],dim=-1)
    return x,te


def restore_original_nodes(model, rex):
    """Map padded/reordered PatchSTG slots [B,Tp,Npad,D] back to [B,Tp,N,D]."""
    B,Tp,_,D=rex.shape
    out=rex.new_zeros((B,Tp,int(model.node_num),D))
    out[:,:,model.ori_parts_idx,:]=rex[:,:,model.reo_parts_idx,:]
    return out


def decode_prehead(model, prehead_4d):
    """Apply the pinned official PatchSTG regression head to [B,Tp,N,D]."""
    B,Tp,N,D=prehead_4d.shape
    if int(model.regression_conv.in_channels)!=Tp*D:
        raise RuntimeError(
            f'PatchSTG head expects {model.regression_conv.in_channels} channels but prehead has Tp*D={Tp*D}'
        )
    head_in=prehead_4d.transpose(2,3).reshape(B,Tp*D,N,1)
    pred=model.regression_conv(head_in)
    if pred.ndim==4 and pred.shape[-1]==1:
        pred=pred.squeeze(-1)
    return pred


def forward_with_deep_states(model, x, te, capture_layers=(2,4)):
    """Reproduce the pinned PatchSTG forward while exposing deep states.

    No vendor file is modified. Layer indices are zero-based; e.g. capture index 2
    returns the state after the third dual-attention block.
    """
    embedded=model.embedding(x,te)
    rex=embedded[:,:,model.reo_all_idx,:]
    states={'embedding_4d':embedded}
    capture=set(int(i) for i in capture_layers)
    for i,block in enumerate(model.spa_encoder):
        rex=block(rex)
        if i in capture:
            states[f'layer_{i+1}_patched']=rex
            states[f'layer_{i+1}_node_4d']=restore_original_nodes(model,rex)
    prehead=restore_original_nodes(model,rex)
    pred=decode_prehead(model,prehead)
    states['prehead_4d']=prehead
    states['prehead_node']=prehead.transpose(1,2).reshape(prehead.shape[0],prehead.shape[2],-1)
    return pred,states


def patch_evidence(model, evidence_node):
    """Reorder/pad [B,N,F] evidence into the exact PatchSTG internal slot order."""
    idx=torch.as_tensor(model.reo_all_idx,device=evidence_node.device,dtype=torch.long)
    return evidence_node.index_select(1,idx)


class LatentEvidenceAdapter(nn.Module):
    """Zero-start residual evidence injection in PatchSTG latent space."""
    def __init__(self, latent_dim, evidence_dim, evidence_hidden=128, hidden=256, dropout=0.10,
                 evidence_mean=None, evidence_std=None, evidence_clip=8.0):
        super().__init__()
        self.latent_norm=nn.LayerNorm(int(latent_dim))
        self.evidence_encoder=nn.Sequential(
            nn.Linear(int(evidence_dim),int(evidence_hidden)),nn.GELU(),nn.Dropout(float(dropout)),
            nn.Linear(int(evidence_hidden),int(evidence_hidden)),nn.GELU())
        self.adapter=nn.Sequential(
            nn.Linear(int(latent_dim)+int(evidence_hidden),int(hidden)),nn.GELU(),nn.Dropout(float(dropout)),
            nn.Linear(int(hidden),int(latent_dim)))
        # Protected-reference property: epoch-0 adapter changes no latent value.
        nn.init.zeros_(self.adapter[-1].weight)
        nn.init.zeros_(self.adapter[-1].bias)
        em=torch.zeros(int(evidence_dim)) if evidence_mean is None else torch.as_tensor(evidence_mean,dtype=torch.float32).reshape(-1)
        es=torch.ones(int(evidence_dim)) if evidence_std is None else torch.as_tensor(evidence_std,dtype=torch.float32).reshape(-1)
        if len(em)!=int(evidence_dim) or len(es)!=int(evidence_dim):
            raise RuntimeError('Evidence statistics dimension mismatch')
        self.register_buffer('evidence_mean',em)
        self.register_buffer('evidence_std',torch.clamp(es,min=1e-3))
        self.evidence_clip=float(evidence_clip)

    def _evidence(self,e):
        z=(e-self.evidence_mean)/(self.evidence_std+1e-8)
        z=torch.clamp(torch.nan_to_num(z,nan=0.0,posinf=self.evidence_clip,neginf=-self.evidence_clip),-self.evidence_clip,self.evidence_clip)
        return self.evidence_encoder(z)

    def forward_node4d(self,h,e_node):
        ee=self._evidence(e_node)[:,None,:,:].expand(h.shape[0],h.shape[1],h.shape[2],-1)
        delta=self.adapter(torch.cat([self.latent_norm(h),ee],dim=-1))
        return h+delta,delta

    def forward_patched(self,h,e_patch):
        ee=self._evidence(e_patch)[:,None,:,:].expand(h.shape[0],h.shape[1],h.shape[2],-1)
        delta=self.adapter(torch.cat([self.latent_norm(h),ee],dim=-1))
        return h+delta,delta


class DeepRepresentationIntervention(nn.Module):
    """Strict C0/C1/D1/D2/D3 representation experiment.

    C0: frozen L4/L5 + trainable head (head-only control)
    C1: trainable L4/L5 + head, no evidence (matched control for D3)
    D1: L5/pre-head evidence adapter + head
    D2: L3 evidence adapter + frozen L4/L5 + head
    D3: L3 evidence adapter + trainable L4/L5 + head
    """
    def __init__(self, patchstg, variant, evidence_dim, evidence_mean, evidence_std,
                 evidence_hidden=128, adapter_hidden=256, dropout=0.10, evidence_clip=8.0):
        super().__init__()
        self.patchstg=patchstg
        self.variant=str(variant).upper()
        if self.variant not in {'C0','C1','D1','D2','D3'}:
            raise ValueError(self.variant)
        if len(self.patchstg.spa_encoder)<5:
            raise RuntimeError('v2.1 requires PatchSTG with at least five encoder blocks')
        # Current TraffiDent checkpoint has temporal patch count 1, hence head in_channels == latent width.
        latent_dim=int(self.patchstg.regression_conv.in_channels)
        self.latent_dim=latent_dim
        self.adapter=None if self.variant in {'C0','C1'} else LatentEvidenceAdapter(
            latent_dim,evidence_dim,evidence_hidden,adapter_hidden,dropout,evidence_mean,evidence_std,evidence_clip)
        self._configure_trainable()

    @property
    def trains_deep_blocks(self):
        return self.variant in {'C1','D3'}

    def _configure_trainable(self):
        for p in self.patchstg.parameters():
            p.requires_grad=False
        for p in self.patchstg.regression_conv.parameters():
            p.requires_grad=True
        if self.trains_deep_blocks:
            for block in self.patchstg.spa_encoder[3:5]:
                for p in block.parameters():
                    p.requires_grad=True
        if self.adapter is not None:
            for p in self.adapter.parameters():
                p.requires_grad=True

    def enforce_modes(self, training=True):
        # Frozen official blocks remain eval: dropout cannot silently alter the reference path.
        for i,block in enumerate(self.patchstg.spa_encoder):
            block.train(bool(training and self.trains_deep_blocks and i>=3))
        self.patchstg.regression_conv.train(bool(training))
        if self.adapter is not None:
            self.adapter.train(bool(training))

    def trainable_parameter_groups(self, cfg):
        groups=[]
        if self.adapter is not None:
            ps=[p for p in self.adapter.parameters() if p.requires_grad]
            groups.append({'params':ps,'lr':float(cfg['lr_adapter']),'name':'adapter'})
        if self.trains_deep_blocks:
            bp=[]
            for block in self.patchstg.spa_encoder[3:5]:
                bp.extend([p for p in block.parameters() if p.requires_grad])
            groups.append({'params':bp,'lr':float(cfg['lr_backbone_blocks']),'name':'L4_L5'})
        hp=[p for p in self.patchstg.regression_conv.parameters() if p.requires_grad]
        groups.append({'params':hp,'lr':float(cfg['lr_head']),'name':'head'})
        return groups

    def _run_l4_l5(self,l3):
        h=l3
        if self.variant in {'C0','D1'}:
            # No input gradient is needed before an L5 intervention (D1), and C0 has no adapter.
            with torch.no_grad():
                for block in self.patchstg.spa_encoder[3:5]:
                    h=block(h)
        else:
            # D2 needs gradients through frozen L4/L5 to its L3 adapter; C1/D3 train these blocks.
            for block in self.patchstg.spa_encoder[3:5]:
                h=block(h)
        return h

    def forward(self,l3_patched,evidence_node):
        delta=None
        intervention_base=None
        intervention_adapted=None
        if self.variant in {'D2','D3'}:
            intervention_base=l3_patched
            ep=patch_evidence(self.patchstg,evidence_node)
            l3_patched,delta=self.adapter.forward_patched(l3_patched,ep)
            intervention_adapted=l3_patched
        l5_patched=self._run_l4_l5(l3_patched)
        prehead=restore_original_nodes(self.patchstg,l5_patched)
        if self.variant=='D1':
            intervention_base=prehead
            prehead,delta=self.adapter.forward_node4d(prehead,evidence_node)
            intervention_adapted=prehead
        pred=decode_prehead(self.patchstg,prehead)
        return {
            'pred_norm':pred,
            'delta_h':delta,
            'prehead_4d':prehead,
            'intervention_base':intervention_base,
            'intervention_adapted':intervention_adapted,
        }

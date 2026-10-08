from __future__ import annotations

from .models import ContextRestorer, RetrievalFusionExpert, BackboneAwareRestorer
from .artifacts import load_model_checkpoint

VERSION="0.2.4"

def _alpha_grid(ctx):
    return ctx.cfg["intervention_landscape"]["alpha_grid"]

def build_context_model(ctx):
    c=ctx.cfg["training"]["context"]
    return ContextRestorer(int(ctx.bench.N),ctx.bench.adj,c["hidden_dim"],c["num_layers"],c["num_heads"],c["dropout"],c["ff_mult"],_alpha_grid(ctx)).to(ctx.device)

def build_retrieval_model(ctx):
    c=ctx.cfg["training"]["retrieval"]
    return RetrievalFusionExpert(c["fusion_hidden"],c["dropout"],c.get("support_gate_init_bias",-1.5),_alpha_grid(ctx)).to(ctx.device)

def build_backbone_aware_model(ctx):
    c=ctx.cfg["training"]["backbone_aware"]
    return BackboneAwareRestorer(int(ctx.bench.N),ctx.bench.adj,int(ctx.cfg["experiment"]["horizon"]),c["hidden_dim"],c["num_layers"],c["num_heads"],c["dropout"],c["ff_mult"],_alpha_grid(ctx)).to(ctx.device)

def load_context(ctx):
    m=build_context_model(ctx); load_model_checkpoint(ctx.artifact_root/"context"/"best.pt",m,ctx.device,{"version":VERSION,"target":ctx.target,"backbone_sha":ctx.checkpoint_sha256}); m.eval(); return m

def load_retrieval(ctx):
    m=build_retrieval_model(ctx); load_model_checkpoint(ctx.artifact_root/"retrieval"/"best.pt",m,ctx.device,{"version":VERSION,"target":ctx.target,"backbone_sha":ctx.checkpoint_sha256}); m.eval(); return m

def load_backbone_aware(ctx):
    m=build_backbone_aware_model(ctx); load_model_checkpoint(ctx.artifact_root/"backbone_aware"/"best.pt",m,ctx.device,{"version":VERSION,"target":ctx.target,"backbone_sha":ctx.checkpoint_sha256}); m.eval(); return m

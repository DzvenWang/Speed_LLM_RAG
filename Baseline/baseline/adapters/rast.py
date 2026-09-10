from .base import OfficialAdapter
class RASTAdapter(OfficialAdapter):
    method='RAST'
    def compatibility_audit(self):
        repo=self.ctx.project_root/'vendor'/self.method
        return {'entrypoint':str(repo/'experiments/train.py'),'arch':str(repo/'src/arch/rast_arch.py'),'exists':(repo/'experiments/train.py').exists() and (repo/'src/arch/rast_arch.py').exists(),'formal_protocol':'closest traffic-RAG baseline; same frozen numerical backbone when supported by official universal-backbone interface','retrieval_memory':'train_only'}
    def prepare_bridge(self):
        return {'cache':str(self.ctx.cache_root),'frozen_backbone_json':str(self.ctx.frozen_backbone_json),'needs_frozen_backbone_for_formal':True,'requires_faiss':True,'requires_retrieval_leakage_audit':True}
    def run_command(self, phase):
        if phase=='formal' and not self.ctx.frozen_backbone_json.exists(): raise RuntimeError('FROZEN_BACKBONE.json is required for matched formal RAST.')
        raise RuntimeError('RAST architecture bridge remains explicit; generate a matched-backbone BasicTS config only after the frozen backbone is known.')

from .base import OfficialAdapter
class RASTAdapter(OfficialAdapter):
    method='RAST'
    def compatibility_audit(self):
        repo=self.ctx.project_root/'vendor'/self.method
        return {'entrypoint':str(repo/'experiments/train.py'),'arch':str(repo/'src/arch/rast_arch.py'),'exists':(repo/'experiments/train.py').exists() and (repo/'src/arch/rast_arch.py').exists(),'formal_protocol':'official RAST architecture under external Shared-Protocol Train/Val runner','retrieval_memory':'training only','matched_patchstg_control':'optional appendix adaptation'}
    def prepare_bridge(self):
        return {'cache':str(self.ctx.cache_root),'runner':'baseline/runners/rast_runner.py','requires_faiss':True,'requires_retrieval_leakage_audit':True}
    def run_command(self,phase):
        if phase not in ('development','formal'): raise RuntimeError('RAST selection runner cannot access Test.')
        return ['python','baseline/runners/rast_runner.py','--stage',phase,'--variant','native']

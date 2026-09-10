from .base import OfficialAdapter
class LEAFAdapter(OfficialAdapter):
    method='LEAF'
    def compatibility_audit(self):
        repo=self.ctx.project_root/'vendor'/self.method
        return {'entrypoint':str(repo/'main.py'),'dataset':str(repo/'dataset.py'),'exists':(repo/'main.py').exists() and (repo/'dataset.py').exists(),'native_assumption':'complete continuous PEMS signal plus graph/hypergraph branches and LLM selector','formal_protocol':'framework reference in isolated LEAF environment; dynamic incomplete-view data requires an external dataset runner without changing model architecture'}
    def prepare_bridge(self):
        return {'cache':str(self.ctx.cache_root),'needs_isolated_environment':True,'recommended_python':str(self.ctx.project_root.parent/'envs/LEAF/bin/python'),'install_vllm_into_TFP':False}
    def run_command(self, phase):
        raise RuntimeError('LEAF must run in an isolated environment after its external dynamic-view dataset runner passes smoke tests.')

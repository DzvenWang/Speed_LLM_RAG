from .base import OfficialAdapter
class LEAFAdapter(OfficialAdapter):
    method='LEAF'
    def compatibility_audit(self):
        repo=self.ctx.project_root/'vendor'/self.method
        return {
            'entrypoint':str(repo/'main.py'),'dataset':str(repo/'dataset.py'),
            'exists':(repo/'main.py').exists() and (repo/'dataset.py').exists(),
            'native_assumption':'complete continuous PEMS signal plus GraphBranch/HypergraphBranch and selector',
            'formal_protocol':'official GraphBranch/HypergraphBranch classes under the shared TFP runtime; TraffiDent external dynamic-view runner; LLM selection supplied as post-hoc JSON so no vLLM installation is required in TFP',
        }
    def prepare_bridge(self):
        return {
            'cache':str(self.ctx.cache_root),'needs_isolated_environment':False,
            'recommended_python':'/scratch/mh20/zw1165/envs/TFP/bin/python',
            'install_vllm_into_TFP':False,
            'selector_contract':'row-major query-node JSON entries with final_answer in {1,2}',
        }
    def run_command(self,phase):
        if phase=='branches': return 'python baseline/runners/leaf_runner.py --phase branches'
        if phase=='selector': return 'python baseline/runners/leaf_runner.py --phase selector --selector-json <path>'
        raise ValueError(phase)

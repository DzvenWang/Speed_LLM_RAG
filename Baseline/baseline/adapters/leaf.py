from .base import OfficialAdapter


class LEAFAdapter(OfficialAdapter):
    method='LEAF'
    def compatibility_audit(self):
        repo=self.ctx.project_root/'vendor'/self.method
        return {
            'entrypoint':str(repo/'main.py'),'dataset':str(repo/'dataset.py'),
            'exists':(repo/'main.py').exists() and (repo/'dataset.py').exists(),
            'native_assumption':'complete continuous PEMS signal plus GraphBranch/HypergraphBranch and frozen LLM selector',
            'formal_protocol':'TraffiDent external runner preserves LEAF dual branches, six official augmentations per branch, LLM selection, Huber ranking supervision, update_iters=5 and K=2; original LLaMA-3-70B is replaced only by local Qwen2.5-1.5B-Instruct.',
        }
    def prepare_bridge(self):
        return {
            'cache':str(self.ctx.cache_root),'needs_isolated_environment':False,
            'recommended_python':'python',
            'install_vllm_into_TFP':False,
            'selector_model':'Qwen2.5-1.5B-Instruct',
            'selector_cache':str((next((p for p in Path(__file__).resolve().parents if p.name == 'Baseline'), Path.cwd())).parent / 'model_cache' / 'Qwen2.5-1.5B-Instruct'),
            'selector_contract':'12 official LEAF candidates per query-node; frozen Qwen returns choice 1..12; no future labels in prompt',
        }
    def run_command(self,phase):
        if phase=='branches': return 'python baseline/runners/leaf_runner.py --phase branches'
        if phase in ('selector','qwen'):
            return 'python baseline/runners/leaf_qwen_runner.py --stage development --seed 10'
        raise ValueError(phase)

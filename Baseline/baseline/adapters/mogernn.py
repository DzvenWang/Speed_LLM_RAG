from .base import OfficialAdapter


class MoGERNNAdapter(OfficialAdapter):
    method = 'MoGERNN'

    def compatibility_audit(self):
        repo = self.ctx.project_root / 'vendor' / self.method
        return {
            'entrypoint': str(repo / 'model' / 'MoGERNN.py'),
            'exists': (repo / 'main.py').exists() and (repo / 'model' / 'MoGERNN.py').exists(),
            'native_assumption': 'partially observed sensor network with unobserved locations detected from zero-valued histories',
            'formal_protocol': (
                'Official MoGERNN model is driven by the frozen TraffiDent dynamic-view tensors. '
                'Batch size is fixed to one so sample-specific masks do not get collapsed across queries. '
                'The comparison is reported in the partial-sensing sub-track.'
            ),
        }

    def prepare_bridge(self):
        return {
            'cache': str(self.ctx.cache_root),
            'runner': 'baseline/runners/mogernn_runner.py',
            'batch_size_must_equal_one': True,
            'zero_is_missing_sentinel': True,
            'test_selection': False,
        }

    def run_command(self, phase):
        if phase not in ('development', 'formal'):
            raise ValueError(phase)
        return ['python', 'baseline/runners/mogernn_runner.py', '--stage', phase]

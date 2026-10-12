from .base import OfficialAdapter


class ReMFTAdapter(OfficialAdapter):
    method = 'ReMFT'

    def compatibility_audit(self):
        repo = self.ctx.project_root / 'vendor' / self.method
        return {
            'entrypoint': str(repo / 'run.ipynb'),
            'exists': (repo / 'run.ipynb').exists(),
            'native_assumption': 'complete traffic histories with the original notebook configured around 96-step input and a local frozen language model',
            'formal_protocol': (
                'Only the architecture-definition cells of the official notebook are executed. '
                'Stock data loading/training/testing is never executed. The official H96/P12 working scale is retained; '
                'the extra causal H84 prefix is sensing-matched and the benchmark H12 tail is reused exactly. '
                'Training uses frozen Train, selection uses frozen Val, and a local BERT checkpoint is required.'
            ),
        }

    def prepare_bridge(self):
        return {
            'cache': str(self.ctx.cache_root),
            'runner': 'baseline/runners/remft_runner.py',
            'runtime_check': 'scripts/check_remft_runtime.py',
            'bert_setup': 'scripts/setup_remft_bert.sh',
            'history_adaptation': 'native H96 retained; extra H84 is sensing-matched; final H12 is exact benchmark query',
            'test_selection': False,
        }

    def run_command(self, phase):
        if phase not in ('development', 'formal'):
            raise ValueError(phase)
        return ['python', 'baseline/runners/remft_runner.py', '--stage', phase]

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
                'Stock data loading/training/testing is never executed. The architecture is adapted to H=12, '
                'trained on frozen Train, selected on frozen Val, and requires a local BERT checkpoint.'
            ),
        }

    def prepare_bridge(self):
        return {
            'cache': str(self.ctx.cache_root),
            'runner': 'baseline/runners/remft_runner.py',
            'runtime_check': 'scripts/check_remft_runtime.py',
            'bert_setup': 'scripts/setup_remft_bert.sh',
            'history_adaptation': '96 -> 12 without changing the benchmark horizon',
            'test_selection': False,
        }

    def run_command(self, phase):
        if phase not in ('development', 'formal'):
            raise ValueError(phase)
        return ['python', 'baseline/runners/remft_runner.py', '--stage', phase]

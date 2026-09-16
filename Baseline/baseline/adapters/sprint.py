from .base import OfficialAdapter


class SPRINTAdapter(OfficialAdapter):
    method = 'SPRINT'

    def compatibility_audit(self):
        repo = self.ctx.project_root / 'vendor' / self.method
        return {
            'entrypoint': str(repo / 'src'),
            'exists': (repo / 'src').exists() and (repo / 'pyproject.toml').exists(),
            'native_assumption': 'traffic imputation over a recurring day x link x within-day-interval tensor',
            'formal_protocol': (
                'Causal historical-day adaptation: for each query, SPRINT receives only earlier calendar days '
                'at the same clock-aligned H12 segment plus the current incomplete H12 history. Historical values '
                'come from the canonical SupportBench raw-speed/raw-valid timeline and every historical interval '
                'ends before the query window_start. No P12 future or later-day information is exposed. Current '
                'observed cells are restored exactly after imputation, then the same frozen PatchSTG forecasts.'
            ),
            'reporting_track': 'specialized_reconstruction_subtrack',
            'input_matching_note': (
                'SPRINT uses additional causal historical-day context required by its native recurring-pattern design; '
                'therefore it is not claimed as a strict H12 input-matched main-table baseline.'
            ),
        }

    def prepare_bridge(self):
        return {
            'cache': str(self.ctx.cache_root),
            'runner': 'baseline/runners/sprint_runner.py',
            'context_shape_per_query': '[historical_days+1, N, H] at official call boundary',
            'context_policy': 'same clock-aligned H12 segment from earlier calendar days + current incomplete H12',
            'protect_observed_values': True,
            'frozen_backbone_json': str(self.ctx.frozen_backbone_json),
            'warning': (
                'Specialized comparator only: causal historical-day context gives SPRINT more temporal evidence than '
                'strict H12 main-table methods, so results must be reported separately.'
            ),
        }

    def run_command(self, phase):
        if phase not in ('development', 'formal'):
            raise RuntimeError('SPRINT adapter supports development/formal before evaluation dispatch.')
        if not self.ctx.frozen_backbone_json.exists():
            raise RuntimeError('SPRINT -> frozen PatchSTG requires the frozen Backbone registry.')
        return (
            f'python baseline/runners/sprint_runner.py --stage {phase} '
            f'--seed {self.ctx.seed}'
        )

"""Local CPU-only regression tests for dual-mode reference Backbone routing."""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from baseline.backbones.training_mode import (
    checkpoint_mode, current_mode, method_reference_parent, reference_stage,
    set_mode_from_cli,
)
from baseline.backbones.reference_backbone import resolve_reference_checkpoint
from baseline.runners.common import make_run_dir


class ReferenceBackboneTrainingModeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.cfg = {
            'project': {'root': str(self.root), 'backbone_exp_root': str(self.root / 'Backbone_exps'),
                        'exp_root': str(self.root / 'Baseline_exps')},
            'experiment': {'target': 'speed', 'subgraph': 'TraffiDent_Net_06'},
        }
        self.envpatch = patch.dict(os.environ, {}, clear=False)
        self.envpatch.start()
        self.addCleanup(self.envpatch.stop)
        os.environ.pop('BACKBONE_TRAIN_MODE', None)
        os.environ.pop('BASELINE_BACKBONE_CHECKPOINT', None)

    def _save(self, mode: str, *, legacy: bool = False) -> Path:
        root = (self.root / 'Backbone_exps' / 'speed' / 'STLAformer' / 'TraffiDent_Net_06'
                / reference_stage(mode))
        dest = root / '20261009_100000_seed10' / 'best.pt'
        dest.parent.mkdir(parents=True, exist_ok=True)
        ck = {
            'model': 'STLAformer', 'target': 'speed', 'seed': 10,
            'state_dict': {'weight': torch.ones(1)}, 'model_config': {},
            'mean': 1., 'std': 2., 'run_identity': {'project': 'Backbone', 'consumer_stage': 'formal'},
        }
        if not legacy:
            ck['train_data_mode'] = mode
        torch.save(ck, dest)
        return dest

    @patch('baseline.backbones.reference_backbone._formal_cache_identity', return_value={})
    def test_resolves_two_modes_and_preserves_original_default(self, _cache):
        inc = self._save('incomplete', legacy=True)
        full = self._save('complete')
        p, _ = resolve_reference_checkpoint(self.cfg, 'STLAformer')
        self.assertEqual(p, inc)
        os.environ['BACKBONE_TRAIN_MODE'] = 'complete'
        p, ck = resolve_reference_checkpoint(self.cfg, 'STLAformer')
        self.assertEqual(p, full)
        self.assertEqual(checkpoint_mode(ck), 'complete')

    @patch('baseline.backbones.reference_backbone._formal_cache_identity', return_value={})
    def test_explicit_checkpoint_must_match_requested_mode(self, _cache):
        inc = self._save('incomplete', legacy=True)
        full = self._save('complete')
        with self.assertRaisesRegex(RuntimeError, 'train_data_mode'):
            resolve_reference_checkpoint(self.cfg, 'STLAformer', checkpoint=full)
        os.environ['BACKBONE_TRAIN_MODE'] = 'complete'
        with self.assertRaisesRegex(RuntimeError, 'train_data_mode'):
            resolve_reference_checkpoint(self.cfg, 'STLAformer', checkpoint=inc)
        p, _ = resolve_reference_checkpoint(self.cfg, 'STLAformer', checkpoint=full)
        self.assertEqual(p, full)

    @patch('baseline.backbones.reference_backbone._formal_cache_identity', return_value={})
    def test_no_silent_fallback_to_incomplete_if_complete_missing(self, _cache):
        self._save('incomplete', legacy=True)
        os.environ['BACKBONE_TRAIN_MODE'] = 'complete'
        with self.assertRaisesRegex(RuntimeError, 'formal_complete'):
            resolve_reference_checkpoint(self.cfg, 'STLAformer')

    @patch('baseline.backbones.reference_backbone._formal_cache_identity', return_value={})
    def test_rejects_mislabeled_checkpoint_inside_complete_directory(self, _cache):
        dest = self._save('complete')
        ck = torch.load(dest, weights_only=False)
        ck['train_data_mode'] = 'incomplete'
        torch.save(ck, dest)
        os.environ['BACKBONE_TRAIN_MODE'] = 'complete'
        with self.assertRaisesRegex(RuntimeError, 'train_data_mode'):
            resolve_reference_checkpoint(self.cfg, 'STLAformer')

    def test_legacy_and_complete_baseline_artifacts_cannot_collide(self):
        incomplete_dir = make_run_dir(self.cfg, 'RAFT-STLAformer', 'formal', 10, backbone_model='STLAformer')
        os.environ['BACKBONE_TRAIN_MODE'] = 'complete'
        complete_dir = make_run_dir(self.cfg, 'RAFT-STLAformer', 'formal', 10, backbone_model='STLAformer')
        self.assertNotEqual(incomplete_dir, complete_dir)
        self.assertIn('/STLAformer/complete/TraffiDent_Net_06/formal/', complete_dir.as_posix())
        self.assertIn('/STLAformer/TraffiDent_Net_06/formal/', incomplete_dir.as_posix())
        self.assertTrue(incomplete_dir.is_dir() and complete_dir.is_dir())

    def test_eval_output_path_isolated(self):
        base = Path('Baseline_exps/speed/evaluation/ImputeFormer')
        self.assertEqual(method_reference_parent(base, 'STLAformer'), base / 'STLAformer')
        self.assertEqual(method_reference_parent(base, 'STLAformer', 'complete'), base / 'STLAformer' / 'complete')

    def test_mode_cli_and_env(self):
        self.assertEqual(set_mode_from_cli(None), 'incomplete')
        self.assertEqual(set_mode_from_cli('complete'), 'complete')
        self.assertEqual(current_mode(), 'complete')
        with self.assertRaisesRegex(ValueError, 'Invalid'):
            set_mode_from_cli('wrong')


if __name__ == '__main__':
    unittest.main()

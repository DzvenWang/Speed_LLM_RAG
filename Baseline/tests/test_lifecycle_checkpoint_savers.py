"""CPU-only regression tests for lifecycle checkpoint helpers.

Extract the function AST to test the actual helper implementation without
importing heavy optional method/vendor dependencies or initializing GPUs.
"""
from __future__ import annotations

import ast
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

RUNNERS = Path(__file__).resolve().parents[1] / 'baseline' / 'runners'


def helper_from_source(method: str):
    path = RUNNERS / f'{method}_runner.py'
    module = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    matches = [n for n in module.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
               and n.name == '_save_lifecycle_smoke_checkpoint']
    if len(matches) != 1:
        raise AssertionError(f'{method} expected one lifecycle helper')
    helper_ast = matches[0]
    isolated = ast.Module(body=[helper_ast], type_ignores=[])
    ast.fix_missing_locations(isolated)
    globals_ = {'os': os, 'Path': Path, 'torch': torch}
    exec(compile(isolated, str(path), 'exec'), globals_)
    return globals_['_save_lifecycle_smoke_checkpoint'], helper_ast, module


class LifecycleCheckpointTests(unittest.TestCase):
    def test_non_retrieval_store_helpers_have_no_undefined_call(self):
        for method in ('ginar', 'raft', 'remft', 'saraf'):
            with self.subTest(method=method):
                helper, node, module = helper_from_source(method)
                self.assertEqual([a.arg for a in node.args.args], ['payload'])
                self.assertFalse(any(isinstance(n, ast.Name) and n.id == 'save_store' and
                                     isinstance(n.ctx, ast.Load) for n in ast.walk(module)))
                self.assertTrue(callable(helper))

    def test_checkpoint_save_works_for_every_affected_runner(self):
        for method in ('ginar', 'raft', 'remft', 'saraf'):
            with self.subTest(method=method), tempfile.TemporaryDirectory() as temp:
                helper, _, _ = helper_from_source(method)
                path = Path(temp) / method / 'lifecycle.pt'
                payload = {'model': {'weight': torch.tensor([1.0, 2.0])}, 'best_epoch': 1}
                with patch.dict(os.environ, {'BASELINE_LIFECYCLE_CHECKPOINT': str(path)}):
                    self.assertEqual(helper(payload), path)
                self.assertTrue(path.is_file())
                saved = torch.load(path, map_location='cpu', weights_only=True)
                self.assertEqual(saved['best_epoch'], 1)
                self.assertTrue(torch.equal(saved['model']['weight'], payload['model']['weight']))
                self.assertFalse((path.parent / 'retrieval_store_best.npz').exists())

    def test_checkpoint_disabled_without_env(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop('BASELINE_LIFECYCLE_CHECKPOINT', None)
            for method in ('ginar', 'raft', 'remft', 'saraf'):
                with self.subTest(method=method):
                    helper, _, _ = helper_from_source(method)
                    self.assertIsNone(helper({'model': {}}))

    def test_rast_store_support_is_preserved(self):
        helper, node, module = helper_from_source('rast')
        self.assertEqual([a.arg for a in node.args.args], ['payload', 'retrieval_store'])
        self.assertTrue(any(isinstance(n, ast.FunctionDef) and n.name == 'save_store'
                            for n in module.body))
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'lifecycle.pt'
            npz = path.with_name('retrieval_store_best.npz')
            # Use actual production serializer by extracting only save_store.
            func_ast = next(n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == 'save_store')
            globals_ = {'os': os, 'Path': Path, 'torch': torch, 'np': np}
            exec(compile(ast.fix_missing_locations(ast.Module(body=[func_ast], type_ignores=[])),
                         str(RUNNERS / 'rast_runner.py'), 'exec'), globals_)
            isolated = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
            exec(compile(isolated, str(RUNNERS / 'rast_runner.py'), 'exec'), globals_)
            snapshot = {'tv': np.ones((2, 3), np.float32), 'sv': np.zeros((2, 3), np.float32),
                        'tt': [np.ones((3,), np.float32)], 'st': [np.zeros((3,), np.float32)]}
            with patch.dict(os.environ, {'BASELINE_LIFECYCLE_CHECKPOINT': str(path)}):
                globals_['_save_lifecycle_smoke_checkpoint']({'model': {}}, snapshot)
            self.assertTrue(path.is_file())
            self.assertTrue(npz.is_file())
            with np.load(npz, allow_pickle=False) as data:
                np.testing.assert_array_equal(data['temporal_vectors'], snapshot['tv'])
                self.assertEqual(int(data['nt']), 1)


if __name__ == '__main__':
    unittest.main()

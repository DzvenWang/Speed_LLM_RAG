"""Regression tests for legacy Formal validation and vendor tree warning.

These tests do not require the actual traffic caches, checkpoints or GPU.
"""
from __future__ import annotations
import contextlib
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


def load_source_lock():
    spec = importlib.util.spec_from_file_location('baseline_source_lock_test', ROOT/'scripts/source_lock.py')
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


class LegacyCompatibilityTest(unittest.TestCase):
    def test_legacy_version_is_supported_without_removing_current_versions(self):
        import yaml
        cfg = yaml.safe_load((ROOT/'configs/baseline.yaml').read_text(encoding='utf-8'))
        supported = {str(v) for v in cfg['compatible_formal_versions']}
        self.assertIn('1.3.10', supported)
        self.assertIn('1.4.7', supported)
        self.assertIn('1.4.8', supported)
        self.assertEqual(str(cfg['version']), '1.4.8')
        self.assertEqual((ROOT/'VERSION').read_text().strip(), '1.4.8')


class SourceHashWarningTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lock_mod = load_source_lock()

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.root = Path(self.dir.name)
        self.repo = self.root/'vendor'/'SARAF'
        self.repo.mkdir(parents=True)
        (self.repo/'model.py').write_text('original source\n',encoding='utf-8')
        self.commit = 'a' * 40
        self.sources = {'SARAF': {'remote': 'https://example.test/saraf.git',
                                  'commit': self.commit, 'policy': 'exact'}}
        self.lock = {'repositories': {'SARAF': {
            'remote': 'https://example.test/saraf.git',
            'commit': self.commit,
            'tree_sha256': self.lock_mod.tree_hash(self.repo),
        }}}

    def test_matching_source_passes_without_warning(self):
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            _, mode = self.lock_mod.verify_one(self.root, self.sources, 'SARAF', self.lock)
        self.assertEqual(mode, 'tree-lock-no-git')
        self.assertEqual(err.getvalue(), '')

    def test_different_tree_only_warns_and_preserves_original_lock(self):
        expected = self.lock['repositories']['SARAF']['tree_sha256']
        (self.repo/'model.py').write_text('changed source\n', encoding='utf-8')
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            _, mode = self.lock_mod.verify_one(self.root, self.sources, 'SARAF', self.lock)
        self.assertIn('WARNING-tree-hash-mismatch', mode)
        self.assertIn('WARNING SOURCE_TREE_HASH_MISMATCH', err.getvalue())
        self.assertIn('expected_tree=', err.getvalue())
        self.assertIn('actual_tree=', err.getvalue())
        self.assertEqual(self.lock['repositories']['SARAF']['tree_sha256'], expected)
        self.assertNotEqual(expected, self.lock_mod.tree_hash(self.repo))

    def test_other_critical_checks_still_fail(self):
        self.sources['SARAF']['remote'] = 'https://example.test/different.git'
        with self.assertRaisesRegex(RuntimeError, 'remote mismatch'):
            self.lock_mod.verify_one(self.root, self.sources, 'SARAF', self.lock)
        self.sources['SARAF']['remote'] = 'https://example.test/saraf.git'
        self.repo.rename(self.repo.with_name('missing'))
        with self.assertRaisesRegex(RuntimeError, 'Missing source directory'):
            self.lock_mod.verify_one(self.root, self.sources, 'SARAF', self.lock)


if __name__ == '__main__':
    unittest.main()

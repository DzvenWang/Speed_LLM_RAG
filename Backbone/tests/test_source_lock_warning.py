"""Source-lock compatibility tests that do not require official vendored code."""
from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'source_lock.py'
spec = importlib.util.spec_from_file_location('backbone_source_lock_under_test', SCRIPT)
source_lock = importlib.util.module_from_spec(spec)
spec.loader.exec_module(source_lock)

CONFIG = {'model': {'official_source': {
    'commit': 'pin-commit', 'remote': 'https://example.com/PatchSTG.git',
}}}


class SourceLockWarningsTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        (self.root / 'vendor' / 'PatchSTG').mkdir(parents=True)
        (self.root / 'vendor' / 'PatchSTG' / 'models').mkdir()
        (self.root / 'vendor' / 'PatchSTG' / 'lib').mkdir()
        (self.root / 'vendor' / 'PatchSTG' / 'models' / 'model.py').write_text('original\n')
        (self.root / 'vendor' / 'PatchSTG' / 'lib' / 'utils.py').write_text('utility\n')
        self.lock = self.root / 'vendor' / 'SOURCE_LOCK.json'
        self.digest = source_lock.tree_hash(self.root / 'vendor' / 'PatchSTG')
        self._write_lock()
        self.patcher = patch.object(source_lock, 'ROOT', self.root)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)

    def _write_lock(self, commit='pin-commit', remote='https://example.com/PatchSTG.git'):
        self.lock.write_text(json.dumps({'format': 2, 'repositories': {'PatchSTG': {
            'commit': commit, 'remote': remote, 'tree_sha256': self.digest,
        }}}))

    def test_matching_patchstg_accepted(self):
        source_lock.verify('PatchSTG', CONFIG)

    def test_hash_mismatch_warning_does_not_rewrite_lock(self):
        original = self.lock.read_bytes()
        (self.root / 'vendor' / 'PatchSTG' / 'models' / 'model.py').write_text('modified\n')
        source_lock.verify('PatchSTG', CONFIG)
        self.assertEqual(self.lock.read_bytes(), original)

    def test_strict_hash_still_fails(self):
        (self.root / 'vendor' / 'PatchSTG' / 'models' / 'model.py').write_text('modified\n')
        with self.assertRaisesRegex(SystemExit, 'SOURCE_TREE_HASH_MISMATCH_STRICT'):
            source_lock.verify('PatchSTG', CONFIG, strict_hash=True)

    def test_missing_vendor_is_fatal(self):
        with self.assertRaisesRegex(SystemExit, 'SOURCE_MISSING'):
            source_lock.verify('STLAformer', CONFIG)

    def test_non_patch_vendor_is_independent_from_patchstg(self):
        repo = self.root / 'vendor' / 'STLAformer'
        repo.mkdir()
        (repo / 'model').mkdir()
        (repo / 'model' / 'STLformer.py').write_text('stla\n')
        (self.root / 'vendor' / 'PatchSTG' / 'models' / 'model.py').unlink()
        (self.root / 'vendor' / 'PatchSTG' / 'models').rmdir()
        (self.root / 'vendor' / 'PatchSTG' / 'lib' / 'utils.py').unlink()
        (self.root / 'vendor' / 'PatchSTG' / 'lib').rmdir()
        (self.root / 'vendor' / 'PatchSTG').rmdir()
        source_lock.verify('STLAformer', CONFIG)

    def test_strict_requires_a_lock_entry(self):
        repo = self.root / 'vendor' / 'STLAformer' / 'model'
        repo.mkdir(parents=True)
        (repo / 'STLformer.py').write_text('stla\n')
        with self.assertRaisesRegex(SystemExit, 'SOURCE_LOCK_NOT_AVAILABLE_STRICT'):
            source_lock.verify('STLAformer', CONFIG, strict_hash=True)

    def test_missing_model_file_is_fatal(self):
        (self.root / 'vendor' / 'PatchSTG' / 'models' / 'model.py').unlink()
        with self.assertRaisesRegex(SystemExit, 'SOURCE_FILE_MISSING'):
            source_lock.verify('PatchSTG', CONFIG)

    def test_patch_source_identity_still_fatal(self):
        self._write_lock(commit='incorrect-commit')
        with self.assertRaisesRegex(SystemExit, 'SOURCE_IDENTITY_MISMATCH'):
            source_lock.verify('PatchSTG', CONFIG)

    def test_git_does_not_affect_tree_hash(self):
        nested = self.root / 'vendor' / 'PatchSTG' / '.git'
        nested.mkdir()
        (nested / 'config').write_text('test')
        self.assertEqual(self.digest, source_lock.tree_hash(self.root / 'vendor' / 'PatchSTG'))

    def test_missing_patch_lock_is_fatal(self):
        self.lock.unlink()
        with self.assertRaisesRegex(SystemExit, 'SOURCE_LOCK_MISSING'):
            source_lock.verify('PatchSTG', CONFIG)

    def test_invalid_lock_is_fatal(self):
        self.lock.write_text('{broken-json')
        with self.assertRaisesRegex(SystemExit, 'SOURCE_LOCK_INVALID'):
            source_lock.verify('PatchSTG', CONFIG)


if __name__ == '__main__':
    unittest.main()

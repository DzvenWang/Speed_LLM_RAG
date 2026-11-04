#!/usr/bin/env python3
"""Check dependencies on Backbone and Base_SupportBench without training/GPU.

This script checks actual Python importability; Pylance additionally needs
pyrightconfig.json in the *opened VS Code workspace*.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path


def check(repo: Path) -> None:
    for relative in ('PCFT-STLA', 'Backbone', 'Base_SupportBench'):
        path = repo / relative
        if not path.is_dir():
            raise FileNotFoundError(f'Missing sibling directory: {path}')
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    expected = {
        'backbone.utils': repo / 'Backbone/backbone/utils.py',
        'backbone.shared_cache': repo / 'Backbone/backbone/shared_cache.py',
        'supportbench.dataset': repo / 'Base_SupportBench/supportbench/dataset.py',
    }
    for name, expected_file in expected.items():
        spec = importlib.util.find_spec(name)
        if spec is None or not spec.origin:
            raise ImportError(f'Cannot resolve {name}; expected {expected_file}')
        actual = Path(spec.origin).resolve()
        if actual != expected_file.resolve():
            raise ImportError(f'{name} resolves to {actual}, expected {expected_file}')
        print(f'IMPORT_RESOLVED {name} -> {actual}', flush=True)

    from backbone.utils import load_yaml  # noqa: F401
    from backbone.shared_cache import cache_path, validate_shared_cache  # noqa: F401
    from supportbench.dataset import SupportBench  # noqa: F401
    print('PCFT_V6_IMPORTS_PASS', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', type=Path,
                        default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    check(args.repo.expanduser().resolve())

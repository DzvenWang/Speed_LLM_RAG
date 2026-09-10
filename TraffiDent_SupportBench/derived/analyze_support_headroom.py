#!/usr/bin/env python3
"""Compatibility wrapper for v1.2.2 RAG headroom diagnostics."""
from __future__ import annotations
import runpy
from pathlib import Path
if __name__=='__main__':
    runpy.run_path(str(Path(__file__).with_name('analyze_rag_headroom.py')),run_name='__main__')

#!/usr/bin/env python3
"""Static contract: ReMFT12 must expose the original PHI -> llmodel forward path."""
from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "baseline/runners/remft_runner.py"

tree = ast.parse(RUNNER.read_text())
cls = next((n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ReMFT12"), None)
assert cls is not None, "ReMFT12 class missing"
forward = next((n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "forward"), None)
assert forward is not None, "ReMFT12.forward missing"

# Require a return expression equivalent in structure to self.llmodel(self.phi(x)).
returns = [n for n in ast.walk(forward) if isinstance(n, ast.Return)]
assert returns, "ReMFT12.forward has no return"
text = ast.unparse(returns[0].value)
assert "self.llmodel" in text and "self.phi" in text, text
print("REMFT FORWARD CONTRACT STATIC SMOKE PASS", text)

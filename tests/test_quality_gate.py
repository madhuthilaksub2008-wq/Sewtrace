"""SPEC 11 quality gate: every test names the requirement it checks."""

from __future__ import annotations

import ast
import re
from pathlib import Path

TESTS = Path(__file__).parent
SPEC = TESTS.parent / "SPEC.md"
ID = re.compile(r"\b(?:REQ-[A-Z]+-\d{2}|C-\d{2})\b")
NAME_ID = re.compile(r"(?:^|_)(?:req_[a-z]+_\d{2}|c_\d{2})(?:_|$)")


def _test_functions():
    for path in sorted(TESTS.glob("test_*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test"):
                yield path, node


def test_c_08_every_test_names_a_requirement_or_constraint():
    """C-08: each test function has a REQ- or C- ID in its name or docstring, and the ID is in SPEC.md."""
    known = set(ID.findall(SPEC.read_text()))
    missing, unknown = [], []
    for path, node in _test_functions():
        ids = ID.findall(ast.get_docstring(node) or "")
        if not ids and not NAME_ID.search(node.name):
            missing.append(f"{path.name}:{node.lineno} {node.name}")
        unknown += [f"{path.name}:{node.lineno} {i}" for i in ids if i not in known]
    assert not missing, "tests without a REQ-/C- ID:\n" + "\n".join(missing)
    assert not unknown, "IDs that SPEC.md does not define:\n" + "\n".join(unknown)

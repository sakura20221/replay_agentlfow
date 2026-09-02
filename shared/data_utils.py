"""Dependency-free helpers used while freezing benchmark data."""

from __future__ import annotations

import ast
import hashlib
import json
import re
import unicodedata
from pathlib import Path


def normalized_task_text(value: str) -> str:
    """Normalize task text for search/evaluation leakage checks."""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(value))).strip().casefold()


def mbpp_entry_point(code: str, tests: list[str]) -> str:
    """Return the top-level candidate definition called by MBPP's tests.

    The first definition in the reference solution is often a helper or class.
    The official tests, rather than source order, define the public entry point.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        tree = None

    definitions = []
    if tree is not None:
        definitions = [
            node.name
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        ]
    definition_set = set(definitions)

    for test in tests:
        try:
            test_tree = ast.parse(str(test))
        except SyntaxError:
            continue
        for node in ast.walk(test_tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in definition_set:
                    return node.func.id

    functions = []
    if tree is not None:
        functions = [
            node.name
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
    return (functions or definitions or [""])[0]


def refresh_file_integrity(manifest: dict, data_dir: Path) -> None:
    """Refresh exact-byte metadata for every frozen JSONL in ``data_dir``."""
    files = {}
    for path in sorted(data_dir.glob("*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            rows = sum(1 for line in handle if line.strip())
        files[path.name] = {
            "rows": rows,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    manifest["file_integrity"] = {
        "algorithm": "sha256 over exact file bytes",
        "files": files,
    }


def write_manifest(path: Path, manifest: dict) -> None:
    refresh_file_integrity(manifest, path.parent)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")

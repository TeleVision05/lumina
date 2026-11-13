"""The pre-commit file-length hook."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from lumina_train.vectors import REPO_ROOT

_spec = importlib.util.spec_from_file_location(
    "check_file_length", REPO_ROOT / "scripts" / "hooks" / "check_file_length.py"
)
assert _spec is not None and _spec.loader is not None
hook = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hook)


def _write(root: Path, rel: str, lines: int) -> str:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x = 1\n" * lines)
    return rel


def test_fails_on_long_source_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ok = _write(tmp_path, "training/ok.py", 500)
    long = _write(tmp_path, "web/src/long.ts", 501)
    assert hook.main([ok]) == 0
    assert hook.main([ok, long]) == 1


def test_ignores_non_source_paths(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    data = _write(tmp_path, "shared/sh_vectors.json", 2000)
    docs = _write(tmp_path, "docs/notes.md", 2000)
    assert hook.main([data, docs]) == 0

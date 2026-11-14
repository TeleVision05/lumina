"""docs/EVAL.md is generated from the committed results, metadata and training log."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lumina_train import paths, report


@pytest.fixture
def redirected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Write EVAL.md and figures under tmp_path; read the committed inputs."""
    monkeypatch.setattr(report, "DOCS_DIR", tmp_path)
    monkeypatch.setattr(report, "MEDIA_DIR", tmp_path / "media")
    (tmp_path / "media").mkdir()
    return tmp_path


def test_write_eval_md_runs_on_the_committed_files(redirected: Path) -> None:
    results = json.loads((paths.EVAL_DIR / "results.json").read_text())
    report.write_eval_md(results)
    text = (redirected / "EVAL.md").read_text()
    meta = json.loads(paths.MODEL_META_PATH.read_text())
    assert meta["training"]["command"] in text
    assert meta["training"]["checkpoint_rule"] in text


def test_write_eval_md_survives_metadata_without_a_checkpoint_rule(
    redirected: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    meta = json.loads(paths.MODEL_META_PATH.read_text())
    del meta["training"]["checkpoint_rule"]
    old = tmp_path / "old-meta.json"
    old.write_text(json.dumps(meta))
    monkeypatch.setattr(report, "MODEL_META_PATH", old)
    results = json.loads((paths.EVAL_DIR / "results.json").read_text())
    report.write_eval_md(results)
    assert "not recorded" in (redirected / "EVAL.md").read_text()

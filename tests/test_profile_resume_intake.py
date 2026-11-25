"""Tests for the standalone intake profiler's input expansion."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_profiler():
    script = Path(__file__).parents[1] / "scripts" / "profile_resume_intake.py"
    spec = importlib.util.spec_from_file_location("profile_resume_intake", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_expand_resume_inputs_includes_immediate_supported_files_once(tmp_path: Path) -> None:
    (tmp_path / "resume-b.PDF").touch()
    (tmp_path / "resume-a.pdf").touch()
    (tmp_path / "notes.docx").touch()
    (tmp_path / "ignore.csv").touch()
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "not-included.pdf").touch()

    profiler = _load_profiler()
    resumes, invalid = profiler._expand_resume_inputs([tmp_path, tmp_path / "resume-a.pdf"])

    assert [resume.name for resume in resumes] == ["notes.docx", "resume-a.pdf", "resume-b.PDF"]
    assert invalid == []


def test_expand_resume_inputs_reports_empty_or_missing_paths(tmp_path: Path) -> None:
    profiler = _load_profiler()
    empty = tmp_path / "empty"
    empty.mkdir()
    missing = tmp_path / "missing.pdf"

    resumes, invalid = profiler._expand_resume_inputs([empty, missing])

    assert resumes == []
    assert invalid == [empty, missing]

"""Run manifest: git SHA, versions, models, dataset hashes, seed."""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_sha() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10, check=True)
        return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def build_manifest(
    *,
    seed: int,
    models: list[str],
    datasets: Sequence[str | Path] = (),
    config_version: str = "unversioned",
    matcher_version: str = "unversioned",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "created_at": datetime.now(UTC).isoformat(),
        "git_sha": git_sha(),
        "config_version": config_version,
        "matcher_version": matcher_version,
        "models": sorted(set(models)),
        "seed": seed,
        "datasets": {Path(p).name: sha256_file(p) for p in datasets},
        **(extra or {}),
    }


def write_manifest(run_dir: str | Path, manifest: dict[str, Any]) -> Path:
    d = Path(run_dir)
    d.mkdir(parents=True, exist_ok=True)
    p = d / "manifest.json"
    p.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return p

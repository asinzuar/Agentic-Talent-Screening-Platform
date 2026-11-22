"""Adversarial-twin generator: a clean base resume plus exactly one attack, with labels by construction.

Grid (architecture section 4.4): family x intensity (3) x position (family-specific, up to 3) x
visibility (visible / hidden). The **core** grid uses medium intensity, the first two positions of each
family and every visibility the family allows (32 twins per base across all families, so 20-25
bases give the 600-800 twins the plan calls for); **full** uses every combination.

Everything is deterministic for a given (base, seed, grid cell), offline, and written only under
``data/eval`` or ``runs``.

Hidden twins are rendered only in formats where the channel is real (never TXT). The off_page channel is
excluded from the automatic grids (L0 and the pipeline extractor both clip to the page, so it can never be
scored); request it with ``channel="off_page"``. File metadata uses the base resume id as title, so a twin's
metadata equals its base's and leaks no label information through :func:`~..datasets.synthetic.safe_output_dir`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from ..datasets.synthetic import (
    _RENDERERS,
    FORMATS,
    SyntheticResume,
    _unlink_if_symlink,
    make_resume,
    safe_output_dir,
)
from .families import (
    FAMILIES,
    DEFAULT_HIDDEN_CHANNELS,
    HIDDEN_CHANNELS,
    INTENSITIES,
    MAX_FILE_BYTES,
    AttackLabel,
    Built,
    Ctx,
)
from .render import (
    CHANNEL_FORMATS,
    Item,
    inject,
    inspect_docx_payload,
    inspect_pdf_payload,
    render_docx_items,
    render_pdf_items,
    render_text,
)

GridCell = tuple[str, str, str, str]  # family, intensity, position, visibility


@dataclass
class Twin:
    base: SyntheticResume
    label: AttackLabel
    items: list[Item]
    built: Built

    @property
    def text(self) -> str:
        return render_text(self.items)[0]


def _channel_for(built: Built, visibility: str, cycle: int, explicit: str | None) -> str:
    if visibility != "hidden":
        if explicit not in (None, "visible"):
            raise ValueError("a channel other than 'visible' needs visibility='hidden'")
        return "visible"
    if explicit is not None:
        if explicit not in HIDDEN_CHANNELS:
            raise ValueError(f"unknown hidden channel {explicit!r}; choose from {HIDDEN_CHANNELS}")
        if not (set(CHANNEL_FORMATS[explicit]) & set(built.formats) - {"txt"}):
            raise ValueError(f"channel {explicit!r} cannot be rendered for variant {built.variant!r}")
        return explicit
    usable = [c for c in DEFAULT_HIDDEN_CHANNELS if set(CHANNEL_FORMATS[c]) & set(built.formats) - {"txt"}]
    return usable[cycle % len(usable)]


def make_twin(base: SyntheticResume, family: str, intensity: str, position: str, visibility: str, *,
              seed: int = 0, channel: str | None = None, variant: str | None = None,
              jd_keywords: tuple[str, ...] | None = None) -> Twin:
    """Build one twin: ``base`` plus exactly one injection described by the returned label."""
    if family not in FAMILIES:
        raise ValueError(f"unknown family {family!r}; choose from {sorted(FAMILIES)}")
    fam = FAMILIES[family]
    if intensity not in INTENSITIES:
        raise ValueError(f"unknown intensity {intensity!r}")
    if position not in fam.positions:
        raise ValueError(f"{family} does not allow position {position!r}; allowed: {fam.positions}")
    if visibility not in fam.visibilities:
        raise ValueError(f"{family} does not allow visibility {visibility!r}; allowed: {fam.visibilities}")
    rng = random.Random(f"twin:{seed}:{base.resume_id}:{family}:{intensity}:{position}:{visibility}")
    built = fam.build(Ctx(base, intensity, position, rng, variant, jd_keywords))  # type: ignore[arg-type]
    cycle = int(base.resume_id[-4:]) + fam.positions.index(position) + INTENSITIES.index(intensity)  # type: ignore[arg-type]
    ch = _channel_for(built, visibility, cycle, channel)
    items = inject(base.layout(), position, built, ch)
    _, spans = render_text(items)
    vis = "n/a" if visibility == "n/a" else visibility
    twin_id = f"{base.resume_id}-{family}-{intensity}-{position}-{vis.replace('/', '')}" + (
        f"-{ch}" if ch != "visible" else "")
    # TXT has no hidden channel, so a hidden twin gets no TXT input file (it would read as visible text)
    formats = [f for f in FORMATS if f in CHANNEL_FORMATS[ch] and f in built.formats
               and not (f == "txt" and ch != "visible")]
    label = AttackLabel(
        twin_id=twin_id, base_id=base.resume_id, family=family, family_name=fam.name, variant=built.variant,
        channel=ch, intensity=intensity, position=position, visibility=vis, seed=seed,  # type: ignore[arg-type]
        payload=built.text, payload_sha=hashlib.sha256(built.text.encode("utf-8")).hexdigest(), spans=spans,
        expected_layers=list(fam.expected_layers), slide_channel=fam.slide_channel, formats=formats,
        target_skill=built.target_skill, notes=built.notes, reaches_default_extractor=ch != "off_page")
    return Twin(base, label, items, built)


def grid(kind: str = "core", families: Iterable[str] | None = None) -> list[GridCell]:
    if kind not in ("core", "full"):
        raise ValueError("grid must be 'core' or 'full'")
    wanted = list(families) if families is not None else list(FAMILIES)
    unknown = [f for f in wanted if f not in FAMILIES]
    if unknown:
        raise ValueError(f"unknown families: {unknown}")
    cells: list[GridCell] = []
    for fid in wanted:
        fam = FAMILIES[fid]
        intensities = ("medium",) if kind == "core" else INTENSITIES
        positions = fam.positions[:2] if kind == "core" else fam.positions
        cells.extend((fid, i, p, v) for i in intensities for p in positions for v in fam.visibilities)
    return cells


def generate_twins(bases: Iterable[SyntheticResume], grid_kind: str = "core", seed: int = 0,
                   families: Iterable[str] | None = None) -> list[Twin]:
    cells = grid(grid_kind, families)
    return [make_twin(b, *cell, seed=seed) for b in bases for cell in cells]


def _verify_hidden(twin: Twin, fmt: str, path: Path, boxes: list[list[float]]) -> None:
    """Self-check: a hidden-channel payload must be in the file but not visible."""
    if twin.label.channel == "visible" or fmt == "txt":
        return
    if fmt == "pdf":
        reports = inspect_pdf_payload(path, boxes, twin.built.text)
        ok = bool(reports) and all(r.hidden for r in reports)
    else:
        ok = all(inspect_docx_payload(path, line) for line in twin.built.lines)
    if not ok:
        raise RuntimeError(f"{twin.label.twin_id}: {fmt} payload is not hidden as labelled ({twin.label.channel})")


def write_twins(twins: list[Twin], out_dir: str | Path = "data/eval/twins", formats: tuple[str, ...] = FORMATS,
                base: str | Path | None = None) -> Path:
    """Write twin files, their clean bases, ``labels.jsonl`` and ``index.json`` (with SHA-256s).

    Each rendered file must stay under ``MAX_FILE_BYTES`` (the A10 safety cap); a larger file is
    deleted and the call fails.
    """
    if not formats or set(formats) - set(FORMATS):
        raise ValueError(f"formats must be a non-empty subset of {FORMATS}")
    out = safe_output_dir(out_dir, base)
    bases_dir = out / "bases"
    bases_dir.mkdir(parents=True, exist_ok=True)
    for name in ("labels.jsonl", "index.json"):
        _unlink_if_symlink(out / name)
    index: list[dict] = []
    seen_bases: dict[str, SyntheticResume] = {}
    with (out / "labels.jsonl").open("w", encoding="utf-8") as lab:
        for t in twins:
            seen_bases[t.base.resume_id] = t.base
            files: dict[str, dict] = {}
            for fmt in (f for f in formats if f in t.label.formats):
                p = out / f"{t.label.twin_id}.{fmt}"
                _unlink_if_symlink(p)
                boxes: list[list[float]] = []
                if fmt == "txt":
                    p.write_text(t.text, encoding="utf-8")
                elif fmt == "docx":
                    render_docx_items(t.items, p, t.base.resume_id)
                else:
                    boxes = render_pdf_items(t.items, p, t.base.resume_id)
                    t.label.pdf_bbox = boxes
                if p.stat().st_size > MAX_FILE_BYTES:
                    p.unlink()
                    raise RuntimeError(f"{p.name} exceeds the {MAX_FILE_BYTES} byte cap")
                _verify_hidden(t, fmt, p, boxes)
                files[fmt] = {"file": p.name, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
            lab.write(t.label.model_dump_json() + "\n")
            index.append({"twin_id": t.label.twin_id, "base_id": t.base.resume_id, "files": files})
    base_files: dict[str, dict] = {}
    for rid, r in sorted(seen_bases.items()):
        base_files[rid] = {}
        for fmt in formats:
            p = bases_dir / f"{rid}.{fmt}"
            _unlink_if_symlink(p)
            _RENDERERS[fmt](r, p)
            base_files[rid][fmt] = {"file": f"bases/{p.name}", "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
    (out / "index.json").write_text(json.dumps({"n": len(twins), "twins": index, "bases": base_files}, indent=1),
                                    encoding="utf-8")
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="agentic_talent_screener.evaluation.attacks.twins")
    p.add_argument("--n-bases", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--grid", choices=["core", "full"], default="core")
    p.add_argument("--families", default="", help="comma-separated family ids, default all")
    p.add_argument("--out", default="data/eval/twins")
    p.add_argument("--formats", default=",".join(FORMATS))
    a = p.parse_args(argv)
    try:
        if not 1 <= a.n_bases <= 200:
            raise ValueError("--n-bases must be in [1, 200]")
        fams = [f for f in a.families.split(",") if f] or None
        twins = generate_twins([make_resume(i, a.seed) for i in range(a.n_bases)], a.grid, a.seed, fams)
        out = write_twins(twins, a.out, tuple(a.formats.split(",")))
    except ValueError as exc:
        print(f"error: {exc}")
        return 2
    print(f"wrote {len(twins)} twins to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

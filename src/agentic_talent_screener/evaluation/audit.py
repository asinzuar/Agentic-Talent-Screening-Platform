"""E11 human audit kit (slide 9: "audit ~10 explanations ... ~95% of cited lines check out").

Agents cannot grade. This module exports about ten explanation lines to a sheet (CSV and Markdown), humans rate
each one, and ``import_ratings`` merges the ratings back into the run's report.

Sheet columns the rater fills in: ``usefulness_1_5`` (an integer 1 to 5), ``citation_correct`` (``right`` or
``wrong``: does the quoted text really back the line?), ``notes`` and ``rater`` (any short label; use initials,
not a name). The sheet hides the automatic check so raters are not anchored by it; ``audit_manifest.json`` keeps
it for the agreement figure.

Rules the importer enforces: the file must be a CSV of at most ``MAX_BYTES`` and ``MAX_ROWS``; every ``row_id`` must
exist in the manifest (ratings for unknown rows are rejected and listed, not ignored); an unrated row is pending,
not zero; an invalid rating (out of range, not right/wrong) rejects that row with a reason; one rater cannot rate a
row twice. Cells that start with ``= + - @`` are prefixed with a quote on export so a spreadsheet never runs them.

Verdicts: the citation share is judged against the slide-9 target with a Wilson interval; with fewer than
``AUDIT_MIN_N`` rated rows it reads ``not measurable (n<min)``. Ten rows can never prove 95%
(all ten right gives a lower bound near 72%), so the honest result of a ten-row audit is ``inconclusive`` at best.
"""

from __future__ import annotations

import csv
import re
import hashlib
import io
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .metrics.citation import CITATION_TARGET, rate, verdict_lower

AUDIT_DIR = "audit"
SHEET_CSV, SHEET_MD, MANIFEST, RESULTS = "audit_sheet.csv", "audit_sheet.md", "audit_manifest.json", "audit_results.json"
DEFAULT_AUDIT_N = 10
AUDIT_MIN_N = 10
MAX_BYTES, MAX_ROWS, MAX_NOTE = 1_000_000, 2_000, 500
RATING_COLUMNS = ("usefulness_1_5", "citation_correct", "notes", "rater")
SHOWN_COLUMNS = ("row_id", "candidate_id", "requirement", "explanation", "quote", "span_id", "span_text")
COLUMNS = (*SHOWN_COLUMNS, *RATING_COLUMNS)
_FORMULA = ("=", "+", "-", "@", "\t", "\r")
_RIGHT, _WRONG = "right", "wrong"


def _cell(value: Any) -> str:
    return "" if value is None else str(value).replace("\r", " ")


def _digest(explanation: Any, quote: Any, span_id: Any) -> str:
    return hashlib.sha256(json.dumps([_cell(explanation), _cell(quote), _cell(span_id)]).encode()).hexdigest()


def _unsafe(cell: str) -> str:
    """Undo the export's formula guard: a leading quote in front of ``= + - @`` was added by ``_safe``."""
    return cell[1:] if cell.startswith("'") and cell[1:].startswith(_FORMULA) else cell


def _audit_dir(run_dir: str | Path, create: bool = False) -> Path:
    d = Path(run_dir) / AUDIT_DIR
    if d.is_symlink():  # a linked audit/ folder could send writes elsewhere
        raise ValueError("audit folder is a symbolic link: refusing to use it")
    if create:
        d.mkdir(parents=True, exist_ok=True)
    return d


def _echo_id(rid: str) -> str:
    """Only ids of the exported shape are echoed back; anything else (a wrong file's first column) is not."""
    return rid if re.fullmatch(r"A\d{2,3}", rid) else "(not an audit row id)"


def _write(path: Path, text: str, **kw: Any) -> None:
    if path.is_symlink():  # never write through a pre-existing link (as build_claims_report does)
        path.unlink()
    path.write_text(text, encoding="utf-8", **kw)


def _safe(cell: Any) -> str:
    s = "" if cell is None else str(cell).replace("\r", " ")
    return "'" + s if s.startswith(_FORMULA) else s


def select_items(items: Sequence[Mapping[str, Any]], n: int = DEFAULT_AUDIT_N, seed: int = 0) -> list[dict[str, Any]]:
    """Deterministic, spread-out pick: at most one line per candidate and per requirement before repeating."""
    ranked = sorted((dict(i) for i in items),
                    key=lambda i: hashlib.sha256(f"{seed}|{i['key']}".encode()).hexdigest())
    chosen: list[dict[str, Any]] = []
    seen_c: set[str] = set()
    seen_r: set[str] = set()
    for strict in (True, False):
        for it in ranked:
            if len(chosen) >= n:
                break
            if it in chosen:
                continue
            if strict and (it["candidate_id"] in seen_c or it["requirement"] in seen_r):
                continue
            chosen.append(it)
            seen_c.add(it["candidate_id"])
            seen_r.add(it["requirement"])
    return chosen


def export_kit(items: Sequence[Mapping[str, Any]], run_dir: str | Path, *, n: int = DEFAULT_AUDIT_N,
               seed: int = 0) -> dict[str, Any]:
    """Write ``audit/audit_sheet.csv``, ``audit_sheet.md`` and ``audit_manifest.json`` for ~``n`` explanation lines.

    NOTE for any non-synthetic pool: ``span_text`` is the raw L0 span text, so it can show the employer and other
    identity text that L3 redacted out of the quote. With synthetic resumes that is harmless; before the claims suite
    accepts real resumes, ``span_text`` must get the same redaction as the quote.

    Each item needs: key, candidate_id, requirement, explanation, quote, span_id, span_text and ``auto_ok`` (the
    automatic citation check, kept in the manifest only). Returns the manifest."""
    d = _audit_dir(run_dir, create=True)
    picked = select_items(items, n, seed)
    rows, manifest_rows = [], {}
    for i, it in enumerate(picked, start=1):
        row_id = f"A{i:02d}"
        rows.append({"row_id": row_id, **{c: it[c] for c in SHOWN_COLUMNS[1:]}})
        manifest_rows[row_id] = {"candidate_id": it["candidate_id"], "auto_ok": bool(it["auto_ok"]),
                                 "auto_failure": it.get("auto_failure"),
                                 "content_sha256": _digest(it["explanation"], it["quote"], it["span_id"])}
    manifest = {"schema": 1, "n_rows": len(rows), "seed": seed, "rows": manifest_rows,
                "labels": "synthetic resumes; humans rate, agents do not", "target": CITATION_TARGET}
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=COLUMNS, lineterminator="\n")
    w.writeheader()
    for r in rows:
        w.writerow({**{k: _safe(v) for k, v in r.items()}, **{c: "" for c in RATING_COLUMNS}})
    _write(d / SHEET_CSV, buf.getvalue(), newline="")
    _write(d / SHEET_MD, _sheet_md(rows))
    _write(d / MANIFEST, json.dumps(manifest, indent=1, sort_keys=True))
    return manifest


def _sheet_md(rows: Sequence[Mapping[str, Any]]) -> str:
    L = ["# Explanation audit sheet", "",
         "Rate each line. Fill the CSV (`audit_sheet.csv`), not this page; this is the same content to read.", "",
         "- `usefulness_1_5`: 1 = useless, 5 = exactly what a recruiter needs.",
         "- `citation_correct`: `right` if the quoted text really backs the line, otherwise `wrong`.",
         "- `rater`: your initials. Do not write a candidate's identity anywhere.", ""]
    for r in rows:
        L += [f"## {r['row_id']} (candidate {r['candidate_id']})", "",
              f"- Requirement: {r['requirement']}", f"- Explanation: {r['explanation']}",
              f"- Quote: \"{r['quote']}\"", f"- Cited span `{r['span_id']}` reads: \"{r['span_text']}\"",
              "- usefulness (1-5): ____    citation: right / wrong    notes: ____", ""]
    return "\n".join(L)


# ----------------------------------------------------------------------------- import


def _parse(path: Path) -> list[dict[str, str]]:
    if path.stat().st_size > MAX_BYTES:
        raise ValueError(f"{path.name}: larger than {MAX_BYTES} bytes")
    text = path.read_text(encoding="utf-8-sig")
    rows = list(csv.DictReader(io.StringIO(text)))
    if len(rows) > MAX_ROWS:
        raise ValueError(f"{path.name}: more than {MAX_ROWS} rows")
    if not rows or "row_id" not in (rows[0] or {}):
        raise ValueError(f"{path.name}: not an audit sheet (no row_id column)")
    return rows


def import_ratings(run_dir: str | Path, csv_path: str | Path, *, min_n: int = AUDIT_MIN_N) -> dict[str, Any]:
    """Merge human ratings into ``audit/audit_results.json`` and return it. Raises ``ValueError`` for a file that is
    not an audit sheet or a run with no exported kit; invalid rows are rejected with reasons inside the result."""
    d = _audit_dir(run_dir)
    try:
        manifest = json.loads((d / MANIFEST).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError("this run has no audit kit (run the claims suite first)") from exc
    known: Mapping[str, Any] = manifest["rows"]
    rated: dict[tuple[str, str], dict[str, Any]] = {}
    edited: list[str] = []
    rejected: list[dict[str, str]] = []
    pending = 0
    for line, row in enumerate(_parse(Path(csv_path)), start=2):
        rid = (row.get("row_id") or "").strip()
        use, cit = (row.get("usefulness_1_5") or "").strip(), (row.get("citation_correct") or "").strip().casefold()
        rater = (row.get("rater") or "").strip()[:40]
        if rid not in known:
            rejected.append({"line": str(line), "row_id": _echo_id(rid), "reason": "unknown row_id"})
            continue
        if not use and not cit:
            pending += 1
            continue
        want = known[rid].get("content_sha256")
        if want and _digest(_unsafe(row.get("explanation") or ""), _unsafe(row.get("quote") or ""),
                            _unsafe(row.get("span_id") or "")) != want and rid not in edited:
            edited.append(rid)  # a warning, not a rejection: the rating is kept but flagged
        if not use.isdigit() or not 1 <= int(use) <= 5:
            rejected.append({"line": str(line), "row_id": rid, "reason": "usefulness_1_5 must be an integer 1-5"})
            continue
        if cit not in (_RIGHT, _WRONG):
            rejected.append({"line": str(line), "row_id": rid, "reason": "citation_correct must be right or wrong"})
            continue
        if (rid, rater) in rated:
            rejected.append({"line": str(line), "row_id": rid, "reason": "this rater already rated this row"})
            continue
        rated[(rid, rater)] = {"row_id": rid, "usefulness": int(use), "citation_right": cit == _RIGHT,
                               "rater": rater, "notes": (row.get("notes") or "")[:MAX_NOTE]}
    out = summarize_ratings(list(rated.values()), known, pending=pending, rejected=rejected, min_n=min_n)
    out["rows_with_edited_content"] = sorted(edited)  # explanation, quote or span cells differ from the export
    _write(d / RESULTS, json.dumps(out, indent=1, sort_keys=True))
    return out


def summarize_ratings(ratings: Sequence[Mapping[str, Any]], known: Mapping[str, Any], *, pending: int = 0,
                      rejected: Sequence[Mapping[str, str]] = (), min_n: int = AUDIT_MIN_N) -> dict[str, Any]:
    """Citation share (Wilson, verdict), usefulness distribution and agreement with the automatic check."""
    n = len(ratings)
    right = sum(r["citation_right"] for r in ratings)
    out: dict[str, Any] = {
        "schema": 1, "n_ratings": n, "n_rows_rated": len({r["row_id"] for r in ratings}),
        "n_rows_exported": len(known), "n_pending": pending, "n_rejected": len(rejected),
        "rejected": list(rejected)[:50], "target": CITATION_TARGET, "min_n": min_n,
        "source": "human ratings (not agent-graded)"}
    if n == 0:
        out["citation_right"] = {"status": "not measurable (no human ratings imported)", "value": None, "n": 0,
                                 "ci95": None, "target": CITATION_TARGET}
        out["usefulness"] = {"mean": None, "n": 0, "distribution": {}}
        out["agreement_with_automatic_check"] = rate(0, 0, min_n)
        return out
    out["citation_right"] = verdict_lower(right, n, CITATION_TARGET, min_n)
    scores = [r["usefulness"] for r in ratings]
    out["usefulness"] = {"mean": sum(scores) / n, "n": n, "min": min(scores), "max": max(scores),
                         "distribution": {str(k): v for k, v in sorted(Counter(scores).items())}}
    agree = sum(bool(r["citation_right"]) == bool(known[r["row_id"]].get("auto_ok")) for r in ratings)
    out["agreement_with_automatic_check"] = rate(agree, n, min_n)
    out["note"] = ("A handful of rows cannot prove 95%: with n=10 even ten right leaves the interval's lower bound "
                   "near 72%. Treat the verdict as 'not enough yet' until more rows are rated.")
    return out


def load_results(run_dir: str | Path) -> dict[str, Any] | None:
    try:
        return json.loads((Path(run_dir) / AUDIT_DIR / RESULTS).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


__all__ = ["AUDIT_MIN_N", "COLUMNS", "DEFAULT_AUDIT_N", "export_kit", "import_ratings", "load_results",
           "select_items", "summarize_ratings"]

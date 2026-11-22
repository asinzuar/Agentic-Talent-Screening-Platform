import json
import re

import pymupdf
import pytest
from docx import Document

from agentic_talent_screener.evaluation.datasets.schemas import GoldFact
from agentic_talent_screener.evaluation.datasets.synthetic import (
    FORMATS,
    generate_pool,
    main,
    make_resume,
    render_docx,
    render_pdf,
    render_txt,
    safe_output_dir,
    write_pool,
)
from agentic_talent_screener.evaluation.cli import main as cli_main
from agentic_talent_screener.resume_intake.l0_forensics import scan_file
from agentic_talent_screener.resume_intake.l2_resume_agent import ProposedClaim
from agentic_talent_screener.resume_intake.l3_evidence_verifier import _evidence_level

_CLAIM_KIND = {"skills": "skill", "certifications": "certification", "experience": "experience"}


def _l3_level(fact):
    claim = ProposedClaim(claim_id="c1", kind=_CLAIM_KIND[fact.section], value=fact.value, source_span_ids=["p1:s0001"])
    return _evidence_level(claim, fact.source_text)


@pytest.fixture(scope="module")
def pool():
    return generate_pool(160, seed=7)


def _skills(resume):
    return [(f.value, f.evidence_level) for f in resume.gold_facts() if f.kind == "skill"]


def test_pool_size_and_bounds(pool):
    assert len(pool) == 160 and len({r.resume_id for r in pool}) == 160
    for bad in (149, 201, 0, -1, True, "160"):
        with pytest.raises(ValueError):
            generate_pool(bad)


def test_deterministic_and_seed_sensitive(pool):
    again = generate_pool(160, seed=7)
    assert [r.to_text() for r in pool] == [r.to_text() for r in again]
    assert [r.to_text() for r in pool] != [r.to_text() for r in generate_pool(160, seed=8)]
    assert make_resume(5, 7).to_text() == pool[5].to_text()


def test_gold_facts_match_text_by_construction(pool):
    kinds = set()
    for r in pool:
        text = r.to_text()
        facts = r.gold_facts()
        for f in facts:
            kinds.add(f.kind)
            assert f.source_text in text
            assert (f.evidence_level is not None) == (f.kind == "skill")
        skills = [f for f in facts if f.kind == "skill"]
        assert {f.value for f in skills} == {s for s, _ in r.skills}
        assert len(skills) == len(r.skills)
        assert len([f for f in facts if f.kind == "role"]) == len(r.roles)
    assert kinds == {"skill", "role", "date_range", "education", "institution"}


def test_evidence_levels_consistent_with_text(pool):
    seen = set()
    for r in pool:
        for f in r.gold_facts():
            if f.kind != "skill":
                continue
            seen.add(f.evidence_level)
            assert f.value in f.source_text
            assert f.section == {"E1": "skills", "E2": "certifications"}.get(f.evidence_level, "experience")
            assert _l3_level(f) == f.evidence_level  # real L3 function (pipeline-consistency, not accuracy)
    assert seen == {"E1", "E2", "E3", "E4"}


def test_dates_ordered_and_ranges_valid(pool):
    for r in pool:
        for a, b in zip(r.roles, r.roles[1:]):
            assert a.end is not None and a.end <= b.start
            assert a.start < b.start
        for role in r.roles:
            assert role.end is None or role.start < role.end
        assert all(f.value.count("..") == 1 for f in r.gold_facts() if f.kind == "date_range")


def test_identity_swap_changes_only_identity(pool):
    r = pool[0]
    s = r.with_identity(name="Test Person", gender="nonbinary", college="Other College", location="Oslo, NO")
    assert s.skills == r.skills and s.roles == r.roles
    text = s.to_text()
    assert "Test Person" in text and "Other College" in text and "Oslo, NO" in text
    assert r.identity.name not in text and r.identity.college not in text
    inst = [f for f in s.gold_facts() if f.kind == "institution"][0]
    assert inst.value == "Other College"
    assert _skills(s) == _skills(r)
    changed = {a for a, b in zip(r.to_text().splitlines(), s.to_text().splitlines()) if a != b}
    old, new = r.to_text().splitlines(), s.to_text().splitlines()
    assert len(old) == len(new)
    identity_lines = {0, 1, len(old) - 1}  # name, contact, education
    assert [i for i, (a, b) in enumerate(zip(old, new)) if a != b] == sorted(identity_lines)
    assert changed
    keep = lambda x: [f for f in x.gold_facts() if f.kind not in ("education", "institution")]  # noqa: E731
    assert keep(s) == keep(r)
    with pytest.raises(ValueError):
        r.with_identity(gender="robot")
    with pytest.raises(ValueError):
        r.with_identity(bogus="x")


def test_identity_variety(pool):
    assert len({r.identity.name for r in pool}) > 30
    assert {r.identity.gender for r in pool} == {"female", "male", "nonbinary"}
    assert all("@example.com" in r.to_text() for r in pool[:5])


def test_rendered_files_carry_gold_text_and_pass_intake(tmp_path):
    for i in range(4):
        r = make_resume(i, 1)
        paths = {f: tmp_path / f"{i}.{f}" for f in FORMATS}
        render_txt(r, paths["txt"])
        render_docx(r, paths["docx"])
        render_pdf(r, paths["pdf"])
        extracted = {
            "txt": paths["txt"].read_text(encoding="utf-8"),
            "docx": "\n".join(p.text for p in Document(str(paths["docx"])).paragraphs),
            "pdf": "\n".join(pg.get_text() for pg in pymupdf.open(str(paths["pdf"]))),
        }
        for fmt, text in extracted.items():
            flat = re.sub(r"\s+", " ", text)
            for f in r.gold_facts():
                assert re.sub(r"\s+", " ", f.source_text) in flat, (fmt, f.source_text)
        for fmt, p in paths.items():  # the real L0 intake scan accepts them with no findings
            report = scan_file(p)
            assert report["decision"] == "allow" and report["issues"] == [], (fmt, report["issues"])


def test_long_content_paginates_pdf(tmp_path):
    r = make_resume(0, 0).with_identity(college="Long College " * 400)
    p = tmp_path / "x.pdf"
    render_pdf(r, p)
    assert pymupdf.open(str(p)).page_count >= 2


def test_safe_output_dir(tmp_path):
    assert safe_output_dir("data/eval/x", tmp_path) == (tmp_path / "data/eval/x").resolve()
    assert safe_output_dir("runs", tmp_path) == (tmp_path / "runs").resolve()
    for bad in ("data", "data/eval/../..", "../data/eval", "src", str(tmp_path / "elsewhere"), "data/evalx"):
        with pytest.raises(ValueError):
            safe_output_dir(bad, tmp_path)


def test_write_pool_outputs_and_gold_roundtrip(tmp_path):
    pool = generate_pool(150, seed=2)[:3]
    out = write_pool(pool, "data/eval/syn", base=tmp_path)
    index = json.loads((out / "index.json").read_text())
    assert index["n"] == 3 and index["seed"] == 2
    for r in index["resumes"]:
        assert set(r["files"]) == set(FORMATS)
        for meta in r["files"].values():
            assert (out / meta["file"]).stat().st_size > 0 and len(meta["sha256"]) == 64
    rows = [json.loads(line) for line in (out / "gold.jsonl").read_text().splitlines()]
    assert len(rows) == 3
    assert [GoldFact(**f) for f in rows[0]["facts"]] == pool[0].gold_facts()
    write_pool(pool, "runs/x", formats=("txt",), base=tmp_path)
    assert not list((tmp_path / "runs/x").glob("*.pdf"))


def test_write_pool_rejects_bad_paths_and_formats(tmp_path):
    pool = generate_pool(150)[:1]
    with pytest.raises(ValueError):
        write_pool(pool, "elsewhere", base=tmp_path)
    assert not (tmp_path / "elsewhere").exists()
    with pytest.raises(ValueError):
        write_pool(pool, "data/eval/x", formats=("exe",), base=tmp_path)
    with pytest.raises(ValueError):
        write_pool(pool, "data/eval/x", formats=(), base=tmp_path)


def test_cli_main(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert main(["--n", "150", "--out", "data/eval/cli", "--formats", "txt"]) == 0
    assert len(list((tmp_path / "data/eval/cli").glob("*.txt"))) == 150
    assert main(["--n", "150", "--out", "../escape"]) == 2
    assert main(["--n", "5", "--out", "data/eval/cli"]) == 2
    assert "error" in capsys.readouterr().out


def test_docx_pdf_bytes_deterministic(tmp_path, monkeypatch):
    import hashlib
    import time

    r = make_resume(4, 3)

    def digests(tag):
        out = {}
        for fmt, fn in (("txt", render_txt), ("docx", render_docx), ("pdf", render_pdf)):
            p = tmp_path / f"{tag}.{fmt}"
            fn(r, p)
            out[fmt] = hashlib.sha256(p.read_bytes()).hexdigest()
        return out

    first = digests("a")
    time.sleep(2.2)  # past the 2 s zip timestamp resolution
    assert digests("b") == first


def test_write_pool_hashes_stable_across_runs(tmp_path):
    pool = generate_pool(150, seed=9)[:2]
    i1 = json.loads((write_pool(pool, "data/eval/a", base=tmp_path) / "index.json").read_text())
    i2 = json.loads((write_pool(pool, "data/eval/b", base=tmp_path) / "index.json").read_text())
    assert i1 == i2


def test_write_pool_does_not_follow_symlink(tmp_path):
    victim = tmp_path / "victim.txt"
    victim.write_text("keep")
    out = tmp_path / "data/eval/s"
    out.mkdir(parents=True)
    try:
        (out / "S-0000.txt").symlink_to(victim)
    except OSError:
        pytest.skip("symlinks unavailable")
    write_pool(generate_pool(150)[:1], "data/eval/s", formats=("txt",), base=tmp_path)
    assert victim.read_text() == "keep"
    assert not (out / "S-0000.txt").is_symlink() and (out / "S-0000.txt").stat().st_size > 0


def test_cli_paths_go_through_safe_output_dir(capsys):
    assert cli_main(["run", "--runs-dir", "src"]) == 2
    assert cli_main(["run", "--runs-dir", "../x"]) == 2
    assert cli_main(["run", "--runs-dir", "data/eval/../../src"]) == 2
    assert cli_main(["twins", "--out", "src/evil"]) == 2
    assert "must stay under" in capsys.readouterr().err
    assert cli_main(["twins", "--out", "data/eval/twins"]) == 2  # valid path, subcommand still a stub
    assert "not implemented" in capsys.readouterr().err

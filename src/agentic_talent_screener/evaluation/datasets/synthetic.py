"""Template-based synthetic resume generator: clean pool with gold facts by construction.

Deterministic for a given seed. Identity fields (name, gender, college, location) live in
:class:`~.schemas.Identity` and can be swapped with :meth:`SyntheticResume.with_identity`; gold
facts are recomputed from the layout, so they always match the rendered text. Names are drawn from a
small built-in list of arbitrary first/last-name pairings (no real people).

Evidence ladder, aligned with the intake L3 heuristic (``l3_evidence_verifier._evidence_level``):
  E1  skill listed only in the Skills section
  E2  also named in a certification / coursework line
  E3  also used in an experience bullet that states what was done, with no outcome words
  E4  also used in an experience bullet with a concrete outcome (a % or an outcome verb)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Literal

from .schemas import EvidenceLevel, GoldFact, Identity

Format = Literal["txt", "docx", "pdf"]
FORMATS: tuple[Format, ...] = ("txt", "docx", "pdf")
ALLOWED_OUTPUT_ROOTS: tuple[str, ...] = ("data/eval", "runs")
DEFAULT_POOL_SIZE = 160
MIN_POOL_SIZE, MAX_POOL_SIZE = 150, 200
_MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
_REFERENCE_YM = (2026, 1)  # fixed "now" so output never depends on the clock

_FEMALE = ("Amara", "Priya", "Lena", "Sofia", "Mei", "Nadia", "Elena", "Ingrid", "Zara", "Hana", "Rosa", "Tara")
_MALE = ("Arjun", "Mateo", "Jonas", "Kofi", "Liam", "Omar", "Felix", "Ravi", "Tomas", "Yusuf", "Ivan", "Diego")
_NONBINARY = ("Alex", "Sam", "Robin", "Quinn", "Jordan", "Rowan")
_LAST = ("Okafor", "Lindqvist", "Marchetti", "Tanaka", "Haddad", "Novak", "Sharma", "Duarte", "Kowalski",
         "Mensah", "Petrov", "Ferreira", "Nguyen", "Brennan", "Castillo", "Larsen", "Iyer", "Bauer")
_COLLEGES = ("Lakeview State University", "Northfield Institute of Technology", "Riverbend University",
             "Summit Ridge College", "Harborview University", "Cedar Valley Polytechnic",
             "Eastgate University", "Westmoor Institute", "Pinecrest College", "Stonebridge University")
_LOCATIONS = ("Austin, TX", "Toronto, ON", "Berlin, DE", "Pune, IN", "Lisbon, PT", "Denver, CO",
              "Dublin, IE", "Singapore, SG", "Nairobi, KE", "Seattle, WA", "Warsaw, PL", "Sydney, AU")
_COMPANIES = ("Brightwave Systems", "Alderon Labs", "Quillfeather Software", "Tidewater Analytics",
              "Copperline Digital", "Maple and Finch Technologies", "Ironbridge Data", "Solstice Cloud",
              "Foxglove Apps", "Harrow Logistics Tech", "Kestrel Payments", "Lumen Retail Systems")

# domain -> (titles, skills); skills are distinct tokens, none a substring of another
_DOMAINS: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "backend": (("Backend Engineer", "Software Engineer", "API Developer"),
                ("Python", "FastAPI", "PostgreSQL", "Redis", "Docker", "Kafka", "GraphQL", "Celery", "SQLAlchemy", "RabbitMQ")),
    "data": (("Data Engineer", "Data Analyst", "Machine Learning Engineer"),
             ("Pandas", "Spark", "Airflow", "dbt", "Snowflake", "scikit-learn", "PyTorch", "Tableau", "BigQuery", "NumPy")),
    "frontend": (("Frontend Engineer", "UI Developer", "Web Developer"),
                 ("TypeScript", "React", "Vue", "Webpack", "Tailwind", "Jest", "Cypress", "Redux", "Storybook", "Sass")),
    "devops": (("DevOps Engineer", "Site Reliability Engineer", "Platform Engineer"),
               ("Kubernetes", "Terraform", "Ansible", "Prometheus", "Grafana", "Jenkins", "Helm", "Nginx", "Bash", "Packer")),
    "qa": (("QA Engineer", "Test Automation Engineer", "SDET"),
           ("Selenium", "Pytest", "Playwright", "JMeter", "Postman", "Cucumber", "TestRail", "Appium", "Locust", "Mocha")),
}
_TASKS = ("internal services", "the reporting pipeline", "customer-facing features", "the billing module",
          "the onboarding flow", "the search feature", "scheduled jobs", "the admin tooling")
_METRICS = ("latency", "build time", "error rate", "infrastructure cost", "page load time", "query time")
_DEGREES = {"backend": ("BSc", "Computer Science"), "data": ("MSc", "Statistics"),
            "frontend": ("BSc", "Information Systems"), "devops": ("BTech", "Computer Engineering"),
            "qa": ("BSc", "Software Engineering")}


@dataclass(frozen=True)
class Block:
    style: Literal["name", "contact", "heading", "line", "bullet"]
    text: str
    section: str
    skill: str | None = None  # set on bullets that carry a skill's strongest evidence
    level: EvidenceLevel | None = None
    role_index: int | None = None


@dataclass(frozen=True)
class RoleSpec:
    title: str
    company: str
    start: tuple[int, int]
    end: tuple[int, int] | None  # None = present


@dataclass(frozen=True)
class SyntheticResume:
    resume_id: str
    seed: int
    identity: Identity
    domain: str
    roles: tuple[RoleSpec, ...]
    degree: str
    field_of_study: str
    grad_year: int
    skills: tuple[tuple[str, EvidenceLevel], ...]
    skill_role: dict[str, int] = field(default_factory=dict, compare=False)
    detail: tuple[int, ...] = ()  # per-skill random draw (task / metric / percentage)

    def with_identity(self, **changes: str) -> SyntheticResume:
        """Swap identity fields (name, gender, college, location); everything else is unchanged."""
        return replace(self, identity=Identity(**{**self.identity.model_dump(), **changes}))

    def layout(self) -> list[Block]:
        return _layout(self)

    def to_text(self) -> str:
        return "\n".join(_text_line(b) for b in self.layout()) + "\n"

    def gold_facts(self) -> list[GoldFact]:
        return _gold_facts(self)


def _fmt(ym: tuple[int, int] | None) -> str:
    return "Present" if ym is None else f"{_MONTHS[ym[1] - 1]} {ym[0]}"


def _iso(ym: tuple[int, int] | None) -> str:
    return "present" if ym is None else f"{ym[0]:04d}-{ym[1]:02d}"


def _text_line(b: Block) -> str:
    if b.style == "heading":
        return "\n" + b.text.upper()
    return f"- {b.text}" if b.style == "bullet" else b.text


def _bullet(level: str, skill: str, d: int) -> str:
    task = _TASKS[d % len(_TASKS)]
    if level == "E4":
        return f"Reduced {_METRICS[d % len(_METRICS)]} by {10 + d % 61}% for {task} using {skill}"
    verb = ("Maintained", "Supported", "Worked on", "Wrote code for")[d % 4]
    return f"{verb} {task} with {skill}"


def _layout(r: SyntheticResume) -> list[Block]:
    idn = r.identity
    email = idn.name.lower().replace(" ", ".") + "@example.com"
    phone = f"555-01{int(r.resume_id[-3:]) % 100:02d}"
    out = [Block("name", idn.name, "header"),
           Block("contact", f"{idn.location} | {email} | {phone}", "header"),
           Block("heading", "Skills", "skills"),
           Block("line", ", ".join(s for s, _ in r.skills), "skills"),
           Block("heading", "Experience", "experience")]
    for i, role in enumerate(r.roles):
        out.append(Block("line", f"{role.title}, {role.company} | {_fmt(role.start)} - {_fmt(role.end)}",
                         "experience", role_index=i))
        for (skill, level), d in zip(r.skills, r.detail):
            if level in ("E3", "E4") and r.skill_role[skill] == i:
                out.append(Block("bullet", _bullet(level, skill, d), "experience", skill, level, i))
    certs = [s for s, lv in r.skills if lv == "E2"]
    if certs:
        out.append(Block("heading", "Certifications and Coursework", "certifications"))
        out.extend(Block("bullet", f"Completed professional coursework in {s}", "certifications", s, "E2") for s in certs)
    out.append(Block("heading", "Education", "education"))
    out.append(Block("line", f"{r.degree} {r.field_of_study}, {idn.college}, {r.grad_year}", "education"))
    return out


def _gold_facts(r: SyntheticResume) -> list[GoldFact]:
    facts: list[GoldFact] = []
    blocks = _layout(r)
    for b in blocks:
        if b.section == "experience" and b.style == "line":
            role = r.roles[b.role_index or 0]
            facts.append(GoldFact(kind="role", value=f"{role.title} @ {role.company}", section="experience",
                                  source_text=b.text, role_index=b.role_index))
            facts.append(GoldFact(kind="date_range", value=f"{_iso(role.start)}..{_iso(role.end)}",
                                  section="experience", source_text=b.text, role_index=b.role_index))
        elif b.section == "education" and b.style == "line":
            facts.append(GoldFact(kind="education", value=f"{r.degree} {r.field_of_study} ({r.grad_year})",
                                  section="education", source_text=b.text))
            facts.append(GoldFact(kind="institution", value=r.identity.college, section="education",
                                  source_text=b.text))
    strongest = {b.skill: b for b in blocks if b.skill}
    skills_line = ", ".join(s for s, _ in r.skills)
    for skill, level in r.skills:
        b = strongest.get(skill)
        facts.append(GoldFact(kind="skill", value=skill, evidence_level=level,
                              section=b.section if b else "skills", source_text=b.text if b else skills_line,
                              role_index=b.role_index if b else None))
    return facts


def _ym_add(ym: tuple[int, int], months: int) -> tuple[int, int]:
    n = ym[0] * 12 + ym[1] - 1 + months
    return n // 12, n % 12 + 1


def make_resume(index: int, seed: int = 0) -> SyntheticResume:
    """Build resume ``index`` of the pool for ``seed``. Independent of other indices."""
    rng = random.Random(f"resume:{seed}:{index}")
    gender = rng.choices(["female", "male", "nonbinary"], weights=[45, 45, 10])[0]
    first = rng.choice({"female": _FEMALE, "male": _MALE, "nonbinary": _NONBINARY}[gender])
    identity = Identity(name=f"{first} {rng.choice(_LAST)}", gender=gender,
                        college=rng.choice(_COLLEGES), location=rng.choice(_LOCATIONS))
    domain = rng.choice(sorted(_DOMAINS))
    titles, pool = _DOMAINS[domain]
    n_roles = rng.randint(2, 4)
    end_now = rng.random() < 0.5
    cursor = _ym_add(_REFERENCE_YM, 0 if end_now else -rng.randint(1, 6))
    spans: list[tuple[tuple[int, int], tuple[int, int] | None]] = []
    for k in range(n_roles):  # newest first
        start = _ym_add(cursor, -rng.randint(10, 48))
        spans.append((start, None if (k == 0 and end_now) else cursor))
        cursor = _ym_add(start, -rng.randint(0, 3))
    spans.reverse()
    companies = rng.sample(_COMPANIES, n_roles)
    prefix = ("Junior ", "", "", "Senior ")
    roles = tuple(RoleSpec(f"{prefix[i + (4 - n_roles)]}{rng.choice(titles)}", companies[i], s, e)
                  for i, (s, e) in enumerate(spans))
    chosen = rng.sample(pool, rng.randint(6, 10))
    levels = [rng.choices(["E1", "E2", "E3", "E4"], weights=[30, 15, 30, 25])[0] for _ in chosen]
    degree, fld = _DEGREES[domain]
    return SyntheticResume(
        resume_id=f"S-{index:04d}", seed=seed, identity=identity, domain=domain, roles=roles,
        degree=degree, field_of_study=fld, grad_year=roles[0].start[0] - rng.randint(0, 1),
        skills=tuple(zip(chosen, levels)), skill_role={s: rng.randrange(n_roles) for s in chosen},
        detail=tuple(rng.randrange(1000) for _ in chosen))


def generate_pool(n: int = DEFAULT_POOL_SIZE, seed: int = 0) -> list[SyntheticResume]:
    if not isinstance(n, int) or isinstance(n, bool) or not MIN_POOL_SIZE <= n <= MAX_POOL_SIZE:
        raise ValueError(f"pool size must be an integer in [{MIN_POOL_SIZE}, {MAX_POOL_SIZE}]")
    return [make_resume(i, seed) for i in range(n)]


# ---- rendering ---------------------------------------------------------------------------------

def render_txt(resume: SyntheticResume, path: Path) -> None:
    path.write_text(resume.to_text(), encoding="utf-8")


def render_docx(resume: SyntheticResume, path: Path) -> None:
    from docx import Document  # python-docx, already in uv.lock (docling dependency)

    doc = Document()
    for b in resume.layout():
        if b.style == "name":
            doc.add_heading(b.text, level=0)
        elif b.style == "heading":
            doc.add_heading(b.text, level=1)
        elif b.style == "bullet":
            doc.add_paragraph(b.text, style="List Bullet")
        else:
            doc.add_paragraph(b.text)
    doc.save(str(path))


def render_pdf(resume: SyntheticResume, path: Path) -> None:
    import pymupdf

    doc = pymupdf.open()
    width, height, margin = 595, 842, 54
    page, y = doc.new_page(width=width, height=height), margin
    for b in resume.layout():
        size = {"name": 18, "heading": 13}.get(b.style, 10.5)
        font = "hebo" if b.style in ("name", "heading") else "helv"
        text = f"- {b.text}" if b.style == "bullet" else b.text
        if b.style == "heading":
            y += 6
        lines: list[str] = []
        cur = ""
        for word in text.split():
            trial = f"{cur} {word}".strip()
            if cur and pymupdf.get_text_length(trial, fontname=font, fontsize=size) > width - 2 * margin:
                lines.append(cur)
                cur = word
            else:
                cur = trial
        lines.append(cur)
        for line in lines:
            if y + size > height - margin:
                page, y = doc.new_page(width=width, height=height), margin
            page.insert_text((margin, y + size), line, fontname=font, fontsize=size)
            y += size * 1.4
    doc.save(str(path))
    doc.close()


_RENDERERS = {"txt": render_txt, "docx": render_docx, "pdf": render_pdf}


def safe_output_dir(path: str | Path, base: str | Path | None = None) -> Path:
    """Resolve ``path`` and require it to sit under ``data/eval`` or ``runs`` of ``base`` (default cwd)."""
    root = Path(base).resolve() if base is not None else Path.cwd().resolve()
    target = (root / path).resolve()  # resolve() collapses ".." and follows symlinks
    for allowed in ALLOWED_OUTPUT_ROOTS:
        allowed_root = (root / allowed).resolve()
        if target == allowed_root or allowed_root in target.parents:
            return target
    raise ValueError(f"output path must stay under {' or '.join(ALLOWED_OUTPUT_ROOTS)}: {path}")


def write_pool(pool: list[SyntheticResume], out_dir: str | Path = "data/eval/synthetic",
               formats: tuple[str, ...] = FORMATS, base: str | Path | None = None) -> Path:
    """Render each resume in ``formats`` plus ``gold.jsonl`` and ``index.json`` (with file SHA-256s)."""
    if not formats or set(formats) - set(FORMATS):
        raise ValueError(f"formats must be a non-empty subset of {FORMATS}")
    out = safe_output_dir(out_dir, base)
    out.mkdir(parents=True, exist_ok=True)
    index = []
    with (out / "gold.jsonl").open("w", encoding="utf-8") as gold:
        for r in pool:
            files = {}
            for fmt in formats:
                p = out / f"{r.resume_id}.{fmt}"
                _RENDERERS[fmt](r, p)
                files[fmt] = {"file": p.name, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
            gold.write(json.dumps({"resume_id": r.resume_id, "identity": r.identity.model_dump(),
                                   "domain": r.domain, "facts": [f.model_dump() for f in r.gold_facts()]}) + "\n")
            index.append({"resume_id": r.resume_id, "files": files})
    (out / "index.json").write_text(
        json.dumps({"seed": pool[0].seed if pool else None, "n": len(pool), "resumes": index}, indent=1),
        encoding="utf-8")
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="agentic_talent_screener.evaluation.datasets.synthetic")
    p.add_argument("--n", type=int, default=DEFAULT_POOL_SIZE)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="data/eval/synthetic")
    p.add_argument("--formats", default=",".join(FORMATS))
    a = p.parse_args(argv)
    try:
        out = write_pool(generate_pool(a.n, a.seed), a.out, tuple(a.formats.split(",")))
    except ValueError as exc:
        print(f"error: {exc}")
        return 2
    print(f"wrote {a.n} resumes to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

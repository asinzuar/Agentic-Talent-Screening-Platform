"""Render a clean base layout plus one injection to TXT / DOCX / PDF, and verify hidden channels.

Visibility semantics (documented so nobody mistakes TXT for a visual format):

* TXT is the canonical *text layer*: it always contains the payload, whatever the channel, because a
  plain-text extractor sees hidden text too. Label offsets refer to this text.
* PDF: ``white_text`` is white on the white page, ``tiny_text`` is 1 pt, ``off_page`` is drawn right of
  the page edge (in the content stream, outside the page box), ``behind_image`` is drawn and then
  covered by an opaque white image. Off-page text is only returned by an extractor that is not clipped
  to the page box: use ``extract_pdf_text_unclipped``.
* DOCX: ``white_text`` (white font), ``tiny_text`` (1 pt) and ``hidden_run`` (Word's "hidden" font
  attribute, ``w:vanish``). Off-page and behind-image have no DOCX equivalent here.
"""

from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from ..datasets.synthetic import Block, _text_line
from .families import Built, InjectedSpan

_FIXED_DT = datetime(2026, 1, 1)
_PAGE_W, _PAGE_H, _MARGIN = 595, 842, 54
_TINY_PT = 1.0

# Which formats can carry each channel as a real hidden channel (TXT always holds the text layer).
CHANNEL_FORMATS: dict[str, tuple[str, ...]] = {
    "visible": ("txt", "docx", "pdf"),
    "white_text": ("txt", "docx", "pdf"),
    "tiny_text": ("txt", "docx", "pdf"),
    "off_page": ("txt", "pdf"),
    "behind_image": ("txt", "pdf"),
    "hidden_run": ("txt", "docx"),
}

_SECTION = {"header": "header", "summary": "summary", "footer": "footer", "skills": "skills",
            "experience": "experience", "file": "footer"}


@dataclass(frozen=True)
class Item:
    """One layout block of a twin: a base block (``kind == ""``) or an injected one."""

    block: Block
    channel: str = "visible"
    kind: str = ""  # "", "payload" or "scaffold"
    newpage: bool = False
    nest: int = 0


def inject(layout: list[Block], position: str, built: Built, channel: str) -> list[Item]:
    """Return the base layout with exactly one injection inserted at ``position``."""
    items = [Item(b) for b in layout]
    section = _SECTION[position]
    new: list[Item] = []
    if position == "summary" and channel == "visible":
        new.append(Item(Block("heading", "Summary", "summary"), kind="scaffold"))
    style = "bullet" if position == "experience" else "line"
    new.extend(Item(Block(style, line, section), channel, "payload", built.new_page_per_line, built.nest_depth)
               for line in built.lines)
    if position in ("header", "summary"):
        at = 2  # after name + contact, before the Skills heading
    elif position == "skills":
        at = 1 + max(i for i, b in enumerate(layout) if b.section == "skills")
    elif position == "experience":
        at = 1 + max(i for i, b in enumerate(layout) if b.section == "experience")
    else:  # footer, file
        at = len(items)
    items[at:at] = new
    return items


def render_text(items: list[Item]) -> tuple[str, list[InjectedSpan]]:
    """Canonical text of a twin plus the exact offsets of every injected span."""
    parts: list[str] = []
    spans: list[InjectedSpan] = []
    offset = 0
    for it in items:
        line = _text_line(it.block)
        if it.kind:
            core = it.block.text.upper() if it.block.style == "heading" else line
            start = offset + line.index(core)
            spans.append(InjectedSpan(kind=it.kind, text=core, start=start, end=start + len(core)))  # type: ignore[arg-type]
        parts.append(line)
        offset += len(line) + 1
    return "\n".join(parts) + "\n", spans


# ---- DOCX --------------------------------------------------------------------------------------

def _save_docx(doc, path: Path, title: str) -> None:
    props = doc.core_properties
    props.author = props.last_modified_by = "Synthetic Generator"
    props.created = props.modified = props.last_printed = _FIXED_DT
    props.title, props.comments = title, ""
    buf = io.BytesIO()
    doc.save(buf)
    with zipfile.ZipFile(buf) as src, zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as dst:
        for info in src.infolist():  # fixed timestamps: byte-deterministic output
            fixed = zipfile.ZipInfo(info.filename, date_time=(1980, 1, 1, 0, 0, 0))
            fixed.compress_type, fixed.external_attr = zipfile.ZIP_DEFLATED, info.external_attr
            dst.writestr(fixed, src.read(info.filename))


def render_docx_items(items: list[Item], path: Path, title: str) -> None:
    from docx import Document
    from docx.shared import Pt, RGBColor

    doc = Document()
    for it in items:
        b = it.block
        if it.newpage:
            doc.add_page_break()
        if it.nest:  # bounded nested tables, text in the innermost cell
            cell = doc.add_table(rows=1, cols=1).cell(0, 0)
            for _ in range(it.nest - 1):
                cell = cell.add_table(rows=1, cols=1).cell(0, 0)
            cell.text = b.text
        elif it.channel != "visible":
            run = doc.add_paragraph().add_run(b.text)
            if it.channel == "white_text":
                run.font.color.rgb = RGBColor(255, 255, 255)
            elif it.channel == "tiny_text":
                run.font.size = Pt(_TINY_PT)
            elif it.channel == "hidden_run":
                run.font.hidden = True
        elif b.style == "name":
            doc.add_heading(b.text, level=0)
        elif b.style == "heading":
            doc.add_heading(b.text, level=1)
        elif b.style == "bullet":
            doc.add_paragraph(b.text, style="List Bullet")
        else:
            doc.add_paragraph(b.text)
    _save_docx(doc, path, title)


# ---- PDF ---------------------------------------------------------------------------------------

def render_pdf_items(items: list[Item], path: Path, title: str) -> list[list[float]]:
    """Render ``items``; return ``[page, x0, y0, x1, y1]`` for every injected payload line."""
    import pymupdf

    doc = pymupdf.open()
    white = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 4, 4), False)
    white.set_rect(white.irect, (255, 255, 255))
    page, y = doc.new_page(width=_PAGE_W, height=_PAGE_H), _MARGIN
    boxes: list[list[float]] = []
    for it in items:
        b, ch = it.block, it.channel
        size = {"name": 18, "heading": 13}.get(b.style, 10.5)
        font = "hebo" if b.style in ("name", "heading") else "helv"
        text = f"- {b.text}" if b.style == "bullet" else b.text
        if it.newpage:
            page, y = doc.new_page(width=_PAGE_W, height=_PAGE_H), _MARGIN
        if b.style == "heading":
            y += 6
        if ch == "tiny_text":
            size = _TINY_PT
        x = _MARGIN
        if ch == "off_page":
            x, lines = _PAGE_W + 30, [text]
        else:
            lines, cur = [], ""
            for word in text.split():
                trial = f"{cur} {word}".strip()
                if cur and pymupdf.get_text_length(trial, fontname=font, fontsize=size) > _PAGE_W - 2 * _MARGIN:
                    lines.append(cur)
                    cur = word
                else:
                    cur = trial
            lines.append(cur)
        for line in lines:
            if ch != "off_page" and y + size > _PAGE_H - _MARGIN:
                page, y = doc.new_page(width=_PAGE_W, height=_PAGE_H), _MARGIN
            kw = {"color": (1, 1, 1)} if ch == "white_text" else {}
            page.insert_text((x, y + size), line, fontname=font, fontsize=size, **kw)
            rect = pymupdf.Rect(x, y, x + pymupdf.get_text_length(line, fontname=font, fontsize=size), y + size * 1.4)
            if ch == "behind_image":
                page.insert_image(pymupdf.Rect(rect.x0 - 1, rect.y0 - 1.5, rect.x1 + 1, rect.y1), pixmap=white, keep_proportion=False, overlay=True)
            if it.kind == "payload":
                boxes.append([float(page.number), *(round(v, 2) for v in rect)])
            if ch != "off_page":
                y += size * 1.4
    doc.set_metadata({"producer": "synthetic-generator", "creator": "synthetic-generator", "title": title,
                      "author": "", "subject": "", "keywords": "", "creationDate": "D:20260101000000Z",
                      "modDate": "D:20260101000000Z"})
    doc.save(str(path), no_new_id=True)
    doc.close()
    return boxes


# ---- verification helpers (used by tests and by write-time self-checks) ------------------------

def extract_pdf_text_unclipped(path: Path) -> str:
    """Text layer including text outside the page box (the default extractor drops it)."""
    import pymupdf

    out = []
    with pymupdf.open(str(path)) as doc:
        big = pymupdf.Rect(-5000, -5000, 5000, 5000)
        flags = pymupdf.TEXTFLAGS_TEXT & ~pymupdf.TEXT_MEDIABOX_CLIP
        for page in doc:
            out.append(page.get_text("text", flags=flags, clip=big))
    return "".join(out)


@dataclass
class SpanReport:
    text: str
    size: float
    color: int
    bbox: tuple[float, float, float, float]
    reasons: list[str] = field(default_factory=list)

    @property
    def hidden(self) -> bool:
        return bool(self.reasons)


def inspect_pdf_payload(path: Path, boxes: list[list[float]], payload: str) -> list[SpanReport]:
    """Find the text spans of the payload lines (by recorded bbox) and say why each is not visible.

    Reasons: ``white`` (colour equals the white page), ``tiny`` (size under 4 pt), ``off_page`` (bbox
    outside the page), ``behind_image`` (an opaque image covers it). No reasons means visible.
    """
    import pymupdf

    norm = " ".join(payload.split())
    reports: list[SpanReport] = []
    with pymupdf.open(str(path)) as doc:
        big = pymupdf.Rect(-5000, -5000, 5000, 5000)
        flags = pymupdf.TEXTFLAGS_DICT & ~pymupdf.TEXT_MEDIABOX_CLIP
        for pg, x0, y0, x1, y1 in boxes:
            page = doc[int(pg)]
            target = pymupdf.Rect(x0, y0, x1, y1)
            images = [pymupdf.Rect(i["bbox"]) for i in page.get_image_info()]
            for blk in page.get_text("dict", flags=flags, clip=big)["blocks"]:
                for ln in blk.get("lines", []):
                    for sp in ln["spans"]:
                        rect = pymupdf.Rect(sp["bbox"])
                        if not sp["text"].strip() or not rect.intersects(target):
                            continue
                        if " ".join(sp["text"].split()) not in norm:  # a neighbouring base line, not payload
                            continue
                        rep = SpanReport(sp["text"], sp["size"], sp["color"], tuple(sp["bbox"]))
                        if sp["color"] == 0xFFFFFF:
                            rep.reasons.append("white")
                        if sp["size"] < 4:
                            rep.reasons.append("tiny")
                        if not rect.intersects(page.rect):
                            rep.reasons.append("off_page")
                        if any(img.contains(rect) for img in images):
                            rep.reasons.append("behind_image")
                        reports.append(rep)
    return reports


def inspect_docx_payload(path: Path, text: str) -> list[str]:
    """Reasons the run(s) carrying ``text`` are invisible in Word: ``hidden``, ``white``, ``tiny``."""
    from docx import Document

    reasons: set[str] = set()
    doc = Document(str(path))
    for p in doc.paragraphs:
        for r in p.runs:
            if r.text == text:
                if r.font.hidden:
                    reasons.add("hidden")
                if r.font.color is not None and r.font.color.rgb is not None and str(r.font.color.rgb) == "FFFFFF":
                    reasons.add("white")
                if r.font.size is not None and r.font.size.pt < 4:
                    reasons.add("tiny")
    return sorted(reasons)

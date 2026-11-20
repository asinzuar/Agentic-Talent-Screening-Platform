"""L0 forensic admission and visible-text indexing for resume uploads.

All document content is untrusted.  The module never invokes Office, macros,
embedded content, links, or external programs.  PDF inspection uses PyMuPDF's
native parser and renderer only; image-only documents remain unsupported.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import mimetypes
import re
import unicodedata
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

import pymupdf


# Initial intake limits. Revisit only through a documented security review.
MAX_FILE_BYTES = 10 * 1024 * 1024
MAX_ZIP_MEMBERS = 2_000
MAX_ZIP_UNCOMPRESSED_BYTES = 50 * 1024 * 1024
MAX_ZIP_RATIO = 100
MAX_URL_LENGTH = 2_048
MAX_PDF_PAGES = 20
MAX_EXTRACTED_TEXT_CHARS = 30_000
MAX_SOURCE_SPANS = 5_000
MIN_VISIBLE_FONT_SIZE = 4.0
BACKGROUND_COLOUR_DISTANCE = 24

PDF_SIGNATURE = b"%PDF-"
ZIP_SIGNATURE = b"PK\x03\x04"
URL_PATTERN = re.compile(rb"https?://[^\s<>\"'()\\]+", re.IGNORECASE)
READABLE_ASCII_PATTERN = re.compile(rb"[\x20-\x7e]{8,}")
ZERO_WIDTH_OR_FORMAT = {"\u200b", "\u200c", "\u200d", "\u2060", "\ufeff"}
WORD_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
SECTION_HEADINGS = {
    "experience", "work experience", "professional experience", "employment",
    "projects", "education", "certifications", "skills", "summary",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _issue(code: str, severity: str, detail: str) -> dict[str, str]:
    return {"code": code, "severity": severity, "detail": detail}


def _append_issue_once(issues: list[dict[str, str]], code: str, severity: str, detail: str) -> None:
    if not any(issue["code"] == code for issue in issues):
        issues.append(_issue(code, severity, detail))


def _risk_level(issues: list[dict[str, str]]) -> str:
    severities = {issue["severity"] for issue in issues}
    if "high" in severities:
        return "high"
    if "medium" in severities:
        return "medium"
    if "low" in severities:
        return "low"
    return "none"


def _decision(risk_level: str) -> str:
    return {"high": "quarantine", "medium": "manual_review"}.get(risk_level, "allow")


def _detect_type(header: bytes) -> tuple[str, str]:
    if header.startswith(PDF_SIGNATURE):
        return "pdf", "application/pdf"
    if header.startswith(ZIP_SIGNATURE):
        return "zip", "application/zip"
    return "unknown", "application/octet-stream"


def _extract_urls(data: bytes) -> list[str]:
    """Extract bounded URLs only from readable ASCII runs in untrusted data."""
    urls: set[str] = set()
    for fragment in READABLE_ASCII_PATTERN.findall(data):
        for match in URL_PATTERN.findall(fragment):
            url = match.decode("ascii", "ignore").rstrip(".,;:")
            if len(url) <= MAX_URL_LENGTH:
                urls.add(url)
    return sorted(urls)


def _normalise_text(text: str) -> tuple[str, list[str]]:
    """Return NFKC text and suspicious zero-width/format characters found."""
    normalised = unicodedata.normalize("NFKC", text)
    suspicious = sorted(
        {
            f"U+{ord(char):04X}"
            for char in normalised
            if char in ZERO_WIDTH_OR_FORMAT or unicodedata.category(char) == "Cf"
        }
    )
    return "".join(
        char for char in normalised
        if char not in ZERO_WIDTH_OR_FORMAT and unicodedata.category(char) != "Cf"
    ), suspicious


def _mixed_script(text: str) -> bool:
    scripts: set[str] = set()
    for char in text:
        if not char.isalpha():
            continue
        name = unicodedata.name(char, "")
        for script in ("LATIN", "CYRILLIC", "GREEK"):
            if script in name:
                scripts.add(script)
    return "LATIN" in scripts and bool(scripts - {"LATIN"})


def _section_for(text: str, current: str) -> str:
    candidate = re.sub(r"\s+", " ", text).strip().lower().rstrip(":")
    return candidate if len(candidate) <= 80 and candidate in SECTION_HEADINGS else current


def _visibility_from_signals(
    *, font_size: float | None, foreground: tuple[int, int, int] | None,
    background: tuple[int, int, int] | None, off_page: bool, invisible: bool,
    occluded: bool,
) -> tuple[str, list[str]]:
    reasons: list[str] = []
    if invisible:
        reasons.append("unrendered")
    if font_size is not None and font_size < MIN_VISIBLE_FONT_SIZE:
        reasons.append("font_below_minimum")
    if foreground is not None and background is not None:
        distance = sum((left - right) ** 2 for left, right in zip(foreground, background)) ** 0.5
        if distance <= BACKGROUND_COLOUR_DISTANCE:
            reasons.append("colour_near_background")
    if off_page:
        reasons.append("off_page")
    if occluded:
        reasons.append("occluded")
    if "off_page" in reasons:
        return "off_page", reasons
    if "occluded" in reasons:
        return "occluded", reasons
    if "unrendered" in reasons:
        return "unrendered", reasons
    if reasons:
        return "hidden", reasons
    return "visible", reasons


def _rgb_from_pdf_colour(colour: int | tuple[int, int, int] | None) -> tuple[int, int, int] | None:
    if isinstance(colour, tuple) and len(colour) >= 3:
        return tuple(int(channel) for channel in colour[:3])
    if isinstance(colour, int):
        return ((colour >> 16) & 255, (colour >> 8) & 255, colour & 255)
    return None


def _pixel_colour(pixmap: Any, point: tuple[float, float], page_rect: Any) -> tuple[int, int, int] | None:
    """Sample a native-rendered pixel at a PDF coordinate without saving an image."""
    if page_rect.width <= 0 or page_rect.height <= 0:
        return None
    x = max(0, min(pixmap.width - 1, int((point[0] - page_rect.x0) / page_rect.width * pixmap.width)))
    y = max(0, min(pixmap.height - 1, int((point[1] - page_rect.y0) / page_rect.height * pixmap.height)))
    offset = y * pixmap.stride + x * pixmap.n
    sample = pixmap.samples[offset:offset + pixmap.n]
    if len(sample) < 3:
        return None
    return (sample[0], sample[1], sample[2])


def _background_near_span(pixmap: Any, bbox: tuple[float, float, float, float], page_rect: Any) -> tuple[int, int, int] | None:
    x0, y0, x1, y1 = bbox
    points = ((x0 - 1, y0 - 1), (x1 + 1, y0 - 1), (x0 - 1, y1 + 1), (x1 + 1, y1 + 1))
    colours = [_pixel_colour(pixmap, point, page_rect) for point in points]
    known = [colour for colour in colours if colour is not None]
    if not known:
        return None
    return tuple(sorted(channel[index] for channel in known)[len(known) // 2] for index in range(3))


def _overlaps(left: tuple[float, float, float, float], right: tuple[float, float, float, float]) -> bool:
    return left[0] < right[2] and left[2] > right[0] and left[1] < right[3] and left[3] > right[1]


def _scan_pdf(path: Path, data: bytes, issues: list[dict[str, str]]) -> dict[str, Any]:
    """Inspect PDF bytes and build a visible-text index from native PDF data."""
    lower = data.lower()
    flags = {
        "password_protected": b"/encrypt" in lower,
        "contains_javascript": b"/javascript" in lower or b"/js" in lower,
        "launch_action": b"/launch" in lower,
        "embedded_files": b"/embeddedfile" in lower or b"/filespec" in lower,
        "open_action": b"/openaction" in lower,
    }
    if flags["password_protected"]:
        issues.append(_issue("PDF_ENCRYPTED", "medium", "PDF contains an encryption dictionary."))
    if flags["contains_javascript"]:
        issues.append(_issue("PDF_JAVASCRIPT", "high", "PDF contains JavaScript markers."))
    if flags["launch_action"]:
        issues.append(_issue("PDF_LAUNCH_ACTION", "high", "PDF contains a launch action marker."))
    if flags["embedded_files"]:
        issues.append(_issue("PDF_EMBEDDED_FILE", "high", "PDF contains embedded-file markers."))
    if flags["open_action"]:
        issues.append(_issue("PDF_OPEN_ACTION", "medium", "PDF contains an automatic open action marker."))
    urls = _extract_urls(data)
    result: dict[str, Any] = {**flags, "external_urls": urls[:100], "url_count": len(urls)}
    try:
        with pymupdf.open(path) as document:
            if document.page_count > MAX_PDF_PAGES:
                issues.append(_issue("PDF_TOO_MANY_PAGES", "high", f"PDF exceeds {MAX_PDF_PAGES} page limit."))
            spans: list[dict[str, Any]] = []
            visible_parts: list[str] = []
            section = "other"
            text_offset = 0
            for page_number, page in enumerate(document, start=1):
                if len(spans) >= MAX_SOURCE_SPANS:
                    issues.append(_issue("PDF_TOO_MANY_SPANS", "high", "PDF exceeds the source-span limit."))
                    break
                page_rect = page.rect
                pixmap = page.get_pixmap(matrix=pymupdf.Matrix(1, 1), alpha=False, annots=False)
                bbox_log = page.get_bboxlog()
                text_traces = [
                    {
                        "bbox": tuple(trace["bbox"]),
                        "seqno": int(trace.get("seqno", -1)),
                        "invisible": trace.get("type") == 3 or float(trace.get("opacity", 1.0)) <= 0.01,
                    }
                    for trace in page.get_texttrace()
                ]
                for block in page.get_text("dict").get("blocks", []):
                    if block.get("type") != 0:
                        continue
                    for line in block.get("lines", []):
                        for raw_span in line.get("spans", []):
                            text = raw_span.get("text", "")
                            if not text:
                                continue
                            normalised, zero_width = _normalise_text(text)
                            section = _section_for(normalised, section)
                            bbox = tuple(float(value) for value in raw_span["bbox"])
                            off_page = bbox[2] <= page_rect.x0 or bbox[0] >= page_rect.x1 or bbox[3] <= page_rect.y0 or bbox[1] >= page_rect.y1
                            matching_traces = [trace for trace in text_traces if _overlaps(bbox, trace["bbox"])]
                            trace = max(matching_traces, key=lambda item: item["seqno"], default=None)
                            invisible = bool(trace and trace["invisible"])
                            occluded = bool(trace) and any(
                                index > trace["seqno"]
                                and item[0] in {"fill-image", "fill-shade"}
                                and _overlaps(bbox, tuple(item[1]))
                                for index, item in enumerate(bbox_log)
                            )
                            foreground = _rgb_from_pdf_colour(raw_span.get("color"))
                            background = _background_near_span(pixmap, bbox, page_rect)
                            visibility, reasons = _visibility_from_signals(
                                font_size=float(raw_span.get("size", 0.0)), foreground=foreground,
                                background=background, off_page=off_page, invisible=invisible,
                                occluded=occluded,
                            )
                            span_id = f"p{page_number}:s{len(spans) + 1:04d}"
                            span = {
                                "span_id": span_id,
                                "page": page_number,
                                "text": text,
                                "visible_text": text if visibility == "visible" else "",
                                "location": {"bbox": [round(value, 2) for value in bbox], "char_start": text_offset, "char_end": text_offset + len(text)},
                                "section": section,
                                "visibility": visibility,
                                "visibility_reasons": reasons,
                                "normalisation": {"form": "NFKC", "text": normalised, "zero_width_or_format": zero_width},
                            }
                            spans.append(span)
                            text_offset += len(text) + 1
                            if visibility == "visible":
                                visible_parts.append(text)
                            else:
                                _append_issue_once(issues, "PDF_NONVISIBLE_TEXT", "medium", "PDF contains text excluded from evidence because it is non-visible or suspiciously placed.")
                            if zero_width:
                                _append_issue_once(issues, "UNICODE_ZERO_WIDTH_OR_FORMAT", "medium", "Document contains zero-width or Unicode format characters.")
                            if _mixed_script(text):
                                _append_issue_once(issues, "UNICODE_MIXED_SCRIPT", "low", "Document contains mixed Latin/Greek/Cyrillic text requiring review.")
            visible_text = "\n".join(visible_parts)
            if len(visible_text) > MAX_EXTRACTED_TEXT_CHARS:
                issues.append(_issue("PDF_TEXT_LIMIT", "high", "Visible PDF text exceeds the extraction-character limit."))
            result.update({
                "page_count": document.page_count,
                "span_count": len(spans),
                "visible_text_chars": len(visible_text),
                "spans": spans,
                "sanitised_text": visible_text[:MAX_EXTRACTED_TEXT_CHARS],
            })
    except (pymupdf.FileDataError, RuntimeError, ValueError) as error:
        issues.append(_issue("PDF_PARSE_ERROR", "high", f"PDF could not be safely indexed: {error}"))
    return result


def _scan_text(data: bytes, issues: list[dict[str, str]]) -> dict[str, Any]:
    """Validate a TXT upload is textual data and report bounded external URLs."""
    encoding = "utf-8"
    text: str | None = None
    try:
        if data.startswith((b"\xff\xfe", b"\xfe\xff")):
            encoding = "utf-16"
        text = data.decode(encoding)
    except UnicodeDecodeError:
        issues.append(_issue("TEXT_INVALID_ENCODING", "high", "TXT file is not valid UTF-8 or UTF-16 text."))

    # NUL bytes are expected in UTF-16, but otherwise strongly indicate a
    # binary payload disguised with a .txt extension.
    if encoding == "utf-8" and b"\x00" in data:
        issues.append(_issue("TEXT_BINARY_CONTENT", "high", "TXT file contains NUL bytes."))
    if data.startswith((PDF_SIGNATURE, ZIP_SIGNATURE, b"MZ")):
        issues.append(_issue("TEXT_BINARY_SIGNATURE", "high", "TXT file has a known binary file signature."))

    url_source = text.encode("ascii", "ignore") if text is not None else data
    urls = _extract_urls(url_source)
    spans: list[dict[str, Any]] = []
    visible_parts: list[str] = []
    section = "other"
    text_offset = 0
    for line_number, line in enumerate((text or "").splitlines(), start=1):
        if len(spans) >= MAX_SOURCE_SPANS:
            issues.append(_issue("TEXT_TOO_MANY_SPANS", "high", "TXT exceeds the source-span limit."))
            break
        if not line:
            continue
        normalised, zero_width = _normalise_text(line)
        section = _section_for(normalised, section)
        span_id = f"p0:s{len(spans) + 1:04d}"
        spans.append({
            "span_id": span_id,
            "page": None,
            "text": line,
            "visible_text": line,
            "location": {"line": line_number, "char_start": text_offset, "char_end": text_offset + len(line)},
            "section": section,
            "visibility": "visible",
            "visibility_reasons": [],
            "normalisation": {"form": "NFKC", "text": normalised, "zero_width_or_format": zero_width},
        })
        visible_parts.append(normalised)
        text_offset += len(line) + 1
        if zero_width:
            _append_issue_once(issues, "UNICODE_ZERO_WIDTH_OR_FORMAT", "medium", "Document contains zero-width or Unicode format characters.")
        if _mixed_script(line):
            _append_issue_once(issues, "UNICODE_MIXED_SCRIPT", "low", "Document contains mixed Latin/Greek/Cyrillic text requiring review.")
    visible_text = "\n".join(visible_parts)
    if len(visible_text) > MAX_EXTRACTED_TEXT_CHARS:
        issues.append(_issue("TEXT_LIMIT", "high", "TXT visible text exceeds the extraction-character limit."))
    return {
        "encoding": encoding,
        "external_urls": urls[:100],
        "url_count": len(urls),
        "span_count": len(spans),
        "visible_text_chars": len(visible_text),
        "spans": spans[:MAX_SOURCE_SPANS],
        "sanitised_text": visible_text[:MAX_EXTRACTED_TEXT_CHARS],
    }


def _scan_docx(path: Path, issues: list[dict[str, str]]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "contains_macros": False,
        "embedded_files": False,
        "external_relationships": [],
        "member_count": 0,
        "uncompressed_bytes": 0,
    }
    try:
        with zipfile.ZipFile(path) as archive:
            members = archive.infolist()
            result["member_count"] = len(members)
            result["uncompressed_bytes"] = sum(member.file_size for member in members)
            compressed = sum(member.compress_size for member in members)
            ratio = result["uncompressed_bytes"] / max(compressed, 1)
            result["compression_ratio"] = round(ratio, 2)
            names = [member.filename.lower() for member in members]
            if "[content_types].xml" not in names or "word/document.xml" not in names:
                issues.append(_issue("ZIP_NOT_DOCX", "high", "ZIP file does not have the required DOCX members."))
            if len(members) > MAX_ZIP_MEMBERS:
                issues.append(_issue("ZIP_TOO_MANY_MEMBERS", "high", "Archive exceeds the member-count limit."))
            if result["uncompressed_bytes"] > MAX_ZIP_UNCOMPRESSED_BYTES:
                issues.append(_issue("ZIP_TOO_LARGE", "high", "Archive exceeds the uncompressed-size limit."))
            if ratio > MAX_ZIP_RATIO:
                issues.append(_issue("ZIP_HIGH_RATIO", "high", "Archive compression ratio is suspiciously high."))
            result["contains_macros"] = any("vbaproject.bin" in name for name in names)
            result["embedded_files"] = any(name.startswith("word/embeddings/") for name in names)
            if result["contains_macros"]:
                issues.append(_issue("DOCX_MACRO", "high", "Office macro payload detected."))
            if result["embedded_files"]:
                issues.append(_issue("DOCX_EMBEDDED_OBJECT", "high", "Embedded Office object detected."))

            urls: set[str] = set()
            for member in members:
                if not member.filename.lower().endswith(".rels") or member.file_size > 2_000_000:
                    continue
                payload = archive.read(member)
                if b'TargetMode="External"' in payload or b"TargetMode='External'" in payload:
                    urls.update(_extract_urls(payload))
            result["external_relationships"] = sorted(urls)[:100]
            if urls:
                issues.append(_issue("DOCX_EXTERNAL_RELATIONSHIP", "medium", "DOCX has external relationship targets."))
            try:
                root = ElementTree.fromstring(archive.read("word/document.xml"))
            except (KeyError, ElementTree.ParseError) as error:
                issues.append(_issue("DOCX_XML_INVALID", "high", f"DOCX main document XML could not be indexed: {error}"))
                return result

            spans: list[dict[str, Any]] = []
            visible_parts: list[str] = []
            section = "other"
            text_offset = 0
            for paragraph_number, paragraph in enumerate(root.iter(f"{WORD_NS}p"), start=1):
                for run_number, run in enumerate(paragraph.findall(f"{WORD_NS}r"), start=1):
                    parts: list[str] = []
                    for node in run.iter():
                        local_name = node.tag.rsplit("}", 1)[-1]
                        if local_name == "t" and node.text:
                            parts.append(node.text)
                        elif local_name in {"tab", "br", "cr"}:
                            parts.append(" ")
                    text = "".join(parts)
                    if not text:
                        continue
                    normalised, zero_width = _normalise_text(text)
                    section = _section_for(normalised, section)
                    properties = run.find(f"{WORD_NS}rPr")
                    vanished = properties is not None and properties.find(f"{WORD_NS}vanish") is not None
                    colour_node = properties.find(f"{WORD_NS}color") if properties is not None else None
                    colour = colour_node.get(f"{WORD_NS}val", "") if colour_node is not None else ""
                    white_text = colour.upper() in {"FFFFFF", "FFF", "AUTO"} and colour.upper() != "AUTO"
                    size_node = properties.find(f"{WORD_NS}sz") if properties is not None else None
                    try:
                        font_size = int(size_node.get(f"{WORD_NS}val", "0")) / 2 if size_node is not None else None
                    except ValueError:
                        font_size = None
                    visibility, reasons = _visibility_from_signals(
                        font_size=font_size, foreground=None, background=None, off_page=False,
                        invisible=vanished or white_text, occluded=False,
                    )
                    span_id = f"p0:s{len(spans) + 1:04d}"
                    spans.append({
                        "span_id": span_id,
                        "page": None,
                        "text": text,
                        "visible_text": text if visibility == "visible" else "",
                        "location": {"paragraph": paragraph_number, "run": run_number, "char_start": text_offset, "char_end": text_offset + len(text)},
                        "section": section,
                        "visibility": visibility,
                        "visibility_reasons": reasons,
                        "normalisation": {"form": "NFKC", "text": normalised, "zero_width_or_format": zero_width},
                    })
                    text_offset += len(text) + 1
                    if visibility == "visible":
                        visible_parts.append(normalised)
                    else:
                        _append_issue_once(issues, "DOCX_NONVISIBLE_TEXT", "medium", "DOCX contains text excluded from evidence because it is hidden or suspiciously formatted.")
                    if zero_width:
                        _append_issue_once(issues, "UNICODE_ZERO_WIDTH_OR_FORMAT", "medium", "Document contains zero-width or Unicode format characters.")
                    if _mixed_script(text):
                        _append_issue_once(issues, "UNICODE_MIXED_SCRIPT", "low", "Document contains mixed Latin/Greek/Cyrillic text requiring review.")
                    if len(spans) >= MAX_SOURCE_SPANS:
                        issues.append(_issue("DOCX_TOO_MANY_SPANS", "high", "DOCX exceeds the source-span limit."))
                        break
                if len(spans) >= MAX_SOURCE_SPANS:
                    break
            visible_text = "\n".join(visible_parts)
            if len(visible_text) > MAX_EXTRACTED_TEXT_CHARS:
                issues.append(_issue("DOCX_TEXT_LIMIT", "high", "Visible DOCX text exceeds the extraction-character limit."))
            result.update({
                "span_count": len(spans),
                "visible_text_chars": len(visible_text),
                "spans": spans,
                "sanitised_text": visible_text[:MAX_EXTRACTED_TEXT_CHARS],
            })
    except zipfile.BadZipFile:
        issues.append(_issue("INVALID_ZIP", "high", "File advertises ZIP/DOCX but cannot be read as a ZIP archive."))
    return result


def scan_file(file_path: str | Path) -> dict[str, Any]:
    """Scan a single local file and return a stable, JSON-serializable report."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    size = path.stat().st_size
    with path.open("rb") as stream:
        header = stream.read(16)
    actual_type, detected_mime = _detect_type(header)
    declared_mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    issues: list[dict[str, str]] = []
    if size > MAX_FILE_BYTES:
        issues.append(_issue("FILE_TOO_LARGE", "high", f"File exceeds {MAX_FILE_BYTES} byte limit."))
    suffix = path.suffix.lower()
    expected = {".pdf": "pdf", ".docx": "zip", ".txt": "text"}.get(suffix)
    if expected is None:
        issues.append(_issue("UNSUPPORTED_EXTENSION", "high", "Only .pdf, .docx, and .txt are accepted."))
    elif expected != "text" and expected != actual_type:
        issues.append(_issue("TYPE_MISMATCH", "high", "File signature does not match its extension."))

    details: dict[str, Any] = {}
    # Do not read a very large upload into memory; it is quarantined above.
    if size <= MAX_FILE_BYTES and suffix == ".txt":
        details = _scan_text(path.read_bytes(), issues)
        text_issue_codes = {"TEXT_INVALID_ENCODING", "TEXT_BINARY_CONTENT", "TEXT_BINARY_SIGNATURE"}
        if not any(issue["code"] in text_issue_codes for issue in issues):
            actual_type, detected_mime = "text", "text/plain"
        else:
            issues.append(_issue("TYPE_MISMATCH", "high", "TXT extension does not contain valid text data."))
    elif size <= MAX_FILE_BYTES and actual_type == "pdf":
        details = _scan_pdf(path, path.read_bytes(), issues)
    elif size <= MAX_FILE_BYTES and actual_type == "zip":
        details = _scan_docx(path, issues)

    risk_level = _risk_level(issues)
    return {
        "schema_version": "1.0",
        "scanned_at": datetime.now(timezone.utc).isoformat(),
        "file": {
            "name": path.name,
            "size_bytes": size,
            "sha256": _sha256(path),
            "declared_mime": declared_mime,
            "detected_mime": detected_mime,
            "detected_type": actual_type,
        },
        "risk_level": risk_level,
        "decision": _decision(risk_level),
        "issues": issues,
        "details": details,
    }


def summarize_report(report: dict[str, Any]) -> dict[str, Any]:
    """Return the PII-minimised default representation for CLI/API callers.

    ``scan_file`` deliberately retains the full report for the internal audit
    path and later intake layers.  Callers must opt into that material through
    a restricted audit store or the CLI's explicit ``--debug`` mode.
    """
    issues = report["issues"]
    issue_counts = {severity: 0 for severity in ("low", "medium", "high")}
    for issue in issues:
        issue_counts[issue["severity"]] = issue_counts.get(issue["severity"], 0) + 1
    details = report.get("details", {})
    suspicious_span_ids = [
        span["span_id"]
        for span in details.get("spans", [])
        if span.get("visibility") != "visible" or span.get("normalisation", {}).get("zero_width_or_format")
    ]
    return {
        "schema_version": report["schema_version"],
        "scanned_at": report["scanned_at"],
        "document": {
            "sha256": report["file"]["sha256"],
            "size_bytes": report["file"]["size_bytes"],
            "detected_type": report["file"]["detected_type"],
        },
        "decision": report["decision"],
        "risk_level": report["risk_level"],
        "issue_counts": issue_counts,
        "issue_codes": sorted({issue["code"] for issue in issues}),
        "page_count": details.get("page_count"),
        "span_count": details.get("span_count", 0),
        "visible_text_chars": details.get("visible_text_chars", 0),
        "suspicious_span_ids": suspicious_span_ids,
        "audit_artifact_available": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Non-executing PDF/DOCX/TXT forensic scanner")
    parser.add_argument("file", help="Path to the uploaded resume")
    parser.add_argument("--pretty", action="store_true", help="Indent JSON output")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print the full audit report, including source spans and resume text; use only in an authorised environment.",
    )
    args = parser.parse_args()
    report = scan_file(args.file)
    output = report if args.debug else summarize_report(report)
    print(json.dumps(output, indent=2 if args.pretty else None, sort_keys=True))


if __name__ == "__main__":
    main()

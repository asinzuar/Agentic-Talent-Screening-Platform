from __future__ import annotations

from agentic_talent_screener.resume_intake.contracts import (
    IntakeSecurityDecision,
    SanitisedDocumentView,
    SourceSpan,
)
from agentic_talent_screener.resume_intake.l1_injection import _semantic_messages
from agentic_talent_screener.resume_intake.l2_resume_agent import build_l2_input, validate_agent_output
from agentic_talent_screener.resume_intake.pii_redaction import redact_source_spans


def _span(span_id: str, text: str, section: str = "other") -> SourceSpan:
    return SourceSpan(
        span_id=span_id,
        text=text,
        normalised_text=text,
        visible_text=text,
        section=section,
        visibility="visible",
    )


def _safe_spans() -> tuple[SourceSpan, ...]:
    return (
        _span(
            "p1:s0001",
            "Jane Doe | jane.doe@example.com | +91 98765 43210 | Mumbai, India | https://linkedin.example/jane",
        ),
        _span("p1:s0002", "Built Python data pipelines.", "experience"),
    )


def test_model_safe_spans_keep_name_and_mask_contact_pii() -> None:
    safe_spans, redactions = redact_source_spans(_safe_spans())
    text = "\n".join(span.normalised_text for span in safe_spans)

    assert "Jane Doe" in text
    assert "jane.doe@example.com" not in text
    assert "+91 98765 43210" not in text
    assert "Mumbai, India" not in text
    assert "linkedin.example" not in text
    assert "[EMAIL_REDACTED]" in text
    assert "[PHONE_REDACTED]" in text
    assert "[LOCATION_REDACTED]" in text
    assert "[LINK_REDACTED]" in text
    assert {item.category for item in redactions} == {"email", "phone", "location", "link"}
    assert {item.span_id for item in redactions} == {"p1:s0001"}


def test_l1_semantic_prompt_redacts_raw_source_spans() -> None:
    messages = _semantic_messages("raw fallback", source_spans=_safe_spans())
    prompt = messages[1]["content"]

    for forbidden in ("jane.doe@example.com", "+91 98765 43210", "Mumbai, India", "linkedin.example"):
        assert forbidden not in prompt
    assert "Jane Doe" in prompt
    assert "[EMAIL_REDACTED]" in prompt


def test_l1_semantic_fallback_prompt_redacts_contact_pii() -> None:
    raw = "Jane Doe | jane.doe@example.com | +91 98765 43210 | Mumbai, India"
    messages = _semantic_messages(raw)
    prompt = messages[1]["content"]

    for forbidden in ("jane.doe@example.com", "+91 98765 43210", "Mumbai, India"):
        assert forbidden not in prompt
    assert "[EMAIL_REDACTED]" in prompt


def test_l2_input_redacts_typed_document_view() -> None:
    view = SanitisedDocumentView(
        document_sha256="a" * 64,
        included_spans=_safe_spans(),
        security_decision=IntakeSecurityDecision(risk="low"),
    )
    marked, _ = build_l2_input(view)

    for forbidden in ("jane.doe@example.com", "+91 98765 43210", "Mumbai, India", "linkedin.example"):
        assert forbidden not in marked
    assert "Jane Doe" in marked
    assert "Built Python data pipelines." in marked


def test_l2_input_redacts_legacy_forensic_document_view() -> None:
    marked, _ = build_l2_input({
        "details": {
            "spans": [{
                "span_id": "p1:s0001",
                "visibility": "visible",
                "section": "other",
                "visible_text": "Jane Doe | jane.doe@example.com | +91 98765 43210 | Mumbai, India",
                "normalisation": {"text": "Jane Doe | jane.doe@example.com | +91 98765 43210 | Mumbai, India"},
            }]
        }
    })

    for forbidden in ("jane.doe@example.com", "+91 98765 43210", "Mumbai, India"):
        assert forbidden not in marked
    assert "[LOCATION_REDACTED]" in marked


def test_l2_discards_hallucinated_contact_identity_fields() -> None:
    output = validate_agent_output(
        {
            "identity_envelope": {
                "full_name": "Jane Doe",
                "email": "jane.doe@example.com",
                "phone": "+91 98765 43210",
                "location": "Mumbai, India",
                "links": ["https://linkedin.example/jane"],
            },
            "proposed_claims": [],
            "extraction_quality": {"confidence": 0.8},
        },
        "marker",
    )

    assert output.identity_envelope.full_name == "Jane Doe"
    assert output.identity_envelope.email is None
    assert output.identity_envelope.phone is None
    assert output.identity_envelope.location is None
    assert output.identity_envelope.links == []

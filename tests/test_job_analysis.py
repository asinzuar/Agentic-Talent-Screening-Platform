from __future__ import annotations

import json
from typing import get_args
from unittest.mock import Mock

import pytest

from agentic_talent_screener import job_analysis, llm
from agentic_talent_screener.ranking.schemas import EvidenceLevel, Priority, RequirementKind, RequirementProfile
from agentic_talent_screener.ranking.scorer import ProfileNotApprovedError, score_pool

LANGUAGES = "3+ years using Python (preferred), Java, Go, or another similar language."
EDUCATION = "Bachelor’s degree or equivalent practical experience."
RESPONSIBILITY = "Design and test data pipelines."
LOGISTICS = "Interviews take place over video."
JD = f"Data Engineer\n\nResponsibilities\n{RESPONSIBILITY}\n\nQualifications\n{LANGUAGES}\n{EDUCATION}\n\n{LOGISTICS}"


@pytest.fixture
def model_output() -> dict:
    return {
        "role_title": "Data Engineer",
        "requirements": [
            {
                "statement": "Programming experience",
                "kind": "experience",
                "combine": "any_of",
                "options": [
                    {"text": "Python", "preferred": True}, {"text": "Java"}, {"text": "Go"},
                    {"text": "Another similar language"},
                ],
                "min_years": 3,
                "min_evidence": 3,
                "priority": "M",
                "weight": 8,
                "provenance": "jd_qualification",
                "quote": LANGUAGES,
                "flags": ["Open-ended option: confirm which languages count"],
            },
            {
                "statement": "Bachelor’s degree or equivalent",
                "kind": "education",
                "combine": "all_of",
                "options": [{"text": "Bachelor’s degree"}],
                "substitutes": ["Equivalent practical experience"],
                "priority": "M",
                "weight": 2,
                "provenance": "jd_qualification",
                "quote": EDUCATION,
            },
            {
                "statement": "Data pipeline design and testing",
                "kind": "skill",
                "combine": "all_of",
                "options": [{"text": "Pipeline design"}, {"text": "Pipeline testing"}],
                "min_evidence": 3,
                "priority": "P",
                "weight": 5,
                "provenance": "jd_responsibility",
                "quote": RESPONSIBILITY,
            },
            {
                "statement": "Data quality monitoring",
                "kind": "skill",
                "combine": "all_of",
                "options": [{"text": "Data quality monitoring"}],
                "priority": "P",
                "weight": 2,
                "provenance": "model_suggestion",
                "rationale": "Useful for maintaining reliable pipelines after launch.",
            },
        ],
        "not_used_for_scoring": [LOGISTICS],
    }


def test_drafts_a_reviewable_profile_from_the_full_jd(monkeypatch, model_output) -> None:
    chat = Mock(return_value=model_output)
    monkeypatch.setattr(job_analysis, "chat_json", chat)

    profile = job_analysis.draft_requirement_profile(JD, profile_id="data-eng", version=2, model="test/model")

    request = chat.call_args
    assert request.args[0][0]["role"] == "system"
    assert request.args[0][1]["role"] == "user"
    assert json.loads(request.args[0][1]["content"]) == {"job_description": JD}
    assert request.kwargs["model"] == "test/model"
    assert request.kwargs["schema"]["additionalProperties"] is False
    assert set(request.kwargs["schema"]["properties"]) == {"role_title", "requirements", "not_used_for_scoring"}

    assert (profile.profile_id, profile.version, profile.role_title) == ("data-eng", 2, "Data Engineer")
    assert profile.approved is False
    assert profile.approved_by is None and profile.approved_at is None
    assert profile.not_used_for_scoring == (LOGISTICS,)
    languages, education, responsibility, suggestion = profile.requirements
    assert profile.active_requirements == (languages, education)
    assert languages.combine == "any_of" and languages.options[0].preferred
    assert tuple(option.text for option in languages.options) == ("Python", "Java", "Go", "Another similar language")
    assert languages.min_years == 3 and languages.min_evidence == EvidenceLevel.E3
    assert languages.priority == Priority.MANDATORY and languages.weight == 8
    assert languages.quote == LANGUAGES and "Open-ended" in languages.flags[0]
    assert education.substitutes[0].text == "Equivalent practical experience" and education.min_evidence is None
    assert responsibility.combine == "all_of" and responsibility.quote == RESPONSIBILITY
    assert suggestion.quote is None and suggestion.rationale == model_output["requirements"][3]["rationale"]
    ids = [criterion.option_id for requirement in profile.requirements for criterion in requirement.criteria]
    assert len(ids) == len(set(ids))
    assert RequirementProfile.model_validate_json(profile.model_dump_json()) == profile

    matcher = Mock()
    with pytest.raises(ProfileNotApprovedError, match="not human-approved"):
        score_pool([{}], profile, matcher)
    matcher.match.assert_not_called()


def test_sends_strict_schema_and_extended_budget_via_the_shared_client(tmp_path, monkeypatch, model_output) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("ATS_LLM_MODEL", "qwen/qwen3.8-flash")
    monkeypatch.setenv("ATS_OPENROUTER_BASE_URL", "https://example.invalid")
    response = Mock()
    response.json.return_value = {"choices": [{"message": {"content": json.dumps(model_output)}}]}
    post = Mock(return_value=response)
    monkeypatch.setattr(llm.requests, "post", post)

    profile = job_analysis.draft_requirement_profile(JD)

    assert profile.profile_id.startswith("jd-") and profile.version == 1
    request = post.call_args.kwargs
    payload = request["json"]
    assert payload["model"] == "qwen/qwen3.8-flash"
    assert payload["max_tokens"] == 20_000
    assert request["timeout"] == 360
    assert payload["response_format"]["type"] == "json_schema"
    assert payload["response_format"]["json_schema"]["strict"] is True
    assert "reasoning" not in payload
    assert "tools" not in payload


def test_wire_schema_allows_numeric_min_years_only_for_experience(monkeypatch, model_output) -> None:
    chat = Mock(return_value=model_output)
    monkeypatch.setattr(job_analysis, "chat_json", chat)
    job_analysis.draft_requirement_profile(JD)

    schema = chat.call_args.kwargs["schema"]
    items = schema["properties"]["requirements"]["items"]
    assert "discriminator" not in items and "oneOf" not in items
    branches = [schema["$defs"][ref["$ref"].rsplit("/", 1)[-1]] for ref in items["anyOf"]]
    experience, other = branches
    for branch in branches:
        assert branch["additionalProperties"] is False
        assert "kind" in branch["required"]
        assert "min_years" not in branch["required"]
    assert experience["properties"]["kind"]["const"] == "experience"
    assert experience["properties"]["min_years"]["anyOf"] == [
        {"type": "number", "exclusiveMinimum": 0}, {"type": "null"},
    ]
    assert set(other["properties"]["kind"]["enum"]) == set(get_args(RequirementKind)) - {"experience"}
    assert other["properties"]["min_years"]["type"] == "null"


@pytest.mark.parametrize("kind", ["skill", "education", "leadership", "achievement"])
def test_rejects_non_experience_min_years_even_if_provider_returns_it(monkeypatch, model_output, kind) -> None:
    req = model_output["requirements"][1]
    req.update(kind=kind, min_years=1)
    monkeypatch.setattr(job_analysis, "chat_json", Mock(return_value=model_output))

    with pytest.raises(job_analysis.JobAnalysisError, match=rf"requirements\.1\.{kind}\.min_years"):
        job_analysis.draft_requirement_profile(JD)

    req["min_years"] = None
    assert job_analysis.draft_requirement_profile(JD).requirements[1].min_years is None


@pytest.mark.parametrize(
    ("index", "update", "message"),
    [
        (0, {"quote": "3 years of Python"}, "verbatim excerpt"),
        (2, {"quote": None}, "verbatim excerpt"),
        (2, {"status": "active"}, "invalid draft at requirements.2.skill.status"),
        (3, {"rationale": None}, "suggestions need a rationale"),
        (3, {"quote": LOGISTICS}, "suggestions need a rationale"),
        (3, {"kind": "experience", "min_years": 2}, "suggestions need a rationale"),
        (0, {"combine": "all_of"}, "only makes sense in an any_of group"),
        (0, {"min_years": 0}, "invalid draft at requirements.0.experience.min_years"),
        (0, {"min_evidence": 0}, "invalid draft at requirements.0.experience.min_evidence"),
        (0, {"weight": "8"}, "invalid draft at requirements.0.experience.weight"),
    ],
)
def test_rejects_invalid_requirement_output(monkeypatch, model_output, index, update, message) -> None:
    model_output["requirements"][index].update(update)
    monkeypatch.setattr(job_analysis, "chat_json", Mock(return_value=model_output))

    with pytest.raises(job_analysis.JobAnalysisError, match=message):
        job_analysis.draft_requirement_profile(JD)


@pytest.mark.parametrize(
    ("update", "message"),
    [
        ({"approved": True}, "invalid draft at approved"),
        ({"requirements": []}, "invalid draft at requirements"),
        ({"not_used_for_scoring": ["Interviews are remote."]}, "not_used_for_scoring entries must be verbatim"),
    ],
)
def test_rejects_invalid_profile_output(monkeypatch, model_output, update, message) -> None:
    model_output.update(update)
    monkeypatch.setattr(job_analysis, "chat_json", Mock(return_value=model_output))

    with pytest.raises(job_analysis.JobAnalysisError, match=message):
        job_analysis.draft_requirement_profile(JD)


def test_rejects_blank_jd_before_calling_the_model(monkeypatch) -> None:
    chat = Mock()
    monkeypatch.setattr(job_analysis, "chat_json", chat)

    with pytest.raises(job_analysis.JobAnalysisError, match="must not be blank"):
        job_analysis.draft_requirement_profile(" \n ")
    chat.assert_not_called()


def test_rejects_invalid_json_from_the_model(monkeypatch) -> None:
    monkeypatch.setattr(job_analysis, "chat_json", Mock(side_effect=ValueError("invalid JSON")))

    with pytest.raises(job_analysis.JobAnalysisError, match="did not return a valid JSON draft"):
        job_analysis.draft_requirement_profile(JD)

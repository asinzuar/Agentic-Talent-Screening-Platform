from __future__ import annotations

import json
from unittest.mock import Mock

import pytest

from agentic_talent_screener import llm
from agentic_talent_screener.ranking import LLMMatcher, MatchingAgent, MatchingError, VerifiedProfile, score_pool
from agentic_talent_screener.ranking import matching
from agentic_talent_screener.ranking.sample_data import sample_pool, sample_requirement_profile


def _link(option_id="M1.a", fact_id="c1", relation="equivalent", similarity=1.0):
    return dict(option_id=option_id, fact_id=fact_id, relation=relation, similarity=similarity, rationale="Example link.")


def test_sends_one_delimited_matcher_view_and_only_active_criteria(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("ATS_LLM_MODEL", "qwen/qwen3.8-flash")
    monkeypatch.setenv("ATS_OPENROUTER_BASE_URL", "https://example.invalid")
    response = Mock()
    response.json.return_value = {"choices": [{"message": {"content": json.dumps({"matches": [_link()]})}}]}
    post = Mock(return_value=response)
    monkeypatch.setattr(llm.requests, "post", post)

    profile = sample_requirement_profile()
    candidates = {candidate.candidate_id: candidate for candidate in sample_pool()}
    first, second = candidates["C-0932"], candidates["C-0417"]
    matcher = LLMMatcher()
    assert isinstance(matcher, MatchingAgent)
    for candidate in (first, second):
        result = matcher.match(candidate.matcher_view(), profile)
        assert (result.candidate_id, result.profile_id, result.profile_version) == (
            candidate.candidate_id, profile.profile_id, profile.version,
        )
        assert result.matcher_version == matcher.version == "llm-matcher-v1:qwen/qwen3.8-flash"
        assert result.matches[0].model_dump() == _link()

    assert post.call_count == 2
    markers = []
    for call, candidate in zip(post.call_args_list, (first, second), strict=True):
        payload = call.kwargs["json"]
        assert payload["model"] == "qwen/qwen3.8-flash"
        assert "tools" not in payload and "reasoning" not in payload
        output_format = payload["response_format"]
        assert output_format["type"] == "json_schema" and output_format["json_schema"]["strict"] is True
        schema = output_format["json_schema"]["schema"]
        assert schema["additionalProperties"] is False
        link_schema = schema["$defs"]["_ProposedLink"]
        assert link_schema["additionalProperties"] is False
        assert set(link_schema["required"]) == {"option_id", "fact_id", "relation", "similarity", "rationale"}

        system, user = payload["messages"]
        assert (system["role"], user["role"]) == ("system", "user")
        content = user["content"]
        title, raw_criteria, blank, opening, raw_facts, closing = content.splitlines()
        assert title == "Active requirements:" and not blank
        assert opening.endswith(":CANDIDATE_FACTS_DATA>") and closing == "</" + opening[1:]
        markers.append(opening)
        assert json.loads(raw_facts) == [
            {"fact_id": f.fact_id, "kind": f.kind, "statement": f.statement,
             "evidence_level": f.evidence_level.value, "dates": list(f.dates)}
            for f in candidate.facts
        ]
        criteria = json.loads(raw_criteria)
        assert {c["option_id"] for r in criteria for key in ("options", "substitutes") for c in r[key]} == {
            c.option_id for r in profile.active_requirements for c in r.criteria
        }
        assert all(set(r) == {"kind", "statement", "options", "substitutes"} for r in criteria)
        for value in (first.candidate_id, second.candidate_id, "p2:s0030", "F-0417-1", "Pre-verified by HR"):
            assert value not in content
    assert markers[0] != markers[1]


def test_real_scorer_uses_links_but_owns_evidence_and_credit(monkeypatch) -> None:
    profile = sample_requirement_profile()
    candidate = next(candidate for candidate in sample_pool() if candidate.candidate_id == "C-0101")
    model_output = {"matches": [
        _link(),
        _link("M2.a", "c2", "adjacent", 0.7),
        _link("P4.s", "r1"),
        _link("M3.a", "invented-fact"),
        _link("invented-option", "c1"),
    ]}
    chat = Mock(return_value=model_output)
    monkeypatch.setattr(matching, "chat_json", chat)

    result = score_pool([candidate], profile, LLMMatcher(model="test/model")).results[0]

    assert result.status == "ok"
    assert chat.call_args.kwargs["model"] == "test/model"
    by_id = {entry.requirement_id: entry for entry in result.explanations}
    assert by_id["M1"].deciding is not None
    assert by_id["M1"].deciding.quote == candidate.facts[0].spans[0].quote
    assert by_id["M1"].deciding.declared_level == candidate.facts[0].evidence_level
    assert by_id["M2"].deciding is not None and by_id["M2"].deciding.relation == "adjacent"
    assert by_id["M2"].credit == pytest.approx(0.42)  # E4 × adjacent factor 0.6 × similarity 0.7
    assert by_id["P4"].deciding is not None and by_id["P4"].deciding.option_id == "P4.s"
    assert by_id["P4"].via_substitute
    assert by_id["M5"].credit == 0  # No link is a valid finding, not a protocol error.
    assert any("ignored 2 matcher link(s)" in note for note in result.notes)


def test_rejects_unapproved_or_empty_rules_without_a_model_call(monkeypatch) -> None:
    chat = Mock()
    monkeypatch.setattr(matching, "chat_json", chat)
    matcher = LLMMatcher(model="test/model")
    view = sample_pool()[0].matcher_view()
    profile = sample_requirement_profile()
    inactive = tuple(r.model_copy(update={"status": "inactive"}) for r in profile.requirements)
    for not_matchable in (sample_requirement_profile(approved=False), profile.model_copy(update={"requirements": inactive})):
        with pytest.raises(MatchingError, match="human-approved profile with active requirements"):
            matcher.match(view, not_matchable)
    chat.assert_not_called()


def test_no_facts_yields_no_links_without_a_model_call(monkeypatch) -> None:
    chat = Mock()
    monkeypatch.setattr(matching, "chat_json", chat)
    candidate = VerifiedProfile(candidate_id="C-0001")
    result = LLMMatcher(model="test/model").match(candidate.matcher_view(), sample_requirement_profile())
    assert result.candidate_id == candidate.candidate_id and result.matches == ()
    chat.assert_not_called()


def test_direct_call_cannot_send_a_full_candidate_profile(monkeypatch) -> None:
    chat = Mock()
    monkeypatch.setattr(matching, "chat_json", chat)

    with pytest.raises(MatchingError, match="requires a MatcherView"):
        LLMMatcher(model="test/model").match(sample_pool()[0], sample_requirement_profile())  # type: ignore[arg-type]

    chat.assert_not_called()


@pytest.mark.parametrize(
    "raw",
    [
        {},
        {"matches": [], "candidate_id": "C-0001"},
        {"matches": [{**_link(), "evidence_level": 4}]},
        {"matches": [{**_link(), "quote": "Invented quote."}]},
        {"matches": [{**_link(), "similarity": 1.1}]},
        {"matches": [{**_link(), "similarity": "0.7"}]},
        {"matches": [{**_link(), "relation": "unrelated"}]},
        {"matches": [{**_link(), "rationale": " "}]},
    ],
)
def test_invalid_model_output_is_not_treated_as_no_match(monkeypatch, raw) -> None:
    monkeypatch.setattr(matching, "chat_json", Mock(return_value=raw))
    with pytest.raises(MatchingError, match="invalid matcher response"):
        LLMMatcher(model="test/model").match(sample_pool()[0].matcher_view(), sample_requirement_profile())


def test_provider_error_is_isolated_and_diagnostics_are_not_echoed(monkeypatch) -> None:
    chat = Mock(side_effect=[RuntimeError("private provider payload"), {"matches": []}])
    monkeypatch.setattr(matching, "chat_json", chat)
    first, second = sample_pool()[:2]

    pool = score_pool([first, second], sample_requirement_profile(), LLMMatcher(model="test/model"), shortlist_size=1)

    assert pool.n_errors == 1 and pool.n_scored == 1
    ok, error = pool.results
    assert (ok.candidate_id, ok.status, ok.shortlisted) == (second.candidate_id, "ok", True)
    assert (error.candidate_id, error.status, error.shortlisted) == (first.candidate_id, "error", False)
    assert error.error == "matching failed: MatchingError: OpenRouter did not return usable match data"

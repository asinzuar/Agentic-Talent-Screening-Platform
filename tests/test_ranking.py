import pytest
from pydantic import ValidationError

from agentic_talent_screener.ranking import (
    CandidateMatches,
    OptionMatch,
    ScoringConfig,
    StaticMatcher,
    score_candidate,
    score_pool,
)
from agentic_talent_screener.ranking.sample_data import (
    sample_matcher,
    sample_matches,
    sample_pool,
    sample_requirement_profile,
)
from agentic_talent_screener.ranking.schemas import (
    Criterion,
    Education,
    EvidenceLevel as E,
    Requirement,
    RequirementProfile,
    Role,
    SkillClaim,
    Span,
    VerifiedProfile,
)
from agentic_talent_screener.ranking.scorer import ProfileNotApprovedError

PROFILE = sample_requirement_profile()
MATCHER = sample_matcher()


def _pool():
    return {r.candidate_id: r for r in score_pool(sample_pool(), PROFILE, MATCHER).results}


def _by_id(cid: str) -> VerifiedProfile:
    return next(c for c in sample_pool() if c.candidate_id == cid)


def _matches(cid: str, *links: tuple, profile: RequirementProfile = PROFILE) -> CandidateMatches:
    ms = []
    for option_id, item_id, *rest in links:
        if rest:
            ms.append(OptionMatch(option_id=option_id, item_id=item_id, relation="adjacent", similarity=rest[0]))
        else:
            ms.append(OptionMatch(option_id=option_id, item_id=item_id, relation="equivalent"))
    return CandidateMatches(
        candidate_id=cid, profile_id=profile.profile_id, profile_version=profile.version,
        matcher_version="test", matches=tuple(ms),
    )


def _explanation(result, rid):
    return next(e for e in result.explanations if e.requirement_id == rid)


def _profile(*reqs: Requirement) -> RequirementProfile:
    return RequirementProfile(
        profile_id="t", version=1, role_title="Test", requirements=reqs,
        approved=True, approved_by="hm", approved_at="2026-11-01T00:00:00Z",
    )


def _req(rid: str, *options: str, **kw) -> Requirement:
    base = dict(
        requirement_id=rid, statement=rid, kind="skill",
        options=tuple(Criterion(option_id=f"{rid}.{i}", text=t) for i, t in enumerate(options)),
        priority="M", weight=1.0, provenance="jd_qualification", quote="q",
    )
    return Requirement(**{**base, **kw})


def _claim(cid: str, skill: str, level: E, loc: str = "p1:l1", quote: str = "evidence") -> SkillClaim:
    return SkillClaim(id=cid, skill=skill, evidence_level=level, spans=(Span(location=loc, quote=quote),))


# --------------------------------------------------------------------------------------
# Requirement profile (SPEC §6.2)
# --------------------------------------------------------------------------------------


def test_status_defaults_follow_provenance():
    reqs = {r.requirement_id: r for r in PROFILE.requirements}
    assert reqs["M1"].active
    assert reqs["R1"].provenance == "jd_responsibility" and not reqs["R1"].active
    assert reqs["S1"].provenance == "model_suggestion" and not reqs["S1"].active
    assert {r.requirement_id for r in PROFILE.active_requirements} >= {"M1", "M6", "P4"}


def test_reviewer_can_activate_or_deactivate_any_requirement():
    assert _req("X", "Go", provenance="model_suggestion", quote=None, rationale="r", status="active").active
    assert not _req("Y", "Go", status="inactive").active


def test_inactive_requirements_are_not_scored():
    res = _pool()["C-0932"]
    assert {e.requirement_id for e in res.explanations}.isdisjoint({"R1", "S1"})


@pytest.mark.parametrize(
    "kw",
    [
        {"quote": None},  # JD provenance needs the quote
        {"provenance": "model_suggestion", "quote": None},  # suggestion needs a rationale
        {"min_years": 3},  # only experience requirements have min_years
        {"options": ()},
    ],
)
def test_invalid_requirements_rejected(kw):
    with pytest.raises(ValidationError):
        _req("X", "Go", **kw)


def test_preferred_option_needs_any_of_and_is_unique():
    pref = (Criterion(option_id="a", text="Python", preferred=True), Criterion(option_id="b", text="Go"))
    assert _req("X", options=pref, combine="any_of").options[0].preferred
    with pytest.raises(ValidationError):
        _req("X", options=pref, combine="all_of")
    with pytest.raises(ValidationError):
        _req("X", options=(pref[0], pref[0].model_copy(update={"option_id": "c"})), combine="any_of")


def test_duplicate_ids_rejected():
    with pytest.raises(ValidationError):
        _profile(_req("X", "Go"), _req("X", "Rust"))
    with pytest.raises(ValidationError):
        _profile(_req("X", "Go"), _req("Y", options=(Criterion(option_id="X.0", text="Rust"),)))


def test_unapproved_profile_refused():
    with pytest.raises(ProfileNotApprovedError):
        score_pool(sample_pool(), sample_requirement_profile(approved=False), MATCHER)


# --------------------------------------------------------------------------------------
# Sample pool behaviour
# --------------------------------------------------------------------------------------


def test_ranking_matches_expected_order():
    order = [r.candidate_id for r in score_pool(sample_pool(), PROFILE, MATCHER).results]
    assert order == ["C-0932", "C-0101", "C-0204", "C-0770", "C-0417", "C-0555"]


def test_keyword_stuffer_ranks_below_honest_candidates():
    res = _pool()
    stuffer = res["C-0555"]
    for honest in ("C-0932", "C-0101", "C-0204"):
        assert stuffer.rank > res[honest].rank
    assert stuffer.tier.value == "C"
    assert all(e.credit <= 0.1 for e in stuffer.explanations)


def test_injected_candidate_ranks_low_with_accurate_counterfactual_delta():
    res = _pool()
    inj = res["C-0417"]
    assert inj.flagged and inj.tier.value == "C"
    assert inj.rank > res["C-0204"].rank
    unflagged = _by_id("C-0417").model_copy(update={"security_flags": ()})
    ignoring_flags = score_candidate(unflagged, PROFILE, MATCHER.match(unflagged, PROFILE))
    assert inj.raw_score == pytest.approx(ignoring_flags.fit_score)
    assert inj.counterfactual_delta == pytest.approx(ignoring_flags.fit_score - inj.fit_score, abs=1e-9)
    assert inj.counterfactual_delta > 0.3
    spark = _explanation(inj, "M3")
    assert spark.deciding.effective_level == E.E0 and "L4" in spark.deciding.adjustment


def test_semantic_links_give_credit_keyword_match_would_miss():
    solid = _pool()["C-0101"]
    spark = _explanation(solid, "M3")  # resume says "PySpark"
    assert spark.met and spark.deciding.relation == "equivalent"
    sql = _explanation(solid, "M2")  # resume says "PostgreSQL" (adjacent)
    assert sql.deciding.relation == "adjacent"
    assert sql.credit == pytest.approx(0.6 * 0.9 * 1.0)


def test_substitute_satisfies_education_requirement():
    edu = _explanation(_pool()["C-0204"], "P4")
    assert edu.via_substitute and edu.deciding.option_id == "P4.s"


def test_missing_roles_renormalise_not_zero():
    grad = _by_id("C-0770")
    res = _pool()["C-0770"]
    exp = _explanation(res, "M6")
    assert exp.missing and exp.effective_weight == 0.0
    assert sum(e.effective_weight for e in res.explanations) == pytest.approx(1.0)

    no_roles = grad.model_copy(update={"roles": ()})  # extracted, and there are none
    zeroed = score_candidate(no_roles, PROFILE, MATCHER.match(no_roles, PROFILE))
    assert not _explanation(zeroed, "M6").missing
    assert res.fit_score > zeroed.fit_score
    assert res.unassessed == ("M6",) and zeroed.unassessed == ()


def test_deterministic_and_bounded():
    a = score_pool(sample_pool(), PROFILE, MATCHER)
    b = score_pool(sample_pool(), PROFILE, MATCHER)
    assert a == b
    for r in a.results:
        assert 0.0 <= r.fit_score <= 1.0 and r.shortlisted is None and not r.near_cutoff


def test_shortlist_marks_top_n_and_near_cutoff():
    out = score_pool(sample_pool(), PROFILE, MATCHER, shortlist_size=2)
    by = {r.candidate_id: r for r in out.results}
    assert out.shortlist_size == 2
    assert [r.candidate_id for r in out.results if r.shortlisted] == ["C-0932", "C-0101"]
    assert [r.candidate_id for r in out.results if r.near_cutoff] == ["C-0204", "C-0770", "C-0417"]
    assert by["C-0555"].shortlisted is False and not by["C-0555"].near_cutoff
    with pytest.raises(ValueError):
        score_pool(sample_pool(), PROFILE, MATCHER, shortlist_size=0)


def test_errors_are_never_shortlisted():
    out = score_pool([*sample_pool(), {"junk": 1}], PROFILE, MATCHER, shortlist_size=10)
    assert out.results[-1].status == "error" and out.results[-1].shortlisted is False


def test_equal_scores_break_ties_on_must_haves_met():
    profile = _profile(_req("A", "Go", min_evidence=E.E3))
    # Both earn 0.40 credit. C-0001: equivalent E2 evidence (below min E3, not met).
    # C-0002: adjacent E3 evidence, 0.6 x (8/9) x 0.75 = 0.40 (meets min E3).
    c1 = VerifiedProfile(candidate_id="C-0001", claims=(_claim("x", "Go", E.E2),))
    c2 = VerifiedProfile(candidate_id="C-0002", claims=(_claim("x", "Golang tooling", E.E3),))
    matcher = StaticMatcher({
        "C-0001": _matches("C-0001", ("A.0", "x"), profile=profile).matches,
        "C-0002": _matches("C-0002", ("A.0", "x", 8 / 9), profile=profile).matches,
    })
    cfg = ScoringConfig(gate_floor=1.0)  # gate multiplier is 1.0 either way: isolate the tie-break
    out = score_pool([c1, c2], profile, matcher, cfg)
    a, b = out.results
    assert a.fit_score == pytest.approx(b.fit_score)
    assert [a.candidate_id, b.candidate_id] == ["C-0002", "C-0001"]


# --------------------------------------------------------------------------------------
# Scoring rules
# --------------------------------------------------------------------------------------


def test_e2_plus_claim_without_span_is_capped_to_e1():
    profile = _profile(_req("A", "Spark"), _req("B", "Python"))
    cand = VerifiedProfile(
        candidate_id="C-0001",
        claims=(
            SkillClaim(id="c1", skill="Spark", evidence_level=E.E4),
            _claim("c2", "Python", E.E3, quote="Built Python services"),
        ),
    )
    res = score_candidate(cand, profile, _matches("C-0001", ("A.0", "c1"), ("B.0", "c2"), profile=profile))
    spark, python = _explanation(res, "A").deciding, _explanation(res, "B").deciding
    assert spark.declared_level == E.E4 and spark.effective_level == E.E1
    assert "L3" in spark.adjustment
    assert python.effective_level == E.E3


def test_any_of_takes_best_option_and_all_of_averages():
    any_p = _profile(_req("A", "AWS", "GCP", combine="any_of"))
    all_p = _profile(_req("A", "Docker", "Kubernetes", combine="all_of"))
    cand = VerifiedProfile(
        candidate_id="C-0001", claims=(_claim("c1", "X", E.E4), _claim("c2", "Y", E.E1))
    )
    links = (("A.0", "c2"), ("A.1", "c1"))
    any_r = score_candidate(cand, any_p, _matches("C-0001", *links, profile=any_p))
    all_r = score_candidate(cand, all_p, _matches("C-0001", *links, profile=all_p))
    assert _explanation(any_r, "A").credit == pytest.approx(1.0)
    assert _explanation(all_r, "A").credit == pytest.approx((1.0 + 0.1) / 2)
    assert _explanation(all_r, "A").deciding.option_id == "A.0"  # the weakest option decides


def test_min_years_scales_credit_and_counts_internships_at_half():
    profile = _profile(_req("A", "pipelines", kind="experience", min_years=4, min_evidence=E.E3))
    span = (Span(location="p1:l2", quote="role"),)
    cand = VerifiedProfile(
        candidate_id="C-0001",
        roles=(
            Role(id="r1", kind="job", title="DE", months=24, evidence_level=E.E3, spans=span),
            Role(id="r2", kind="internship", title="Intern", months=12, evidence_level=E.E3, spans=span),
        ),
    )
    res = score_candidate(cand, profile, _matches("C-0001", ("A.0", "r1"), ("A.0", "r2"), profile=profile))
    exp = _explanation(res, "A")
    assert exp.deciding.years == pytest.approx(2.5)  # 24 + 12 * 0.5 months
    assert exp.credit == pytest.approx(0.75 * 2.5 / 4)
    assert not exp.met and "2.5 of 4" in exp.gap


def test_min_evidence_decides_met_and_soft_gate():
    profile = _profile(_req("A", "Python", min_evidence=E.E3), _req("B", "SQL", min_evidence=E.E2))
    cand = VerifiedProfile(
        candidate_id="C-0001", claims=(_claim("c1", "Python", E.E2), _claim("c2", "SQL", E.E2))
    )
    res = score_candidate(cand, profile, _matches("C-0001", ("A.0", "c1"), ("B.0", "c2"), profile=profile))
    assert not _explanation(res, "A").met and _explanation(res, "B").met
    assert res.mandatory_coverage == pytest.approx(0.5)
    assert res.mandatory_gate_multiplier == pytest.approx(0.5 + 0.5 * 0.5 / 0.7)


def test_grounded_education_earns_full_credit():
    profile = _profile(_req("A", "CS degree", kind="education", priority="P"))
    cand = VerifiedProfile(
        candidate_id="C-0001",
        education=(Education(id="e1", level="bachelor", field_of_study="CS",
                             spans=(Span(location="p1:l3", quote="B.Tech CS"),)),),
    )
    res = score_candidate(cand, profile, _matches("C-0001", ("A.0", "e1"), profile=profile))
    assert _explanation(res, "A").credit == 1.0 and _explanation(res, "A").met


# --------------------------------------------------------------------------------------
# Matching Agent output is checked, never trusted
# --------------------------------------------------------------------------------------


def test_matcher_cannot_raise_evidence_level():
    stuffer = _by_id("C-0555")
    res = score_candidate(stuffer, PROFILE, MATCHER.match(stuffer, PROFILE))
    assert _explanation(res, "M1").deciding.effective_level == E.E1  # level comes from the claim


def test_links_to_unknown_options_or_items_are_dropped():
    cand = _by_id("C-0932")
    good = MATCHER.match(cand, PROFILE)
    bad = good.model_copy(update={"matches": good.matches + (
        OptionMatch(option_id="NOPE.a", item_id="c1", relation="equivalent"),
        OptionMatch(option_id="P3.a", item_id="ghost", relation="equivalent"),
    )})
    res = score_candidate(cand, PROFILE, bad)
    assert res.fit_score == score_candidate(cand, PROFILE, good).fit_score
    assert any("ignored 2 matcher link" in n for n in res.notes)


def test_matches_for_another_candidate_or_profile_version_error():
    cand = _by_id("C-0932")
    other = MATCHER.match(_by_id("C-0101"), PROFILE)
    assert score_candidate(cand, PROFILE, other).status == "error"
    stale = MATCHER.match(cand, PROFILE).model_copy(update={"profile_version": 99})
    assert score_candidate(cand, PROFILE, stale).status == "error"


def test_matcher_failure_isolated_to_one_candidate():
    class Flaky(StaticMatcher):
        def match(self, candidate, profile):
            if candidate.candidate_id == "C-0101":
                raise TimeoutError("model unavailable")
            return super().match(candidate, profile)

    out = score_pool(sample_pool(), PROFILE, Flaky(sample_matches()))
    assert out.n_errors == 1 and out.n_scored == len(sample_pool()) - 1
    err = out.results[-1]
    assert err.candidate_id == "C-0101" and "matching failed" in err.error


# --------------------------------------------------------------------------------------
# Candidate data guards
# --------------------------------------------------------------------------------------


def test_identity_fields_and_bad_ids_rejected():
    with pytest.raises(ValidationError):
        VerifiedProfile(candidate_id="C-0001", name="Jane Doe")
    with pytest.raises(ValidationError):
        VerifiedProfile(candidate_id="jane")
    with pytest.raises(ValidationError):
        VerifiedProfile(candidate_id="C-0001", claims=(_claim("x", "A", E.E1), _claim("x", "B", E.E1)))


def test_malformed_candidate_does_not_crash_pool():
    pool = [*sample_pool(), {"candidate_id": "C-9999", "email": "x@y.z"}, {"junk": 1}]
    out = score_pool(pool, PROFILE, MATCHER)
    assert out.n_errors == 2
    assert out.n_scored == len(sample_pool())
    assert [r.status for r in out.results[-2:]] == ["error", "error"]


def test_scoring_config_validation():
    with pytest.raises(ValidationError):
        ScoringConfig(adjacent_factor=1.5)
    with pytest.raises(ValidationError):
        ScoringConfig(weights={"skill_match": 1.0})  # component weights no longer exist


# --------------------------------------------------------------------------------------
# Audit ledger
# --------------------------------------------------------------------------------------

from agentic_talent_screener.ranking import AuditEvent, AuditLedger  # noqa: E402


def _ledger() -> AuditLedger:
    led = AuditLedger()
    led.record_profile_approval(PROFILE)
    led.record_pool_scoring(score_pool(sample_pool(), PROFILE, MATCHER))
    led.record_hr_override("C-0417", "hr-01", "exclude", "hidden-text injection confirmed")
    return led


def test_audit_chain_links_and_verifies():
    led = _ledger()
    ev = led.events
    assert [e.event_type for e in ev] == ["profile_approved", "pool_scored", "hr_override"]
    assert ev[0].prev_hash == "0" * 64
    assert ev[1].prev_hash == ev[0].event_hash and ev[2].prev_hash == ev[1].event_hash
    assert ev[1].details["matcher_version"] == MATCHER.version
    assert led.verify_chain()


def test_audit_tampering_detected():
    led = _ledger()
    led._events[2] = led._events[2].model_copy(update={"actor": "someone-else"})
    assert not led.verify_chain()

    led = _ledger()
    del led._events[1]
    assert not led.verify_chain()


def test_blank_override_reason_rejected():
    led = AuditLedger()
    for bad in ("", "   "):
        with pytest.raises(ValueError):
            led.record_hr_override("C-0417", "hr-01", "override", bad)
    assert len(led) == 0


def test_audit_details_reject_identity_keys():
    with pytest.raises(ValidationError):
        AuditEvent(
            event_id="x", timestamp="2026-01-01T00:00:00Z", event_type="hr_override",
            actor="a", details={"candidate": {"email": "a@b.c"}},
            prev_hash="0" * 64, event_hash="1" * 64,
        )


def test_audit_jsonl_roundtrip():
    led = _ledger()
    back = AuditLedger.from_jsonl(led.to_jsonl())
    assert back.events == led.events
    assert back.verify_chain()
    assert back.head_hash == led.head_hash

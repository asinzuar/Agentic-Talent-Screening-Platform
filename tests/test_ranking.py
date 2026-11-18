import pytest
from pydantic import ValidationError

from agentic_talent_screener.ranking import ScoringConfig, score_candidate, score_pool
from agentic_talent_screener.ranking.config import ComponentWeights
from agentic_talent_screener.ranking.sample_data import (
    sample_catalogue,
    sample_pool,
    sample_requirement_profile,
)
from agentic_talent_screener.ranking.schemas import (
    EvidenceLevel,
    SkillClaim,
    Span,
    VerifiedProfile,
)
from agentic_talent_screener.ranking.scorer import ProfileNotApprovedError

PROFILE = sample_requirement_profile()
CAT = sample_catalogue()


def _pool():
    return {r.candidate_id: r for r in score_pool(sample_pool(), PROFILE, CAT).results}


def _by_id(cid: str) -> VerifiedProfile:
    return next(c for c in sample_pool() if c.candidate_id == cid)


def test_keyword_stuffer_ranks_below_honest_candidates():
    res = _pool()
    stuffer = res["C-0555"]
    for honest in ("C-0932", "C-0101", "C-0204"):
        assert stuffer.rank > res[honest].rank
        assert stuffer.fit_score < res[honest].fit_score
    assert res["C-0932"].rank == 1
    assert res["C-0555"].tier.value == "C"


def test_injected_candidate_ranks_low_with_accurate_counterfactual_delta():
    res = _pool()
    inj = res["C-0417"]
    assert inj.flagged and inj.tier.value == "C"
    assert inj.rank > res["C-0204"].rank
    # delta must equal (score with flags ignored) - (final score)
    unflagged = _by_id("C-0417").model_copy(update={"security_flags": ()})
    ignoring_flags = score_candidate(unflagged, PROFILE, CAT)
    assert inj.counterfactual_delta > 0.1
    assert inj.raw_score == pytest.approx(ignoring_flags.fit_score)
    assert inj.counterfactual_delta == pytest.approx(
        ignoring_flags.fit_score - inj.fit_score, abs=1e-9
    )
    # the attack would have lifted them well above their honest score
    assert inj.raw_score > inj.fit_score


def test_e2_plus_claim_without_span_is_capped_to_e1():
    cand = VerifiedProfile(
        candidate_id="C-0001",
        claims=(
            SkillClaim(skill_id="esco:spark", evidence_level=EvidenceLevel.E4),
            SkillClaim(
                skill_id="esco:python",
                evidence_level=EvidenceLevel.E3,
                spans=(Span(location="p1:l2", quote="Built Python services"),),
            ),
        ),
    )
    res = score_candidate(cand, PROFILE, CAT)
    ex = {e.skill_id: e for e in res.explanations}
    assert ex["esco:spark"].declared_level == EvidenceLevel.E4
    assert ex["esco:spark"].effective_level == EvidenceLevel.E1
    assert "L3" in ex["esco:spark"].adjustment
    assert ex["esco:python"].effective_level == EvidenceLevel.E3


def test_missing_tenure_and_location_renormalise_not_zero():
    base = _by_id("C-0770")
    assert base.location_match is None
    res = score_candidate(base, PROFILE, CAT)
    comps = {c.name: c for c in res.components}
    assert not comps["tenure"].present and not comps["location"].present
    assert comps["tenure"].effective_weight == 0.0
    assert sum(c.effective_weight for c in res.components) == pytest.approx(1.0)
    assert comps["skill_match"].effective_weight > comps["skill_match"].nominal_weight

    zeroed = score_candidate(base.model_copy(update={"location_match": 0.0}), PROFILE, CAT)
    assert res.fit_score > zeroed.fit_score
    # missing data widens the CI
    full = score_candidate(base.model_copy(update={"location_match": 0.5}), PROFILE, CAT)
    assert (res.ci_high - res.ci_low) > (full.ci_high - full.ci_low)


def test_invalid_config_weights_raise():
    with pytest.raises(ValidationError):
        ScoringConfig(weights=ComponentWeights(skill_match=0.5))
    with pytest.raises(ValidationError):
        ScoringConfig(weights=ComponentWeights(location=0.0, tenure=0.0))


def test_identity_fields_and_bad_ids_rejected():
    with pytest.raises(ValidationError):
        VerifiedProfile(candidate_id="C-0001", name="Jane Doe")
    with pytest.raises(ValidationError):
        VerifiedProfile(candidate_id="jane")


def test_unapproved_profile_refused():
    with pytest.raises(ProfileNotApprovedError):
        score_pool(sample_pool(), sample_requirement_profile(approved=False), CAT)


def test_malformed_candidate_does_not_crash_pool():
    pool = [*sample_pool(), {"candidate_id": "C-9999", "email": "x@y.z"}, {"junk": 1}]
    out = score_pool(pool, PROFILE, CAT)
    assert out.n_errors == 2
    assert out.n_scored == len(sample_pool())
    assert [r.status for r in out.results[-2:]] == ["error", "error"]


def test_deterministic_and_bounded():
    a = score_pool(sample_pool(), PROFILE, CAT)
    b = score_pool(sample_pool(), PROFILE, CAT)
    assert a == b
    for r in a.results:
        assert 0.0 <= r.ci_low <= r.fit_score <= r.ci_high <= 1.0

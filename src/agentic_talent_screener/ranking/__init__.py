"""Score-based ranking module (SPEC §6.3, §13)."""

from .audit import AuditEvent, AuditLedger
from .config import DEFAULT_CONFIG, ScoringConfig
from .matching import MatchingAgent, StaticMatcher
from .schemas import (
    CandidateMatches,
    MatcherView,
    OptionMatch,
    PoolRanking,
    RankingResult,
    Requirement,
    RequirementProfile,
    VerifiedProfile,
)
from .scorer import MatchesMismatchError, ProfileNotApprovedError, score_candidate, score_pool

__all__ = [
    "AuditEvent",
    "AuditLedger",
    "CandidateMatches",
    "DEFAULT_CONFIG",
    "MatchesMismatchError",
    "MatcherView",
    "MatchingAgent",
    "OptionMatch",
    "PoolRanking",
    "ProfileNotApprovedError",
    "RankingResult",
    "Requirement",
    "RequirementProfile",
    "ScoringConfig",
    "StaticMatcher",
    "VerifiedProfile",
    "score_candidate",
    "score_pool",
]

"""Score-based ranking module (SPEC §6.4, §13)."""

from .config import DEFAULT_CONFIG, ScoringConfig
from .schemas import (
    PoolRanking,
    RankingResult,
    RequirementProfile,
    SkillCatalogue,
    VerifiedProfile,
)
from .scorer import ProfileNotApprovedError, score_candidate, score_pool

__all__ = [
    "DEFAULT_CONFIG",
    "PoolRanking",
    "ProfileNotApprovedError",
    "RankingResult",
    "RequirementProfile",
    "ScoringConfig",
    "SkillCatalogue",
    "VerifiedProfile",
    "score_candidate",
    "score_pool",
]

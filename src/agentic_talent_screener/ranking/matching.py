"""Semantic Matching Agent interface (SPEC §6.3, §13.3).

The Matching Agent links each requirement option or substitute (free text) to the verified
facts that support it, by meaning rather than by string. It receives a ``MatcherView``: each
fact's kind, statement, evidence level and dates, and nothing else. Quotes (the resume text
around a statement) go to HR and the Explanation Agent, not here (SPEC §13.2). The matcher
cannot set evidence levels or quotes: the scorer takes both from the linked fact and drops
links to unknown options or facts.

The model-backed implementation is pending the model choice (SPEC §11). ``StaticMatcher``
returns fixed links and stands in for it in tests and the sample workbench.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Protocol, runtime_checkable

from .schemas import CandidateMatches, MatcherView, OptionMatch, RequirementProfile


@runtime_checkable
class MatchingAgent(Protocol):
    version: str

    def match(self, candidate: MatcherView, profile: RequirementProfile) -> CandidateMatches:
        """Link the candidate's verified facts to the profile's requirement options."""
        ...


class StaticMatcher:
    """Deterministic stand-in that returns pre-set links per candidate ID."""

    def __init__(
        self, links: Mapping[str, Iterable[OptionMatch]], version: str = "static-matcher-v1"
    ) -> None:
        self.version = version
        self._links = {cid: tuple(ms) for cid, ms in links.items()}

    def match(self, candidate: MatcherView, profile: RequirementProfile) -> CandidateMatches:
        return CandidateMatches(
            candidate_id=candidate.candidate_id,
            profile_id=profile.profile_id,
            profile_version=profile.version,
            matcher_version=self.version,
            matches=self._links.get(candidate.candidate_id, ()),
        )

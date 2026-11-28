"""Read explicit durations from already-verified resume assertions."""

import re
from decimal import Decimal

from .schemas import DurationEvidence, Fact


_NUMBER_WORDS = {
    word: number for number, word in enumerate(
        "zero one two three four five six seven eight nine ten eleven twelve thirteen "
        "fourteen fifteen sixteen seventeen eighteen nineteen twenty".split()
    )
}
_NUMBER = rf"(?:[0-9]{{1,3}}(?:\.[0-9]{{1,3}})?|{'|'.join(_NUMBER_WORDS)})"
_AMOUNT = (
    rf"(?P<reported>(?P<qualifier>at least |over |more than |about |approximately |~\s*)?"
    rf"(?P<lower>{_NUMBER})(?:\s*(?P<plus>\+)|\s*(?:[-–—]|\bto\b)\s*(?P<upper>{_NUMBER}))?"
    r"\s+(?P<unit>years?|yrs?\.?|months?|mos?\.?))"
)
_FORMS = (
    re.compile(rf"{_AMOUNT}\s+(?:of\s+)?experience\s+(?:in|with|using|as)\s+(?P<scope>.+)", re.I),
    re.compile(rf"{_AMOUNT}\s+(?:of\s+)?(?P<scope>.+?)\s+experience", re.I),
    re.compile(rf"(?P<scope>[^:]+):\s*{_AMOUNT}(?:\s+of\s+experience)?", re.I),
)
_AMBIGUOUS_SCOPE = re.compile(
    r"\b(?:including|such as|familiarity with|exposure to|required|preferred|desired|seeking)\b"
    r"|[,;:]|\b(?:years?|yrs?|months?|mos?)\b",
    re.I,
)


def _number(text: str) -> Decimal:
    if text.casefold() in _NUMBER_WORDS:
        return Decimal(_NUMBER_WORDS[text.casefold()])
    return Decimal(text)


def explicit_duration(
    fact: Fact, min_years: float, excluded_locations: frozenset[str]
) -> DurationEvidence | None:
    """Use a narrow literal grammar; all parts of the assertion must still be usable."""
    if fact.kind not in ("experience", "skill") or not fact.spans:
        return None
    if any(
        not span.visible or not span.quote.strip() or span.location in excluded_locations
        for span in fact.spans
    ):
        return None

    statement = " ".join(fact.statement.split()).removesuffix(".")
    cited = " ".join(" ".join(span.quote.casefold().split()) for span in fact.spans)
    # An interior substring can lose a qualifier ("about 12 years") or the
    # first endpoint of a range. Require the assertion at a clause boundary.
    if not re.fullmatch(
        rf"(?:[•▪◦]\s*|[-*] )?(?:i have )?{re.escape(statement.casefold())}"
        r"(?:\.(?: .*)?|, (?:with (?:my )?current role|currently)\b.*)?",
        cited,
    ):
        return None
    if statement.casefold().startswith("i have "):
        statement = statement[len("i have "):]

    for form in _FORMS:
        match = form.fullmatch(statement)
        if match:
            break
    else:
        return None
    scope = match["scope"].strip()
    if not scope or _AMBIGUOUS_SCOPE.search(scope):
        return None

    qualifier, upper = match["qualifier"], match["upper"]
    qualifier = (qualifier or "").strip().casefold()
    if qualifier and (match["plus"] or upper):
        return None
    factor = 12 if match["unit"].lower().startswith("y") else 1
    lower_months = _number(match["lower"]) * factor
    upper_months = _number(upper) * factor if upper else lower_months
    if lower_months <= 0 or upper_months < lower_months:
        return None
    if match["plus"] or qualifier in {"at least", "over", "more than"}:
        upper_months = None
    if qualifier in {"about", "approximately", "~"}:
        lower_months = upper_months = None

    return DurationEvidence(
        reported=match["reported"],
        scope=scope,
        lower_bound_months=float(lower_months) if lower_months is not None else None,
        upper_bound_months=float(upper_months) if upper_months is not None else None,
        meets_minimum=lower_months is not None and lower_months >= Decimal(str(min_years)) * 12,
        evidence_fact=fact.fact_id,
        spans=fact.spans,
    )

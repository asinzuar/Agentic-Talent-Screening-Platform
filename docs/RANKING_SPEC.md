# Module Specification: Score-Based Ranking Engine

> **Module:** `src/agentic_talent_screener/ranking/`  
> **Owner:** Hardik Vyas (Team 30 — AI-ML in Practice, IIT Bombay)  
> **Parent Spec:** `SPEC.md` (§5.1, §6.1–§6.3, §7.1, §9.2, §10, §13)  
> **Status:** Implemented & Unit-Tested (`scoring-v0.4`: explicit, cited durations affect mandatory coverage)

---

## 1. Purpose & Architectural Position

This module implements **Matching, Counterfactual (L4) and Pool-Wide Ranking** in the **Decision Support Lane** (`SPEC.md` §5).

It enforces the project's trust boundary (`SPEC.md` §5.1):
- **Never ingests raw resume text** or unverified claims.
- **Never ingests candidate identity/PII** (blind IDs only, e.g. `C-0932`).
- **No skill catalogue.** Requirements and resume evidence stay free text. A **Semantic Matching Agent** links them by meaning; the scorer doesn't match skills by string.
- **Deterministic scoring.** Given the matcher's links, the scoring maths has zero I/O and zero LLM calls, so it is reproducible and auditable.

```text
VerifiedProfile ─┐
                 ├─► MatchingAgent.match() ─► CandidateMatches ─► scorer ─► RankingResult
RequirementProfile ┘        (semantic, TBD model)     (checked, never trusted)
```

---

## 2. Data Contracts (`schemas.py`)

All inputs and outputs are frozen Pydantic models (`extra="forbid"`).

### 2.1 Inputs

1. **`RequirementProfile`** (Trusted Lane, from JD Analysis): the full `SPEC.md` §6.2 field set.
   - Versioned (`profile_id`, `version`) and **must be human-approved** (`approved`, `approved_by`, `approved_at`). Scoring an unapproved profile raises `ProfileNotApprovedError`.
   - Each `Requirement` has:
     - `requirement_id`, `statement` and `kind` (skill / experience / education / leadership / achievement);
     - free-text `options` combined `all_of` / `any_of`, with an optional `preferred` option;
     - `min_years` (experience only) and `min_evidence`;
     - free-text `substitutes`;
     - `priority` (M / P) and `weight`;
     - three-way `provenance` (`jd_qualification` / `jd_responsibility` / `model_suggestion`) with `quote` or `rationale`;
     - `status` and `flags`.
   - **Status defaults:** JD qualifications start active; responsibilities and model suggestions start **inactive** until a reviewer activates them.
   - The profile also lists JD content `not_used_for_scoring`.

2. **`VerifiedProfile`** (Untrusted Lane output, from the Evidence Verifier via the intake adapter, `RESUME_INTAKE_SPEC.md` §3.4):
   - Identified strictly by `BlindCandidateId` (`^C-\d{4}$`).
   - **PII guardrail:** rejects any payload containing forbidden identity fields (`name`, `email`, `phone`, `gender`, `college`, `university`, `employer`, `linkedin`, etc.).
   - **`facts`:** a flat list of free-text `Fact`s, each with a unique `fact_id`, a `kind` (`skill` / `experience` / `project` / `education` / `certification`), a `statement`, an `evidence_level` (E0–E4), raw `dates` strings (no durations) and cited `Span`s. A `Span` is `location` (`p<page>:s<4 digits>`, e.g. `p1:s0014`), `quote` (the bounded, PII-screened text of the span; never the claim) and `visible`. There are no structured roles, projects or education records: the Matching Agent reads titles, levels and fields from the text.
   - **`extracted_kinds`:** the fact kinds intake attempted to extract (default: all five). A kind that is absent was *not extracted* (missing); a listed kind with no facts means none were found.
   - **`intake_status`** (`verified` / `verified_with_flag`) and **`overall_risk`**: echoed on the ranking result for the reviewer; no effect on the score.
   - **`security_flags`:** one `SecurityFlag` per intake finding (L0 or L1) that cites a span: `attack_family`, `layer`, `risk` (that finding's own severity), `locations[]` (every span ID) and `note`. There is **no quote**: flagged text never crosses into ranking.

3. **`MatchingAgent`** (`matching.py`): interface with `version` and `match(candidate, profile) → CandidateMatches`, where `candidate` is a **`MatcherView`**: per fact, only `fact_id`, `kind`, `statement`, `evidence_level` and `dates`. It never sees quotes, span IDs, security flags, intake status or risk (`SPEC.md` §13.2): quotes can carry college and employer names, so they go to HR and the Explanation Agent only. `score_pool` builds the view with `VerifiedProfile.matcher_view()`.
   - `OptionMatch` links one requirement `option_id` (or substitute) to one verified `fact_id`, with `relation` (`equivalent` / `adjacent`) and `similarity` (0–1).
   - `LLMMatcher` uses OpenRouter to propose structured links for active options and substitutes from an approved profile. It validates locally and adds metadata in code. Calibration remains open (`SPEC.md` §12 Q25). `StaticMatcher` still provides fixed links for tests and the sample workbench (`sample_data.sample_matcher()`).

### 2.2 Outputs
- **`RankingResult`:** per candidate:
  - `fit_score` (0.0–1.0), `raw_score`, `counterfactual_delta`;
  - `tier` (A/B/C), `rank`, `shortlisted`, `near_cutoff`;
  - mandatory coverage and gate, `unassessed` (requirements whose evidence wasn't extracted), and `intake_status` / `overall_risk` echoed from intake;
   - one `RequirementExplanation` per active requirement: credit, met / missing, the deciding criterion with its evidence fact, relation, levels and quote, the detail for every option and substitute (including any cited duration), and a gap.
- **`PoolRanking`:** the ranked pool with summary counts (`n_scored`, `n_errors`, `n_flagged`), the requested `shortlist_size`, and version metadata (`profile_version`, `config_version`, `matcher_version`).

---

## 3. Scoring Methodology (`config.py` & `scorer.py`)

### 3.1 Final Score Formula
The fit score uses the approved profile's **active requirements only**; nothing outside the profile is scored (`SPEC.md` §6.3).

$$\text{Fit Score} = \text{gate} \times \frac{\sum_{r \in \text{present}} w_r \, m_r \, c_r}{\sum_{r \in \text{present}} w_r \, m_r}$$

where $w_r$ is the requirement weight, $m_r = 2.0$ for Mandatory and $1.0$ for Preferred, and $c_r \in [0, 1]$ is the requirement credit.

### 3.2 Requirement Credit

1. **Item credit:** the linked fact's evidence-ladder credit after L3/L4 (`E0 = 0.00`, `E1 = 0.10` (caps keyword stuffing), `E2 = 0.40`, `E3 = 0.75`, `E4 = 1.00`). An `education` fact with a visible span earns `1.0`; the level L3 gave it is ignored.
2. **Relation:** `equivalent` links give the full item credit. `adjacent` links (e.g. PostgreSQL towards SQL) give
   $$c_{\text{adj}} = 0.6 \times \text{similarity} \times \text{item credit}$$
3. **Criterion credit:** the best linked fact. `min_years` does not scale credit. The scorer reads explicit self-reported durations from equivalently linked verified facts and checks that the entire assertion is supported by usable citations. It compares the lower bound to `min_years`. Dates are not used to calculate or add durations (`SPEC.md` §13.5, Q18).
4. **Requirement credit:** `any_of` takes the best option; `all_of` averages the options. The best substitute replaces this if it scores higher.
5. **Met:** credit > 0 and the evidence is at or above `min_evidence`. When `min_years` is set, an option or substitute must also support it as reported. For `all_of`, each option must qualify unless a substitute qualifies. A duration minimum that isn't established keeps the requirement out of the mandatory coverage numerator. The explanation retains the self-report and its citations.

### 3.3 Soft Must-Have Gate
If under `70%` of mandatory weight is **met**, a linear soft penalty applies, down to a floor of `0.50`:
$$\text{gate} = 0.50 + 0.50 \times \frac{\text{coverage}}{0.70}$$
*(Never a hard reject: humans make all final decisions.)*

### 3.4 Tiers
- **Tier A:** `fit_score >= 0.70`
- **Tier B:** `0.50 <= fit_score < 0.70`
- **Tier C:** `fit_score < 0.50`

---

## 4. Defence-in-Depth Enforcement

### Matching Agent output is checked, never trusted
- Output for a different candidate or profile version makes that candidate a `status="error"` result.
- Links to unknown options or facts are dropped and noted.
- Evidence levels, spans and quotes always come from the linked fact, so a link can never raise them.
- A matcher failure is isolated to that candidate as an `error` result.

### L3: Evidence Grounding
- Any fact declared at **`E2` or higher without a cited span** is downgraded to **`E1` (`0.10`)** and logged in the criterion's `adjustment`.
- Facts citing only hidden (`visible=False`) or empty spans drop to **`E0` (`0.00`)**.

### L4: Counterfactual Rescoring
- Spans at any location of a flag of **`medium`** or **`high`** risk (L0/L1) are excluded from the primary pass (`fit_score`).
- A second pass ignores flags (`raw_score`), using the same matcher links:
  $$\text{counterfactual\_delta} = \text{raw\_score} - \text{fit\_score}$$
- This surfaces the score inflation an attack (e.g. candidate `C-0417`'s hidden white text) attempted: `+0.508` on the sample pool.

---

## 5. Ordering, Shortlist & Missing Data

- **Clean ordering:** the pool is sorted by `fit_score`. Equal scores are ordered by `mandatory_coverage` (more must-have weight met first), then `candidate_id`.
- **Shortlist:** `score_pool(..., shortlist_size=N)` marks the top N `shortlisted` and the next `near_cutoff_count` (default 3) `near_cutoff`, the only place a small score gap changes an outcome. Error results are never shortlisted. HR approves or overrides the shortlist.
- **Missing data:** a requirement is missing when it has no credit and its evidence source wasn't extracted (the fact kind is not in `extracted_kinds`: `experience` for experience requirements with min years, `education` for education requirements). It is **never treated as zero**: it's excluded, the weights renormalised over present requirements, and its ID listed in `unassessed` so HR sees the candidate wasn't fully assessed.
- **No confidence intervals or tie bands** (`SPEC.md` §6.3, decided in v0.3). The earlier Monte Carlo interval was built from an assumed noise level rather than measured variation, so it overstated precision and pushed close calls onto HR.

---

## 6. Fault Isolation & Verification

- **Graceful degradation:** in `score_pool()`, a malformed candidate, a matcher failure or a runtime error is returned as `RankingResult(status="error", error=...)` rather than crashing the pool.
- **Verification:** run `uv run pytest` (`tests/test_ranking.py`, `tests/api/test_ranking_api.py`). The tests cover:
  - requirement status defaults, provenance, option and substitute validation, and the approval gate;
  - the sample pool order, the keyword stuffer (`C-0555`) ranking last, and the injected candidate (`C-0417`) with an exact counterfactual delta;
  - semantic links that keyword matching misses ("PySpark" → Spark), adjacent credit (PostgreSQL → SQL) and substitutes;
   - `any_of` / `all_of` combination, explicit duration bounds and complete citations for a minimum-years check, and evidence and duration deciding "met" and the gate;
  - L3 capping of uncited `E2+` facts to `E1`, and unextracted experience facts renormalised and flagged as not assessed (`C-0770`);
  - multi-location flags excluding every cited span, and intake status and risk echoed without changing the score;
  - the top-N shortlist, the near-cut-off marker, errors never shortlisted, and the must-have tie-break;
  - matcher output that can't raise levels, unknown links dropped, mismatched output rejected and matcher failures isolated;
  - rejection of PII fields, plus the audit ledger and API.

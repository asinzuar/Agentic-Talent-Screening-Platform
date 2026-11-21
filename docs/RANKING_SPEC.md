# Module Specification: Score-Based Ranking Engine

> **Module:** `src/agentic_talent_screener/ranking/`  
> **Owner:** Hardik Vyas (Team 30 — AI-ML in Practice, IIT Bombay)  
> **Parent Spec:** `SPEC.md` (§5.1, §6.1–§6.3, §7.1, §9.2, §10, §13)  
> **Status:** Implemented & Unit-Tested (`v0.2`: semantic matching interface, full requirement schema, requirements-only scoring)

---

## 1. Purpose & Architectural Position

This module implements **Matching, Counterfactual (L4) and Pool-Wide Ranking** in the **Decision Support Lane** (`SPEC.md` §5).

It enforces the project's trust boundary (`SPEC.md` §5.1):
- **Never ingests raw resume text** or unverified claims.
- **Never ingests candidate identity/PII** (blind IDs only, e.g. `C-0932`).
- **No skill catalogue.** Requirements and resume evidence stay free text. A **Semantic Matching Agent** links them by meaning; the scorer never compares strings.
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

2. **`VerifiedProfile`** (Untrusted Lane output, from the Evidence Verifier):
   - Identified strictly by `BlindCandidateId` (`^C-\d{4}$`).
   - **PII guardrail:** rejects any payload containing forbidden identity fields (`name`, `email`, `phone`, `gender`, `college`, `university`, `employer`, `linkedin`, etc.).
   - Evidence items, each with a unique `id`:
     - `claims` (free-text `skill`, `evidence_level` E0–E4, cited `Span`s `p<page>:l<line>`);
     - `roles` (job / internship, `title`, `months`, `skills`, `evidence_level`);
     - `projects`;
     - `education` (`level`, `field_of_study`; no institution).
   - `security_flags`.
   - `roles` / `projects` / `education` = `None` means not extracted (missing), as opposed to `()`, which means none.

3. **`MatchingAgent`** (`matching.py`): interface with `version` and `match(candidate, profile) → CandidateMatches`.
   - `OptionMatch` links one requirement `option_id` (or substitute) to one evidence `item_id`, with `relation` (`equivalent` / `adjacent`) and `similarity` (0–1).
   - The model-backed matcher is TBD (`SPEC.md` §12 Q25). `StaticMatcher` returns fixed links and is used for tests and the sample workbench (`sample_data.sample_matcher()`).

### 2.2 Outputs
- **`RankingResult`:** per candidate:
  - `fit_score` (0.0–1.0), `raw_score`, `counterfactual_delta`;
  - `tier` (A/B/C), `rank`, `shortlisted`, `near_cutoff`;
  - mandatory coverage and gate, and `unassessed` (requirements whose evidence wasn't extracted);
  - one `RequirementExplanation` per active requirement: credit, met / missing, the deciding criterion with its evidence item, relation, levels, years and quote, the detail for every option and substitute, and a gap.
- **`PoolRanking`:** the ranked pool with summary counts (`n_scored`, `n_errors`, `n_flagged`), the requested `shortlist_size`, and version metadata (`profile_version`, `config_version`, `matcher_version`).

---

## 3. Scoring Methodology (`config.py` & `scorer.py`)

### 3.1 Final Score Formula
The fit score uses the approved profile's **active requirements only**; nothing outside the profile is scored (`SPEC.md` §6.3).

$$\text{Fit Score} = \text{gate} \times \frac{\sum_{r \in \text{present}} w_r \, m_r \, c_r}{\sum_{r \in \text{present}} w_r \, m_r}$$

where $w_r$ is the requirement weight, $m_r = 2.0$ for Mandatory and $1.0$ for Preferred, and $c_r \in [0, 1]$ is the requirement credit.

### 3.2 Requirement Credit

1. **Item credit:** the linked item's evidence-ladder credit after L3/L4 (`E0 = 0.00`, `E1 = 0.10` (caps keyword stuffing), `E2 = 0.40`, `E3 = 0.75`, `E4 = 1.00`). A degree with a visible span earns `1.0` (education has no ladder level).
2. **Relation:** `equivalent` links give the full item credit. `adjacent` links (e.g. PostgreSQL towards SQL) give
   $$c_{\text{adj}} = 0.6 \times \text{similarity} \times \text{item credit}$$
3. **Criterion credit:** the best linked item. For requirements with `min_years`, credit is scaled by $\min(1, \text{years} / \text{min\_years})$. Years are the months of linked roles that have a visible cited span, with internships at `0.5×`, divided by 12.
4. **Requirement credit:** `any_of` takes the best option; `all_of` averages the options. The best substitute replaces this if it scores higher.
5. **Met:** credit > 0, the deciding evidence is at or above `min_evidence`, and years ≥ `min_years` (every option, for `all_of`).

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
- Links to unknown options or evidence items are dropped and noted.
- Evidence levels, spans and quotes always come from the linked item, so a link can never raise them.
- A matcher failure is isolated to that candidate as an `error` result.

### L3: Evidence Grounding
- Any item declared at **`E2` or higher without a cited span** is downgraded to **`E1` (`0.10`)** and logged in the criterion's `adjustment`.
- Items citing only hidden (`visible=False`) or empty spans drop to **`E0` (`0.00`)**.
- Roles count towards years only with a visible cited span.

### L4: Counterfactual Rescoring
- Spans flagged at **`medium`** or **`high`** risk by L0/L1 are excluded from the primary pass (`fit_score`).
- A second pass ignores flags (`raw_score`), using the same matcher links:
  $$\text{counterfactual\_delta} = \text{raw\_score} - \text{fit\_score}$$
- This surfaces the score inflation an attack (e.g. candidate `C-0417`'s hidden white text) attempted: `+0.508` on the sample pool.

---

## 5. Ordering, Shortlist & Missing Data

- **Clean ordering:** the pool is sorted by `fit_score`. Equal scores are ordered by `mandatory_coverage` (more must-have weight met first), then `candidate_id`.
- **Shortlist:** `score_pool(..., shortlist_size=N)` marks the top N `shortlisted` and the next `near_cutoff_count` (default 3) `near_cutoff`, the only place a small score gap changes an outcome. Error results are never shortlisted. HR approves or overrides the shortlist.
- **Missing data:** a requirement is missing when it has no credit and its evidence source wasn't extracted (`roles` for experience requirements with min years, `education` for education requirements). It is **never treated as zero**: it's excluded, the weights renormalised over present requirements, and its ID listed in `unassessed` so HR sees the candidate wasn't fully assessed.
- **No confidence intervals or tie bands** (`SPEC.md` §6.3, decided in v0.3). The earlier Monte Carlo interval was built from an assumed noise level rather than measured variation, so it overstated precision and pushed close calls onto HR.

---

## 6. Fault Isolation & Verification

- **Graceful degradation:** in `score_pool()`, a malformed candidate, a matcher failure or a runtime error is returned as `RankingResult(status="error", error=...)` rather than crashing the pool.
- **Verification:** run `uv run pytest` (`tests/test_ranking.py`, `tests/api/test_ranking_api.py`). The tests cover:
  - requirement status defaults, provenance, option and substitute validation, and the approval gate;
  - the sample pool order, the keyword stuffer (`C-0555`) ranking last, and the injected candidate (`C-0417`) with an exact counterfactual delta;
  - semantic links that keyword matching misses ("PySpark" → Spark), adjacent credit (PostgreSQL → SQL) and substitutes;
  - `any_of` / `all_of` combination, min-years scaling with internships at half, and min-evidence deciding "met" and the gate;
  - L3 capping of uncited `E2+` claims to `E1`, and missing roles renormalised and flagged as not assessed (`C-0770`);
  - the top-N shortlist, the near-cut-off marker, errors never shortlisted, and the must-have tie-break;
  - matcher output that can't raise levels, unknown links dropped, mismatched output rejected and matcher failures isolated;
  - rejection of PII fields, plus the audit ledger and API.

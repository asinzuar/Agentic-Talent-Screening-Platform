# Module Specification: Score-Based Ranking Engine

> **Module:** `src/agentic_talent_screener/ranking/`  
> **Owner:** Hardik Vyas (Team 30 — AI-ML in Practice, IIT Bombay)  
> **Parent Spec:** `SPEC.md` (§5.1, §6.1–§6.4, §7.1, §9.2, §10)  
> **Status:** Implemented & Unit-Tested (`v0.1`)

---

## 1. Purpose & Architectural Position

This module implements the deterministic **Matching, Counterfactual (L4), and Pool-Wide Ranking** engine in the **Decision Support Lane** (`SPEC.md` §5). 

It enforces the project's non-negotiable trust boundary (`SPEC.md` §5.1):
- **Never ingests raw resume text** or unverified claims.
- **Never ingests candidate identity/PII** (blind IDs only, e.g., `C-0932`).
- **Zero I/O and zero LLM calls:** The scoring math is 100% deterministic, reproducible, and auditable.

---

## 2. Data Contracts (`schemas.py`)

All inputs and outputs are strictly validated using frozen Pydantic models (`extra="forbid"`).

### 2.1 Inputs
1. **`RequirementProfile` (Trusted Lane — from JD Analysis):**
   - Versioned (`profile_id`, `version`) and **must be human-approved** (`approved=True`, `approved_by`, `approved_at`). Scoring against an unapproved profile raises `ProfileNotApprovedError`.
   - Contains `Requirement` items with canonical skill IDs (`esco:<slug>` or `ext:<slug>`), weights (`0.0–10.0`), tags (`M` = Mandatory, `P` = Preferred), and provenance (`jd` with quote, or `suggested` with rationale).
   - Suggested requirements remain inactive until `accepted=True`.

2. **`VerifiedProfile` (Untrusted Lane Output — from Evidence Verifier):**
   - Identified strictly by `BlindCandidateId` (`^C-\d{4}$`).
   - **PII Guardrail:** Explicitly rejects any payload containing forbidden identity fields (`name`, `email`, `phone`, `gender`, `college`, `university`, `employer`, `linkedin`, etc.).
   - Contains `claims` (`SkillClaim` with `EvidenceLevel` `E0–E4` and cited `Span` objects `p<page>:l<line>`), `roles`, `projects`, `education` (level & field relevance only; no institution), `location_match`, and `security_flags`.

3. **`SkillCatalogue`:**
   - Versioned adjacency graph (`AdjacencyEdge`: `has_skill` → `counts_towards` with `similarity` in `[0, 1]`).

### 2.2 Outputs
- **`RankingResult`:** Per-candidate `fit_score` (`0.0–1.0`), 95% confidence interval (`ci_low`, `ci_high`), `raw_score`, `counterfactual_delta`, `tier` (`A`/`B`/`C`), `rank`, `percentile`, `tie_band`, `tie_with`, per-component breakdowns (`ComponentScore`), and per-requirement audit trails (`RequirementExplanation`).
- **`PoolRanking`:** Full ranked pool with summary counts (`n_scored`, `n_errors`, `n_flagged`) and version metadata (`profile_version`, `config_version`, `catalogue_version`).

---

## 3. Scoring Methodology (`config.py` & `scorer.py`)

### 3.1 Final Score Formula
All scores are standardised to **`[0.0, 1.0]`** (matching the HR Dashboard in `SPEC.md` §10):

$$\text{Fit Score} = \text{clip}\!\left(\text{gate} \times \sum_{k \in \text{present}} w'_k \cdot s_k,\; 0,\; 1\right)$$

Where $s_k \in [0, 1]$ is the component sub-score and $w'_k$ is the nominal weight renormalised over present components.

### 3.2 Default Component Weights
| Component | Default Weight | What It Measures |
|---|---|---|
| `skill_match` | **0.45** | Evidence-weighted coverage of active JD skills (direct + adjacent) |
| `experience` | **0.20** | Relevant role duration vs. target years (`4.0` yrs default) |
| `projects` | **0.15** | Span-verified project depth and relevance (`2.0` target) |
| `education` | **0.10** | Degree level (`60%`) blended with field relevance (`40%`) |
| `tenure` | **0.05** | Fairness-bounded average job tenure |
| `location` | **0.05** | Precomputed logistics/relocation fit |

*Validation:* Weights must sum to `1.0` and `skill_match` must be `> 0`.

### 3.3 Component Details

1. **Skill Match (`s_skill`):**
   - **Evidence Ladder Credits (E0–E4):** `E0 = 0.00`, `E1 = 0.10` (caps keyword stuffing), `E2 = 0.40`, `E3 = 0.75`, `E4 = 1.00`.
   - **Mandatory Multiplier:** `M` skills are weighted **`2.0×`** relative to `P` skills.
   - **Adjacent Skills:** If a candidate holds an adjacent skill (e.g., `esco:postgresql` → `esco:sql`), credit is:
     $$\text{credit}_{\text{adj}} = 0.6 \times \text{similarity} \times \text{credit}(\text{best adjacent level})$$
   - **Soft Must-Have Gate:** If `< 70%` of mandatory weight is covered at `E2+`, a linear soft penalty applies down to a floor of `0.50`:
     $$\text{gate} = 0.50 + 0.50 \times \frac{\text{coverage}}{0.70}$$
     *(Never a hard reject—humans make all final decisions.)*

2. **Experience (`s_exp`):**
   - Counts only roles with grounded visible spans. Internships are weighted at `0.5×` months. Scaled by skill overlap with JD requirements and capped at `1.0` at `4.0` relevant years.

3. **Projects (`s_proj`):**
   - Sums `evidence_credit × relevance` across span-verified projects, normalised against a target of `2.0`.

4. **Education (`s_edu`):**
   - Level scores: `secondary=0.30`, `diploma=0.50`, `bachelor=0.70`, `master=0.85`, `doctorate=1.00`, blended `60/40` with `field_relevance`. Institution/college name is excluded for fairness.

5. **Tenure (`s_tenure` — Fairness-Bounded):**
   - Internships and employment gaps are ignored.
   - If a candidate has **`< 3` job roles**, tenure is **not scored** (treated as missing and weight renormalised) to avoid penalising early-career applicants.
   - Floor is bounded at `0.60`, reaching `1.00` at `24` months average tenure.

6. **Location (`s_loc`):**
   - Uses precomputed `location_match` in `[0, 1]`.

### 3.4 Tiers
- **Tier A:** `fit_score >= 0.70`
- **Tier B:** `0.50 <= fit_score < 0.70`
- **Tier C:** `fit_score < 0.50`

---

## 4. Defence-in-Depth Enforcement (L3 & L4)

### L3: Evidence Grounding
- Any claim declared at **`E2` or higher without a cited visible span** is automatically downgraded to **`E1` (`0.10`)** and logged in `RequirementExplanation.adjustment`.
- Claims citing only hidden (`visible=False`) or empty spans are downgraded to **`E0` (`0.00`)**.

### L4: Counterfactual Rescoring
- Spans flagged at **`medium`** or **`high`** risk by L0/L1 are excluded from the primary pass (`fit_score`).
- The engine runs a second pass ignoring flags (`raw_score`) and reports:
  $$\text{counterfactual\_delta} = \text{raw\_score} - \text{fit\_score}$$
- This surfaces the exact score inflation an adversarial attack (e.g., Candidate `C-0417`'s hidden white text) attempted.

---

## 5. Missing Data, Uncertainty & Tie-Bands

- **Missing Data:** Missing components (`None`) are **never treated as zero**. Nominal weights are renormalised over present components, and the uncertainty interval widens.
- **95% Confidence Interval:** Computed via a **200-run seeded Monte Carlo** simulation (`mc_seed` + `CRC32(candidate_id)` for determinism). Present components are perturbed with $\sigma = 0.05$; missing components are sampled around the present-component mean with $\sigma = 0.20$.
- **Tie-Bands:** Candidates whose 95% CIs overlap are grouped into `tie_band` connected components using Union-Find, preventing false precision in HR shortlists.

---

## 6. Fault Isolation & Verification

- **Graceful Degradation:** In `score_pool()`, any malformed candidate dict or runtime error is caught and returned as a `RankingResult(status="error", error=...)` rather than crashing the pool.
- **Verification:** Run `uv run pytest` (`tests/test_ranking.py` — 9 unit tests passing) covering:
  - Keyword stuffer (`C-0555`) ranking below honest candidates (`C-0932`, `C-0101`).
  - Injected candidate (`C-0417`) ranking low with a positive L4 counterfactual delta (`+0.481`).
  - L3 auto-capping of uncited `E2+` claims to `E1`.
  - Missing-component weight renormalisation and widened CIs (`C-0770`).
  - Rejection of PII fields, unapproved profiles, and invalid config weights.
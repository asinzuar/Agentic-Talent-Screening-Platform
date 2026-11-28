# Deterministic scoring engine

How the pure, deterministic score-based ranking engine works.

- **Code:** `src/agentic_talent_screener/ranking/scorer.py` (engine) and
  `src/agentic_talent_screener/ranking/config.py` (frozen defaults).
- **Scope:** matching is *not* done here. The Semantic Matching Agent links
  requirement options to verified facts; this engine checks those links against
  the verified profile, takes evidence levels and quotes from the linked facts
  themselves, and produces a fit score over the approved requirement profile only.
- **Guarantee:** no I/O, no LLM calls, no global state. Identical inputs always
  give identical outputs.

## Inputs

| Input | What it carries |
|---|---|
| `VerifiedProfile` | L3-verified facts: `fact_id`, `kind`, `statement`, `evidence_level` (E0–E4), and `spans` with the visible quote + location. |
| `RequirementProfile` | Human-approved, active requirements: `weight`, `priority` (M/P), `combine`, `min_evidence`, options/substitutes. |
| `CandidateMatches` | Matching Agent links: `option_id`, `fact_id`, `relation` (equivalent/adjacent), `similarity` (0–1). |
| `ScoringConfig` | Frozen, versioned config (`config_version = "scoring-v0.4"`). |

## Step 0 — trust checks

- The profile must be **human-approved** and have active requirements
  (`_require_approved`), otherwise the candidate errors.
- Links must belong to this candidate and this profile version
  (`_validate_matches`). Any link to an unknown option or fact is **dropped and
  counted**.

## Step 1 — evidence grounding per fact (L3/L4)

`_evaluate_fact` computes an **effective** evidence level and credit.

- **Evidence ladder credits (`LadderCredits`):**

  | Level | Credit | Meaning |
  |---|---|---|
  | E0 | 0.00 | no usable evidence |
  | E1 | 0.10 | "listed only" — keyword stuffing is capped here |
  | E2 | 0.40 | |
  | E3 | 0.75 | |
  | E4 | 1.00 | |

- **Grounding caps:** a fact must cite a **visible span with a non-empty quote**
  to keep its declared level. If none of its spans are usable (hidden/empty) it
  drops to **E0**; if it declared E2+ but cites no span at all it is capped to
  **E1**.
- **L4 flag exclusion:** spans whose location matches a **medium/high** security
  flag (`exclude_flag_risks = ("medium", "high")`) are excluded from credit.
- **Education has no ladder:** a grounded education fact earns full credit
  (`education_credit = 1.0`); otherwise 0.

## Step 2 — per-criterion credit

For each option, over all matching links:

```
credit = fact_credit × factor
factor = 1.0                if relation == "equivalent"
factor = 0.6 × similarity   if relation == "adjacent"    # adjacent_factor = 0.6
```

The best link is kept (highest `credit`, tie-break on effective level), capped at
1.0 and rounded to 6 decimals.

```
met = (credit > 0) and (effective level ≥ requirement.min_evidence)
```

Years (`min_years`) are **not** scored yet (SPEC §13.5 / Q18) — they only add a
"years not verified" gap note.

## Step 3 — per-requirement aggregation

- `combine = "any_of"` (or a single option): take the **max** option.
- `combine = "all"`: **mean** of option credits, and `met` requires **all**
  options met.
- **Substitutes** can override: if the best substitute's `(credit, met)` beats the
  options', it decides the requirement (`via_substitute`).
- If credit is 0 and the required fact kind was never extracted → `missing`.

## Step 4 — weighted fit score (renormalised)

```
mult(r) = 2.0 if priority == M (mandatory) else 1.0   # mandatory_multiplier = 2.0

score = Σ weight·mult·credit / Σ weight·mult          # over requirements with data
```

Missing requirements are dropped from the pool and their weights **renormalised**
across the rest (with a note), so an unextracted requirement does not silently
punish the candidate.

## Step 5 — soft must-have gate

```
coverage = mandatory weight met / mandatory weight present

if coverage < 0.70:  gate = 0.50 + 0.50 × (coverage / 0.70)   # gate_floor 0.50, threshold 0.70
else:                gate = 1.0

point = min(1.0, score × gate)
```

A candidate weak on must-haves gets a sliding penalty down to a 0.50 floor rather
than a hard 0.

## Step 6 — two-pass score and output

`score_candidate` runs the pass **twice**:

1. `exclude=True` — flagged spans excluded (the **final** score).
2. `exclude=False` — raw score as if no flags existed.

It returns:

| Field | Meaning |
|---|---|
| `fit_score` | final score (flagged spans excluded) |
| `raw_score` | score with no flag exclusion |
| `counterfactual_delta` | `raw − final` — how much the flags cost |
| `tier` | A (≥ 0.70), B (≥ 0.50), else C |
| `mandatory_coverage` | weighted fraction of mandatory requirements met |
| `mandatory_gate_multiplier` | the gate applied in step 5 |
| `explanations` | per-requirement breakdown |
| `unassessed` | requirement IDs with no extracted facts |

## Ranking (`score_pool`)

Every candidate is matched → scored; malformed candidates get an `error` result
(never crash the pool). Results sort by:

```
(-fit_score, -mandatory_coverage, candidate_id)
```

…so ties break on more must-have coverage, then blind ID (determinism — no
randomness). With `shortlist_size = N`:

- top N → `shortlisted`,
- next `near_cutoff_count = 3` → `near_cutoff`, so HR can see who just missed out,
- errors are never shortlisted and route to manual review.

## Why it matters

The scorer only reuses what L3 already verified (evidence level + quote), so:

- mention-only keywords max out at **E1 credit (0.10)**,
- flagged spans are excluded from the final score,
- every output is reproducible,

which is what makes the adversarial-resume counterfactual (`raw − final`) measurable.

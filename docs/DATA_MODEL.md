# SQLite Data Model: From Job Description to Shortlist

> Persistence model for [Web UI Flows](WEB_UI_FLOWS.md) and [Web UI API Design](WEB_UI_API.md). SQLite is the database for the target workflow. See the [SQL schema](../src/agentic_talent_screener/workflow/schema.sql) for exact column definitions and constraints; [the project spec](../SPEC.md) governs workflow and scoring.

```text
Job → JD → Draft revision → Approved requirements
Job → Candidate → Intake attempt → Verified profile
Approved requirements + Verified profiles → Scoring run → Ranking → Shortlist approval
```

## 1. Conventions

- **One job is one applicant pool.** Job-owned tables include `job_id`. Keys and foreign keys for job-owned records carry the job scope, and attempt/evidence links also carry the candidate scope. For example, an intake attempt cannot point to another applicant's document. Each composite FK targets a primary or unique key on the same columns.
- **SQLite types.** Use `TEXT` for server-generated IDs and fixed-format UTC timestamps, `INTEGER` for counters and booleans (`0`/`1`), `REAL` for scores, and `BLOB` for uploaded bytes. A `?` below marks a nullable field. Use `STRICT` tables, `NOT NULL`, `CHECK` constraints for enums and ranges, and `PRAGMA foreign_keys = ON` on each connection.
- **Use columns for joins, state, and ordering.** Store nested module payloads as schema-versioned JSON `TEXT`, checked with `json_valid()` and validated against the relevant application schema on write and read. A JSON payload never substitutes for a foreign key that controls ownership, eligibility, or shortlist membership.
- **Keep history by reference.** Draft edits are new revisions; approvals, verified profiles, successful scores, and published rankings are immutable once written. Read mutable progress from the current workflow records. Default foreign keys restrict deletion of referenced history; retention policy and administrative deletion are separate decisions.

## 2. Jobs, access, and requirements

| Table | Key fields and relationships | Purpose |
|---|---|---|
| `actors` | PK `actor_id`; `issuer`, `subject`, `disabled_at?`. Unique authenticated `(issuer, subject)`; reserve an actor for system work. | Stable attribution for HR and worker actions. No login credentials in workflow tables. |
| `jobs` | PK `job_id`; unique `profile_id`, `created_by`, `created_at`; `active_profile_version?`, `current_ranking_id?`, `last_approved_shortlist_id?` reference records in this job. | Job workspace and pointers to the active/approved records. The title comes from the active profile or, before approval, the draft. |
| `job_permissions` | PK `(job_id, actor_id, permission)`; FK to `jobs` and `actors`. | Job-scoped grants for listing jobs and authorizing actions. Which roles receive which permissions remains open. |
| `job_descriptions` | PK `(job_id, jd_id)`; `media_type`, `size_bytes`, `sha256`, `source_bytes?`, `extracted_text?`, `uploaded_by`, `uploaded_at`. | Store the accepted JD bytes (nullable after purge) and the exact text used for quote validation. Extraction sets the text once; replacing the JD creates a new row. |
| `requirement_drafts` | PK `(job_id, draft_id)`; `jd_id`, `base_version?`, `state`, `stage?`, `error_json?`, `revision_token`, `current_revision_no?`, creation and lease fields. | Current drafting progress; state is `queued`, `running`, `ready`, `failed`, or `approved`. An initial retry may update the JD reference. A revision starts from the active profile while that profile stays active. |
| `requirement_draft_revisions` | PK `(job_id, draft_id, revision_no)`; `jd_id`, `role_title`, `content_json`, `saved_by`, `saved_at`. | Immutable generated draft or HR edit. `content_json` holds criteria, options and substitutes, provenance, flags and dispositions, and non-scoring passages. Stable criterion/flag IDs let audit records identify activations and removals. |
| `requirement_profiles` | PK `(job_id, version)`; FK `(job_id, draft_id, revision_no)` to the exact reviewed revision; `approved_by`, `approved_at`. | Locked profile. The API's `profile_id` comes from `jobs`; `version` is assigned on approval. The source JD and criteria are read from the pinned revision. |

A unique partial index on `requirement_drafts(job_id)` where `state <> 'approved'` enforces one open draft. A save checks `revision_token` (the API's `expected_revision`), appends a revision and advances the pointer. The token changes on edits and lifecycle transitions and is never reused across drafts. Approval validates the exact current revision, inserts the next version, closes the draft, updates `active_profile_version`, and appends the approval event in one transaction. It invalidates any current ranking on the previous basis without changing that ranking or the last approved shortlist.

## 3. Resume intake and review

| Table | Key fields and relationships | Purpose |
|---|---|---|
| `resume_batches` | PK `(job_id, batch_id)`; `confirmed_profile_version` FK to the approved profile, `uploaded_by`, `uploaded_at`. | One upload request. The profile version records the confirmation for HR; it is not an intake input. |
| `candidates` | PK `(job_id, candidate_id)`; `current_attempt_id` FK to an attempt for the same candidate, `revision_token`, `created_at`. | One blind ID in the job, even after a retry or replacement. Assign IDs collision-free within the job. |
| `resume_documents` | PK `(job_id, candidate_id, document_id)`; `batch_id`, `upload_position`, `sha256`, `size_bytes`, `media_type`, `created_at`. Unique `(job_id, batch_id, upload_position)`. | Metadata for one uploaded resume. A retry reuses the document; a replacement creates one. The batch list reads each document's original attempt. |
| `resume_payloads` | PK/FK `(job_id, candidate_id, document_id)`; `source_bytes`. | Restricted bytes for the intake worker. An upload writes the bounded BLOB and metadata before returning `202`. No original filename is needed for the screening views. |
| `intake_attempts` | PK `(job_id, candidate_id, attempt_id)`; `document_id`, `prior_attempt_id?`, `trigger_kind` (`upload`, `retry`, `replacement`), `requested_by`, `state`, `stage`, `outcome?`, `overall_risk?`, `requires_review`, `hold_code?`, `retryable`, `error_json?`, `provenance_json?`, timestamps and lease fields. | Worker progress and final intake decision. Keep parser/model/schema versions, issue codes, and safe processing/verification details; link findings below. Enforce one original (`upload` or `replacement`) attempt per document. |
| `verified_profiles` | PK/FK `(job_id, candidate_id, attempt_id)`; `schema_version`, `verified_at`, `profile_json`. | One immutable, usable handoff profile per attempt. Contains schema-validated facts, evidence levels, dates, extracted kinds, and bounded verified citations. No row means there is no usable profile. |
| `intake_spans` | PK `(job_id, candidate_id, attempt_id, span_id)`; `page?`, `location_json?`, `visibility`, `excluded_from_extraction`, `excluded_from_evidence`, `controlled_excerpt?`. | Source locations and, where retained, a bounded excerpt for restricted audit access. Span IDs are scoped to the attempt. |
| `intake_findings` | PK `(job_id, candidate_id, attempt_id, finding_id)`; `layer`, `category`, `attack_family?`, `risk`, `detector`, `action_code`. | Security findings, including those with no source span. The codes and counts can be projected for screening. |
| `intake_finding_spans` | PK `(job_id, candidate_id, attempt_id, finding_id, span_id)`; FKs to the finding and span on the same attempt. | A finding may cite several spans. Excerpts stay in the audit lane. |
| `intake_review_dispositions` | PK `(job_id, candidate_id, attempt_id, disposition_id)`; `decision_no`, `action`, `reason`, `actor_id`, `recorded_at`. Unique `(job_id, candidate_id, attempt_id, decision_no)`. | Append-only `clear_for_scoring` or `keep_on_hold`; reason must be non-blank. Latest decision is ordered by `decision_no`. |
| `replacement_requests` | PK `(job_id, candidate_id, request_id)`; `attempt_id`, `requested_by`, `requested_at`, `state` (`open`, `fulfilled`, `superseded`), `replacement_document_id?`, `closed_at?`. | Tracks the request against the current held/quarantined attempt and links it to the replacement upload when fulfilled. One open request per candidate. |

An attempt is `queued`, `running`, or `finished`. The last state has an intake outcome: `verified`, `verified_with_flag`, `manual_review`, or `quarantined`. Map module-specific results to these [intake outcomes](../RESUME_INTAKE_SPEC.md#12-outcomes). Write terminal outcome, findings, provenance and any usable profile together. Advance the candidate revision when its current attempt finishes or receives a disposition. The candidate's UI status comes from the current attempt, profile and decisions:

| UI status | Stored conditions |
|---|---|
| `processing` | The attempt is queued or running. |
| `quarantined` | The attempt ended in quarantine. |
| `ready_to_score` | The attempt ended as `verified` or `verified_with_flag`, has a usable profile, and either needs no review or its latest disposition is `clear_for_scoring`. |
| `review_required` | Remaining finished attempts. Return the hold code and applicable actions. |

A disposition write checks the candidate revision and current attempt, and can clear only a held attempt with a usable profile. The current status changes without editing the intake outcome, findings, or facts. Quarantine removes the file's payload row and retains only its metadata and audit records. A safe retry requires the source bytes to still be available. Retry or replacement advances `current_attempt_id` and the candidate revision in the same transaction as queueing the new attempt; clearance does not travel with it. A replacement upload consumes the still-open request on that exact prior attempt once. A retry supersedes any open request for its prior attempt.

## 4. Scoring runs and ranking snapshots

| Table | Key fields and relationships | Purpose |
|---|---|---|
| `scoring_bases` | PK `(job_id, basis_id)`; `profile_version` FK to an approved profile; `matcher_version`, `scorer_version`, `config_version`, `settings_json`, `fingerprint`. Unique `(job_id, fingerprint)`. | Immutable scoring basis. The fingerprint includes the locked version, effective model identifier, prompt version or hash, and scorer settings. Keep prompt bodies and credentials out of `settings_json`. |
| `scoring_runs` | PK `(job_id, run_id)`; `basis_id`, `mode` (`full` or `incremental`), `plan_fingerprint`, `requested_by`, `state`, `error_json?`, request/start/finish times and lease fields. | Accepted operation, state `queued`, `running`, `completed`, `completed_with_errors`, or `failed`. A partial unique index on `job_id` for `queued`/`running` allows one active run per job. |
| `scoring_run_candidates` | PK `(job_id, run_id, candidate_id)`; `basis_id`, exact `attempt_id`, `clearance_id?`, `planned_action` (`score`, `reuse`, `exclude`), `excluded_reason?`, `captured_risk?`, `finding_summary_json`, `state`, `score_id?`, `error_code?`. | Freeze the roster and input classification at acceptance, including excluded rows. Track each candidate as `pending`, `scored`, `reused`, `excluded`, `failed`, or `skipped` (`input_changed`). |
| `candidate_scores` | PK `(job_id, score_id)`; `origin_run_id`, `candidate_id`, `attempt_id` FK to the usable profile, `basis_id`; `fit_score` (0–1), `tier` (`A`/`B`/`C`), `mandatory_coverage?`, `counterfactual_delta?`, `links_json`, `explanations_json`, `created_at`. Unique `(job_id, origin_run_id, candidate_id)`. | Successful result for one input and basis. Store the validated scoring links and evidence explanations; no pool rank or shortlist membership. No row represents a failed score. `NULL` delta means no valid counterfactual was computed. |
| `rankings` | PK `(job_id, ranking_id)`; `run_id`, `published_at`. Unique `(job_id, run_id)`. | One immutable snapshot per `completed` or `completed_with_errors` run, even if it has zero ranked rows. Basis and run timestamps come from the referenced run. |
| `ranking_rows` | PK `(job_id, ranking_id, candidate_id)`; `run_id`, `rank?`, `unranked_reason?`. FKs on `(job_id, ranking_id, run_id)` and `(job_id, run_id, candidate_id)`. Unique `(job_id, ranking_id, rank)` for ranked rows. | The final order and unranked roster for a run. A ranked row has `rank >= 1` and a successful score reference via its run candidate. An unranked row has a reason and no rank. |

Use matching composite keys for the duplicated join columns: run candidates reference the run on `(job_id, run_id, basis_id)` and, when present, the score on `(job_id, candidate_id, attempt_id, basis_id, score_id)`. A score's producing run must have the same basis. A captured clearance points to the disposition for that same attempt. The service checks that required clearance is valid before matching or reusing a result and again before publishing.

- **Accept:** Build the preview from current attempts; the signed plan token can be validated from its fingerprint and current state without a preview table. Recheck the basis and confirmed roster, then insert the run and all candidate rows atomically. Excluded or subsequently added candidates do not become eligible for that run as intake finishes.
- **Score:** Use only the pinned approved profile and exact verified input. An incremental run reuses successful scores from the last published ranking on the same basis when the candidate's input still matches. A changed basis requires a full run; earlier scores stay available as history. An individual failure records a safe error on the run candidate, without a zero score.
- **Publish:** In one transaction, finalize candidate outcomes and run state, insert the ranking and its rows, and, if the basis is still active, update `jobs.current_ranking_id` and create the shortlist proposal. A run-level failure publishes nothing. A run that finishes on an older basis may publish history, without moving the current pointer. Captured roster, risk summaries and terminal result rows are frozen so later intake or review cannot change historical counts or evidence. The workspace resolves `latest_scoring_run_id` and `latest_ranking_id` from history and checks the current ranking's basis at read time.

## 5. Shortlist proposals and approvals

| Table | Key fields and relationships | Purpose |
|---|---|---|
| `shortlist_proposals` | PK `(job_id, proposal_id)`; `ranking_id`, `proposal_no`, `state` (`draft`, `approved`, `superseded`), `current_revision_no`, `approval_baseline_id?`, `previous_ranking_id?`, `comparison_proposal_id?` and `comparison_revision_no?`, `created_at`. Unique `(job_id, ranking_id, proposal_no)`. | Editable shortlist for one immutable ranking. Pin the previous approved shortlist as the baseline; if none exists, reference the previous ranking and its exact proposal revision when available. |
| `shortlist_proposal_revisions` | PK `(job_id, proposal_id, revision_no)`; `size?`, `overrides_json`, `prior_override_decisions_json`, `saved_by`, `saved_at`. | Immutable revision of the selected N, new include/exclude actions and reasons, and explicit reconfirm/drop decisions. Each entry records the decision's actor and time. The API revision token identifies this proposal and revision. |
| `shortlist_approvals` | PK `(job_id, shortlist_id)`; `ranking_id`, `(proposal_id, revision_no)` FK to the exact proposal revision, `supersedes_shortlist_id?` FK to this job's prior approval, `effective_overrides_json`, `approved_by`, `approved_at`. Unique `(job_id, proposal_id)`. | Immutable approved snapshot. Assign stable override IDs to the directives saved at approval so the next proposal can reference their actions and original reasons. |
| `shortlist_members` | PK `(job_id, shortlist_id, candidate_id)`; `ranking_id`, `rank`. FK to the approval on `(job_id, shortlist_id, ranking_id)` and the ranked row on `(job_id, ranking_id, candidate_id, rank)`; `rank` is non-null. | Materialized membership as approved. List in computed rank order; an unranked candidate cannot be a member. |

Allow one draft proposal per ranking with a partial unique index. New proposals start with the last selected N if any, with no overrides applied. Only the current ranking has an editable proposal; supersede its draft when a newer ranking becomes current or its basis changes. Derive the workspace's proposal from the current ranking's latest `proposal_no`. Top N, near-cutoff markers, and rank changes derive from the immutable rankings, the selected N, and the pinned baseline. The approval's actual membership comes from `shortlist_members`, including any HR overrides.

On edit, validate that new override targets are ranked in this proposal, no candidate has conflicting directives, reasons are non-blank, and prior override IDs belong to the pinned approval. A prior override for an unranked or absent candidate can only be dropped. On approval, recheck the proposal revision, current ranking and approval baseline; require a positive N, ranked candidates, and an explicit decision on every prior override. Insert the approval and its members, lock the proposal, move `last_approved_shortlist_id`, and append audit in one transaction. Later proposals cannot edit prior approvals.

## 6. Audit, retries, and SQLite operation

| Table | Key fields and relationships | Purpose |
|---|---|---|
| `audit_events` | PK `(job_id, event_no)`; `event_id`, `event_type`, `actor_id`, `recorded_at`, `references_json`, `details_json`, `prev_hash`, `event_hash`. | Append-only job timeline: draft/review edits, profile approvals, intake transitions and dispositions, replacement requests, run publication/failure, shortlist changes/approvals, excerpt access and exports. Link to exact revisions, attempts, and snapshots. Adapt the [existing audit ledger](../src/agentic_talent_screener/ranking/audit.py)'s canonical hashing to an ordered chain per job. |
| `idempotency_keys` | PK `(actor_id, method, resource_path, idempotency_key)`; `job_id?`, `request_hash`, `response_status`, `location?`, `response_json?`, `recorded_at`. | Replay the original acceptance or result after authorization. Hash normalized input and file digests rather than storing the original request; reject a different input for the same key. Write the replay record with the mutation. |
| `workflow_secrets` | PK `name`; `value` BLOB. | Stores the generated local key that signs score-plan tokens across API restarts. |

- Persist work on `requirement_drafts`, `intake_attempts`, and `scoring_runs`; their lease fields are `lease_token?` and `lease_until?`. Claim or reclaim expired work transactionally. Workers check their lease token before committing progress, success, or failure. A recovered scoring run resumes from its stored candidate rows; a recovered JD/intake operation can restart processing with the saved source. Uniqueness constraints prevent duplicate results and publications.
- Use WAL on the service host, a bounded busy timeout, and short write transactions (`BEGIN IMMEDIATE` for compare-and-write operations). Perform parsing and model calls outside write transactions. Include a business mutation, its audit event, and the applicable idempotency record in the same commit. Use deferred foreign keys for create-time cycles such as candidate and initial attempt.
- Index the common read paths: permission lookups by actor; draft, batch, run and approval history by job/time; open work by state and lease expiry; candidate attempts and dispositions by their parent keys; rank pages by job/ranking/rank; and audit by job/event number. Use schema migrations for changes to SQLite tables and version the JSON payloads independently.
- Keep raw resume BLOBs and controlled excerpts in restricted access paths. SQLite permissions apply to the database file, so the service enforces job and review permissions and protects the DB, WAL sidecar and backups. Routine queries and idempotency responses use authorized, minimized projections. Audit events store safe references and summaries, never raw resumes, prompts or contact data. Matching loads a quote-free view of the verified profile; scoring workers never read the raw resume store. A separately persisted identity envelope is unnecessary for the current UI.

## 7. Open questions

- [SPEC Q12](../SPEC.md#12-open-questions): retention and deletion by data class (including files, excerpts, history, replay records, WAL and backups); database location and encryption; identity access; and whether restricted source BLOBs need a separate store for deployment.
- [SPEC Q13 and Q32](../SPEC.md#12-open-questions): who can approve requirements, and whether the scorer also enforces the logged clearance gate. Excerpt access, shortlist approval, and export permissions/contents follow the API's open policy decisions.
- [SPEC Q26](../SPEC.md#12-open-questions): how to compute a meaningful counterfactual for evidence already removed during intake. The model can record its absence without inventing a score.

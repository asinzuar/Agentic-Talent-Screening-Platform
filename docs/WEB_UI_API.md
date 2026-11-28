# Web UI API Design: From Job Description to Shortlist

> HTTP contract for the [Web UI flows](WEB_UI_FLOWS.md), implemented under `/api/v1/jobs`. See the [SQLite data model](DATA_MODEL.md) for persistence and the [README](../README.md#run-the-job-api-and-worker) for running the API and worker. [SPEC.md](../SPEC.md) governs workflow and scoring; [resume intake](../RESUME_INTAKE_SPEC.md) and [ranking](RANKING_SPEC.md) define the module contracts.

## 1. Job workspace

One job owns one applicant pool. The UI uses the same job ID when adding resumes, opening a requirements revision, running scoring, and reviewing history.

In the tables, `J` means `/api/v1/jobs/{job_id}`. IDs are server-assigned; `candidate_id` is a blind ID scoped to the job. Timestamps are UTC. Bodies are JSON unless marked multipart, and list endpoints return `{items, next_cursor}` with `limit` and `cursor` parameters. All routes require authentication and job-scoped authorization. Actor and timestamp fields on decisions come from the server.

The initial deployment uses server-configured bearer tokens mapped to HR actors. The creator gets `owner` permission for the job; additional grants are stored per actor and job. The browser sends `Authorization: Bearer <token>`.

| Method and path | Request → response |
|---|---|
| `GET /api/v1/jobs` | List accessible jobs with their title, draft state, and active profile version. |
| `POST /api/v1/jobs` | Multipart `file`: one UTF-8 TXT or text-based PDF job description (JD), up to 10 MiB. `202` → job summary; `Location` points to `J/requirement-draft` for progress. |
| `GET J` | Get the workspace summary and references needed to open or refresh its tabs. |

The workspace returns `job_id`, `role_title` (from the active profile or, before approval, the draft), `draft` (status and link), `active_profile` (`profile_id` and `version`, or `null`), `latest_scoring_run_id`, `current_ranking_id`, `latest_ranking_id`, `current_proposal_id`, and `last_approved_shortlist_id`. Unavailable fields are `null`. Action availability is keyed by action name, with `{allowed, reason_code}` for each action; for example, `upload_resumes` is disabled with `REQUIREMENTS_NOT_APPROVED` before v1. Mutations enforce the same checks.

`ranking_state` is `none` before the first published ranking, `current` when a published ranking matches the active scoring basis, and `rescore_required` when a previous ranking exists but its basis is stale. In the last case `current_ranking_id` is `null` and `latest_ranking_id` still links to historical results. A current ranking describes the pool captured by its run; check the score plan for newly ready work.

### Request and progress conventions

- Successful reads and in-place updates return `200`; synchronous resource creation returns `201`; uploads and requests that start background work return `202` and a `Location` to poll. Poll the resources named below; progress and terminal outcomes survive a page reload.
- Create and action `POST`s take an `Idempotency-Key`. After authorization, replaying the same request by the same actor on the same resource path returns the original acceptance/result, even if state has since changed. A key reused with different input returns `409`. Where specified, `expected_revision` is the opaque revision last read; a stale write returns `409` so the UI can reload and ask the reviewer to confirm again.
- Commit a decision and its audit record together. For background work, persist the accepted operation before returning `202` so the same operation can resume after a worker interruption.
- HTTP errors use `{code, message, retryable, field_errors?}`, where each field error names a field path and safe message. Use `401` for unauthenticated requests, `403` for unavailable permissions, `404` for unknown or out-of-scope resource IDs, `409` for stale state or a disallowed transition, `413` for an oversized request, `415` for an unsupported request media type, and `422` for invalid fields. Failed background work appears on the polled resource with the same error shape; the read itself can succeed. Responses and errors omit raw resumes, prompts, provider diagnostics, and identity fields.

## 2. Draft and approve requirements

| Method and path | Request → response |
|---|---|
| `GET J/requirement-draft` | Draft status `none`, `queued`, `running`, `ready`, or `failed`; when running, stage `extract_jd` or `draft_requirements`. Returns `jd_id`, `revision`, `base_version` (`null` for the first draft), the editable fields when ready, and a safe error and available recovery actions when failed. `none` means no open draft. |
| `GET J/job-descriptions/{jd_id}` | Get the JD text used for drafting and quote validation, with its media type. `text` is `null` if extraction has not succeeded. Drafts and locked versions reference the exact JD source they used. |
| `POST J/requirement-draft/retry` | Retry a failed initial draft. Multipart `expected_revision` and optional replacement `file`. With no file, retry the same source only when `retryable`; if the source is unusable, supply a new file. `202` → poll the draft. |
| `PUT J/requirement-draft` | Save `{expected_revision, role_title, requirements, not_used_for_scoring, activate_requirement_ids, flag_reviews}` for a ready draft. Full replacement of editable content: include existing IDs for edits, omit removed criteria, and omit IDs for new criteria/options. Returns the saved draft with assigned IDs and a new revision. |
| `POST J/requirement-draft/approve` | `{expected_revision}` → `201` locked profile. Validate the reviewed draft (including at least one active requirement and reviewer flag dispositions), record the authenticated approver and time, and set the active version in one transaction. |
| `POST J/requirement-draft` | `{base_version}` → `201` new ready draft cloned from the active profile. Reject if the base is no longer active or a draft is already open. |
| `GET J/requirement-profiles/{version}` | Get an immutable approved profile, including `profile_id`, `version`, `jd_id`, criteria, non-scoring passages, reviewer flag dispositions, and `approved_by` / `approved_at`. |

Use [the requirement fields in SPEC §13.1](../SPEC.md#131-requirementprofile-62): `requirement_id`, `statement`, `kind`, `options` and `combine`, `substitutes`, `min_years`, `min_evidence`, `priority` (`M` or `P`), `weight`, `provenance`, `quote` or `rationale`, optional `onet_source`, `status`, and reviewer flags. The API checks that JD quotes and non-scoring passages occur in the referenced JD. New HR-authored criteria receive `hr_added` provenance and require a rationale. Original provenance is preserved on edits. Reviewer flags carry IDs; `flag_reviews` records `{flag_id, disposition}` (`addressed` or `acknowledged`) for flags on retained criteria. Removing or addressing a flagged criterion is recorded in the draft history.

Drafting never approves the profile. Inferred and model-suggested criteria start inactive. To change one to active, HR must explicitly set `status: "active"` and list its ID in `activate_requirement_ids` on that edit. A revision preserves previously reviewed statuses. Profile version numbers are assigned on approval, independently of draft revisions. The first approval unlocks resume uploads. Opening a revision leaves the locked version active; approving the revision requires a full run on the new basis before any ranking is current for it.

## 3. Upload and verify resumes

| Method and path | Request → response |
|---|---|
| `POST J/resume-batches` | Multipart repeated `files` (PDF, DOCX, or TXT per intake policy), `profile_version` from the upload confirmation, and optional `replacement_request_id` for a single replacement file. `202` → batch ID and per-file candidate and intake-attempt IDs. Requires an approved active version; if the confirmed version changed, return `409`. |
| `GET J/resume-batches` | List batches in this pool with submitted time, status counts, and detail links. |
| `GET J/resume-batches/{batch_id}` | Poll original submissions by upload position. Each row returns `candidate_id`, `attempt_id`, processing stage or terminal status, and safe issue codes. One file's intake failure does not stop the others. |
| `GET J/candidates` | List current applicant records with optional `status` filter. Each row shows blind ID, current attempt, status, risk and flag summary, and available actions; no identifying filenames. Unranked results can link here by candidate. |
| `GET J/candidates/{candidate_id}` | Get the current attempt summary and progress, `revision`, hold reason if present, latest review disposition, any open `replacement_request_id`, and available actions. |
| `GET J/candidates/{candidate_id}/intake-attempts/{attempt_id}/audit` | Authorized review packet for an exact attempt: verified facts and bounded cited evidence, separately from security/exclusion findings, processing issues, provenance, and disposition history. |
| `GET J/candidates/{candidate_id}/intake-attempts/{attempt_id}/excerpts/{span_id}` | Authorized, access-logged read of a bounded controlled excerpt with page/location, when retained and permitted. Use links from the audit packet; no raw file or full span index. |
| `POST J/candidates/{candidate_id}/review-dispositions` | `{expected_revision, attempt_id, action, reason}` (`clear_for_scoring` or `keep_on_hold`; non-blank reason). `201` → recorded disposition and updated eligibility. |
| `POST J/candidates/{candidate_id}/intake-retries` | `{expected_revision, attempt_id}`. `202` → new attempt on the same file and blind ID, only for a retryable hold without a usable profile. Poll the candidate, which references the new attempt. |
| `POST J/candidates/{candidate_id}/replacement-requests` | `{expected_revision, attempt_id}`. `201` → open replacement request ID for a quarantined file or a hold without a usable profile. Reuse that ID when uploading a replacement. |

`processing` rows expose `stage: queued | document_safety | injection_detection | extraction | evidence_verification` as work advances. Terminal UI states map to intake as follows:

| API status | Conditions |
|---|---|
| `ready_to_score` | A usable Verified Profile exists and no clearance is required, or clearance is logged for that exact attempt and profile. Non-blocking or cleared flags remain visible. |
| `review_required` | Mandatory review of a flagged profile, or intake could not produce a usable profile. Return a hold code and available actions. |
| `quarantined` | Intake blocked the file; no retry or release. Replacement remains available. |

Upload limits are 32 files, 10 MiB per file, and 32 MiB of file data per batch. Transport-level rejection creates no batch; an accepted file gets its own intake attempt and, unless it is an explicit replacement, a new blind candidate ID. The confirmed version is upload metadata. L0–L3 receive the file in isolation, without the JD or requirement profile. Upload, retry, replacement, and clearance never start scoring.

The audit packet identifies the blind candidate, attempt, document hash, intake outcome and stopping layer, overall risk, finding category/layer/risk, cited span IDs and page/location, and whether content was excluded. Include relevant model/schema versions and safe failure summaries. Findings without a retained span have no excerpt link.

The same authorized HR user who uploaded the file may clear its hold. Both dispositions apply to the current held attempt and record actor, reason, and timestamp. Clearance additionally requires a usable Verified Profile and changes only eligibility; it leaves the facts, exclusions, and flags intact. A retry or replacement creates a fresh attempt on the same blind candidate ID without inheriting clearance. A replacement upload consumes an open request for the current attempt once. Original attempts and review records remain attributable. The replacement request records that a file is needed; no candidate notification channel is assumed.

## 4. Score and review rankings

| Method and path | Request → response |
|---|---|
| `GET J/score-plan` | Preview the next run against the active profile: role and profile version, matcher and scoring config versions, `mode` (`full` or `incremental`), `pool_count`, `ready_count`, `to_score_count`, `reuse_count`, `excluded_counts` by reason, and `can_start` with a reason when disabled. Returns a `plan_token` when startable. Basis and mode can be `null` before requirements approval. |
| `POST J/scoring-runs` | `{plan_token}` → `202` scoring run; `Location` points to the run. If the basis or confirmed pool/eligibility changed, or a run is already active for the job, return `409`. |
| `GET J/scoring-runs/{run_id}` | Poll `queued`, `running`, `completed`, `completed_with_errors`, or `failed`. Returns frozen plan counts, `scored_count`, `reused_count`, `error_count`, `skipped_count` for inputs that changed, timestamps, and `ranking_id` when a snapshot was published. A run-level failure returns an actionable error and no new ranking. |
| `GET J/rankings` | List published ranking snapshots with basis, run time, and ranked/unranked counts. |
| `GET J/rankings/{ranking_id}` | Immutable ranking header: role, profile/matcher/config versions, `run_id`, run timestamps, `ranked_count`, `unranked_count`, `unranked_counts` by reason, and security finding and flagged-candidate counts for the captured pool. |
| `GET J/rankings/{ranking_id}/candidates` | List rows with `status=ranked` or `status=unranked`. Ranked rows sort by rank; include `candidate_id`, `fit_score` (0–1), `tier`, `rank`, security flag summary, and `unassessed` requirement IDs. Unranked rows include a reason code and a link to the candidate record. |
| `GET J/rankings/{ranking_id}/candidates/{candidate_id}` | Get the exact result for that run, with `attempt_id` and links to authorized audit details. For a ranked result, include per-requirement evidence, gaps, and assessment state. |
| `GET J/rankings/{ranking_id}/audit` | Authorized run provenance scoped to the snapshot: profile approval, scoring initiator and basis, new/reused/failed counts, and run and candidate audit references. |

### Starting and publishing a run

- The preview covers the whole job pool. `ready_count = to_score_count + reuse_count`; exclusions include `processing`, `review_required`, and `quarantined`. A run can start before intake finishes: for example, **18 to score of 24 against v1**, with four processing and two held. Disable start with `REQUIREMENTS_NOT_APPROVED`, `RUN_IN_PROGRESS`, `NO_READY_CANDIDATES`, or `NO_NEW_WORK` as applicable. A failed score from a prior run is work to retry if that candidate is still ready.
- The server binds the token to the confirmed counts, candidate inputs and scoring basis. On acceptance, the run freezes the approved profile, matcher and scoring configuration, and candidate and Verified Profile references. Newly uploaded or newly cleared candidates wait for the next explicit run. A held candidate cannot enter matching without logged clearance on the exact profile. Check eligibility before matching or reusing scores and again before publication. A captured ready input that is no longer eligible or was superseded is unranked with reason `input_changed`.
- A full run scores all eligible candidates against one basis. An incremental run reuses successful results only for the same job, candidate, verified input, and exact basis (approved profile ID/version, matcher and scorer behavior and configuration). It scores the rest and orders **all** successful eligible results with the ranking module's tie-breaks. Changing the basis requires a full run; choosing a new shortlist size does not.
- Keep any previous ranking readable while the run is active. Publish the new ranking with its eligible successes and unranked rows together. Candidate errors produce `completed_with_errors` and a `scoring_failed` unranked reason; processing, held, quarantined, ineligible or failed rows have no score, tier, rank, or shortlist membership. Counts describe the captured pool, including its exclusions. If the basis changed while the run was in flight, its snapshot is historical on publication and does not become current.

### Evidence and shortlist display

Ranked candidate detail includes `mandatory_coverage` and one assessment per active requirement: requirement ID/statement, `assessment: met | not_met | not_assessed`, credit (nullable when not assessed), gap, and deciding evidence and option/substitute results. Map a missing evidence source to `not_assessed` and an assessed but unmet requirement to `not_met`. For a minimum-years requirement, expose `min_years` and `evidence_met` so the UI can distinguish sufficient skill evidence from a duration minimum that wasn't established. The option's optional `duration` includes the self-reported amount and scope, bounds in months, whether the minimum is supported, and all its supporting citations with span/page locations. Include `counterfactual_delta` only for an actual, meaningful counterfactual computation ([SPEC Q26](../SPEC.md#12-open-questions)).

The ranked table combines the ranking rows with its selected proposal or approved shortlist, keyed by blind candidate ID and the **same** `ranking_id`. Do not apply a shortlist from an older ranking to current rows. Excluded evidence is available separately through the exact-attempt audit route, subject to permission.

## 5. Propose, approve, and update the shortlist

| Method and path | Request → response |
|---|---|
| `GET J/shortlist-proposals/{proposal_id}` | Get the proposed membership, cutoff, comparison, and review state for one ranking. `GET J` gives the current proposal ID. |
| `PUT J/shortlist-proposals/{proposal_id}` | Save `{expected_revision, size, overrides, prior_override_decisions}` for the current editable proposal. Replace the submitted override and decision lists; return updated membership and comparison. |
| `POST J/shortlist-proposals/{proposal_id}/approve` | `{expected_revision}` → `201` immutable approved shortlist. Requires a current ranking, unchanged approval baseline, a selected size, ranked candidates, and explicit decisions on all previous overrides. Locks the proposal and advances the job's last-approved pointer atomically. |
| `POST J/rankings/{ranking_id}/shortlist-proposals` | `{}` → `201` new proposal to revise the approved shortlist without rerunning scoring. Only on the current ranking and only when no editable proposal exists. |
| `GET J/shortlists` | List approved snapshots, including superseded approvals, with the ranking and requirements version they reference. |
| `GET J/shortlists/{shortlist_id}` | Get the exact approved snapshot: `ranking_id`, profile version, size, computed top N and cutoff, approved member IDs, applicable overrides and decisions, `approved_by`, `approved_at`, and `supersedes_shortlist_id`. |
| `GET J/shortlists/{shortlist_id}/audit` | Authorized decision trail for the snapshot, with proposal revision, ranking provenance reference, approval, and override actions, reasons, actors, and times. |
| `GET J/shortlists/{shortlist_id}/export?format={format}` | Authorized file download for that exact approved snapshot, with `Content-Type` and an attachment `Content-Disposition`. Include shortlist, ranking and profile-version identifiers in the export and record the export in audit. Advertise supported formats on the shortlist resource once policy is set. |

On publishing a current ranking, create a proposal with the last selected size if any; otherwise `size` is `null` until HR chooses it. `size` is an integer ≥ 1. The proposal returns `proposal_id`, `ranking_id`, `revision`, status (`draft`, `approved`, or `superseded`), `size`, `top_n_ids` (up to N ranked candidates), `near_cutoff_ids` (the next few per the run's scoring configuration), `effective_member_ids`, and `prior_overrides` with their review state. While size is unset, membership and cutoff are unset. Overrides change effective membership while preserving the computed cutoff and rank. When a ranking ceases to be current, make its draft proposal read-only and clear `current_proposal_id` until the next current ranking is published.

- **Override:** send `{candidate_id, action, reason}` in `overrides` with `action: include | exclude` and a non-blank reason. Only ranked candidates can be overridden. Each candidate has at most one effective override. Record edits to decisions, including removing a draft override; never edit the computed score or rank.
- **Reconfirm or drop:** the proposal shows the last approved shortlist's overrides with IDs, original actions, reasons, and decisions (`null` until resolved). No prior override is applied by default. Send `{override_id, decision}` in `prior_override_decisions` (`reconfirm` or `drop`). Reconfirming applies the prior action and preserves the reason; if the candidate is no longer ranked, only `drop` is allowed. To replace a prior action with a new one, drop it and submit a new override with a reason.
- **Compare:** freeze a comparison baseline at proposal creation: the previous approved membership and its ranking, or the preceding ranking's proposed top N if no shortlist has been approved. Return entered/exited IDs for the computed top N and for effective membership, with old/new rank (`null` when unranked or absent). Include rank changes for members who stayed. If there was no prior membership or size has not been chosen yet, omit the membership diff. Only a new approval supersedes the last approved snapshot.

For example, at size 10, an incremental v1 run can reuse earlier scores and rank a newly ready candidate at #3. The new top N shows that candidate entering and the old #10 leaving at #11. The approved shortlist still serves its original ranking and membership until HR resolves the prior overrides and approves the new proposal.

## 6. Open questions and implementation dependencies

- **Roles and exports:** The first implementation exposes JSON export for approved snapshots to actors with `export` or `owner` permission. The deployment's role policy, including who can approve requirements ([SPEC Q13](../SPEC.md#12-open-questions)), approve shortlists, access excerpts, and export snapshots, remains open. Further formats and pending-proposal exports are open.
- **Eligibility enforcement:** The workflow requires logged clearance before matching. Whether the scorer itself also enforces this remains open ([SPEC Q32](../SPEC.md#12-open-questions)). Requirements approval and intake eligibility are distinct gates.
- **Persistence and privacy:** The [SQLite data model](DATA_MODEL.md) describes durable job state, resumable background work, and attributable approvals and audit references. Retention, deployment storage protections, identity access and permitted providers for runtime applicant data remain open ([SPEC Q12, Q29, Q31](../SPEC.md#12-open-questions)). Apply the intake audit access rules to excerpts and use privacy-minimised views for routine reads and exports.
- **Counterfactual:** Evidence cards come from the scorer's recorded requirement explanations. The counterfactual for content already excluded during intake remains open ([SPEC Q26](../SPEC.md#12-open-questions)), so live workflow results omit the delta.

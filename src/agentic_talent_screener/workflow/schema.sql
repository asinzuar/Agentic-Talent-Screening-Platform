CREATE TABLE IF NOT EXISTS actors (
    actor_id TEXT PRIMARY KEY,
    issuer TEXT NOT NULL,
    subject TEXT NOT NULL,
    disabled_at TEXT,
    UNIQUE (issuer, subject)
) STRICT;

CREATE TABLE IF NOT EXISTS jobs (
    job_id TEXT PRIMARY KEY,
    profile_id TEXT NOT NULL UNIQUE,
    created_by TEXT NOT NULL REFERENCES actors,
    created_at TEXT NOT NULL,
    active_profile_version INTEGER,
    current_ranking_id TEXT,
    last_approved_shortlist_id TEXT,
    FOREIGN KEY (job_id, active_profile_version) REFERENCES requirement_profiles,
    FOREIGN KEY (job_id, current_ranking_id) REFERENCES rankings,
    FOREIGN KEY (job_id, last_approved_shortlist_id) REFERENCES shortlist_approvals
) STRICT;

CREATE TABLE IF NOT EXISTS job_permissions (
    job_id TEXT NOT NULL REFERENCES jobs,
    actor_id TEXT NOT NULL REFERENCES actors,
    permission TEXT NOT NULL,
    PRIMARY KEY (job_id, actor_id, permission)
) STRICT;
CREATE INDEX IF NOT EXISTS permissions_by_actor ON job_permissions(actor_id, job_id);

CREATE TABLE IF NOT EXISTS job_descriptions (
    job_id TEXT NOT NULL REFERENCES jobs,
    jd_id TEXT NOT NULL,
    media_type TEXT NOT NULL,
    size_bytes INTEGER NOT NULL CHECK (size_bytes >= 0),
    sha256 TEXT NOT NULL,
    source_bytes BLOB,
    extracted_text TEXT,
    uploaded_by TEXT NOT NULL REFERENCES actors,
    uploaded_at TEXT NOT NULL,
    PRIMARY KEY (job_id, jd_id)
) STRICT;

CREATE TABLE IF NOT EXISTS requirement_drafts (
    job_id TEXT NOT NULL REFERENCES jobs,
    draft_id TEXT NOT NULL,
    jd_id TEXT NOT NULL,
    base_version INTEGER,
    state TEXT NOT NULL CHECK (state IN ('queued','running','ready','failed','approved')),
    stage TEXT CHECK (stage IN ('extract_jd','draft_requirements')),
    error_json TEXT CHECK (json_valid(error_json)),
    revision_token TEXT NOT NULL UNIQUE,
    current_revision_no INTEGER,
    created_by TEXT NOT NULL REFERENCES actors,
    created_at TEXT NOT NULL,
    lease_token TEXT,
    lease_until TEXT,
    PRIMARY KEY (job_id, draft_id),
    FOREIGN KEY (job_id, jd_id) REFERENCES job_descriptions,
    FOREIGN KEY (job_id, base_version) REFERENCES requirement_profiles,
    FOREIGN KEY (job_id, draft_id, current_revision_no) REFERENCES requirement_draft_revisions
        DEFERRABLE INITIALLY DEFERRED
) STRICT;
CREATE UNIQUE INDEX IF NOT EXISTS open_draft ON requirement_drafts(job_id) WHERE state <> 'approved';
CREATE INDEX IF NOT EXISTS draft_queue ON requirement_drafts(state, lease_until, created_at);

CREATE TABLE IF NOT EXISTS requirement_draft_revisions (
    job_id TEXT NOT NULL,
    draft_id TEXT NOT NULL,
    revision_no INTEGER NOT NULL CHECK (revision_no >= 1),
    jd_id TEXT NOT NULL,
    role_title TEXT NOT NULL,
    content_json TEXT NOT NULL CHECK (json_valid(content_json)),
    saved_by TEXT NOT NULL REFERENCES actors,
    saved_at TEXT NOT NULL,
    PRIMARY KEY (job_id, draft_id, revision_no),
    FOREIGN KEY (job_id, draft_id) REFERENCES requirement_drafts,
    FOREIGN KEY (job_id, jd_id) REFERENCES job_descriptions
) STRICT;

CREATE TABLE IF NOT EXISTS requirement_profiles (
    job_id TEXT NOT NULL REFERENCES jobs,
    version INTEGER NOT NULL CHECK (version >= 1),
    draft_id TEXT NOT NULL,
    revision_no INTEGER NOT NULL,
    approved_by TEXT NOT NULL REFERENCES actors,
    approved_at TEXT NOT NULL,
    PRIMARY KEY (job_id, version),
    UNIQUE (job_id, draft_id),
    FOREIGN KEY (job_id, draft_id, revision_no) REFERENCES requirement_draft_revisions
) STRICT;

CREATE TABLE IF NOT EXISTS resume_batches (
    job_id TEXT NOT NULL REFERENCES jobs,
    batch_id TEXT NOT NULL,
    confirmed_profile_version INTEGER NOT NULL,
    uploaded_by TEXT NOT NULL REFERENCES actors,
    uploaded_at TEXT NOT NULL,
    PRIMARY KEY (job_id, batch_id),
    FOREIGN KEY (job_id, confirmed_profile_version) REFERENCES requirement_profiles
) STRICT;
CREATE INDEX IF NOT EXISTS batch_history ON resume_batches(job_id, uploaded_at);

CREATE TABLE IF NOT EXISTS candidates (
    job_id TEXT NOT NULL REFERENCES jobs,
    candidate_id TEXT NOT NULL,
    current_attempt_id TEXT NOT NULL,
    revision_token TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (job_id, candidate_id),
    FOREIGN KEY (job_id, candidate_id, current_attempt_id) REFERENCES intake_attempts
        DEFERRABLE INITIALLY DEFERRED
) STRICT;

CREATE TABLE IF NOT EXISTS resume_documents (
    job_id TEXT NOT NULL,
    candidate_id TEXT NOT NULL,
    document_id TEXT NOT NULL,
    batch_id TEXT NOT NULL,
    upload_position INTEGER NOT NULL CHECK (upload_position >= 1),
    sha256 TEXT NOT NULL,
    size_bytes INTEGER NOT NULL CHECK (size_bytes >= 0),
    media_type TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (job_id, candidate_id, document_id),
    UNIQUE (job_id, batch_id, upload_position),
    FOREIGN KEY (job_id, candidate_id) REFERENCES candidates,
    FOREIGN KEY (job_id, batch_id) REFERENCES resume_batches
) STRICT;

CREATE TABLE IF NOT EXISTS resume_payloads (
    job_id TEXT NOT NULL,
    candidate_id TEXT NOT NULL,
    document_id TEXT NOT NULL,
    source_bytes BLOB NOT NULL,
    PRIMARY KEY (job_id, candidate_id, document_id),
    FOREIGN KEY (job_id, candidate_id, document_id) REFERENCES resume_documents
) STRICT;

CREATE TABLE IF NOT EXISTS intake_attempts (
    job_id TEXT NOT NULL,
    candidate_id TEXT NOT NULL,
    attempt_id TEXT NOT NULL,
    document_id TEXT NOT NULL,
    prior_attempt_id TEXT,
    trigger_kind TEXT NOT NULL CHECK (trigger_kind IN ('upload','retry','replacement')),
    requested_by TEXT NOT NULL REFERENCES actors,
    state TEXT NOT NULL CHECK (state IN ('queued','running','finished')),
    stage TEXT NOT NULL CHECK (stage IN ('queued','document_safety','injection_detection','extraction','evidence_verification')),
    outcome TEXT CHECK (outcome IN ('verified','verified_with_flag','manual_review','quarantined')),
    overall_risk TEXT CHECK (overall_risk IN ('low','medium','high')),
    requires_review INTEGER NOT NULL DEFAULT 0 CHECK (requires_review IN (0,1)),
    hold_code TEXT,
    retryable INTEGER NOT NULL DEFAULT 0 CHECK (retryable IN (0,1)),
    error_json TEXT CHECK (json_valid(error_json)),
    provenance_json TEXT CHECK (json_valid(provenance_json)),
    created_at TEXT NOT NULL,
    finished_at TEXT,
    lease_token TEXT,
    lease_until TEXT,
    PRIMARY KEY (job_id, candidate_id, attempt_id),
    FOREIGN KEY (job_id, candidate_id, document_id) REFERENCES resume_documents,
    FOREIGN KEY (job_id, candidate_id, prior_attempt_id) REFERENCES intake_attempts,
    CHECK ((state = 'finished') = (outcome IS NOT NULL))
) STRICT;
CREATE UNIQUE INDEX IF NOT EXISTS original_attempt ON intake_attempts(job_id, candidate_id, document_id)
    WHERE trigger_kind IN ('upload','replacement');
CREATE INDEX IF NOT EXISTS intake_queue ON intake_attempts(state, lease_until, created_at);

CREATE TABLE IF NOT EXISTS verified_profiles (
    job_id TEXT NOT NULL,
    candidate_id TEXT NOT NULL,
    attempt_id TEXT NOT NULL,
    schema_version INTEGER NOT NULL,
    verified_at TEXT NOT NULL,
    profile_json TEXT NOT NULL CHECK (json_valid(profile_json)),
    PRIMARY KEY (job_id, candidate_id, attempt_id),
    FOREIGN KEY (job_id, candidate_id, attempt_id) REFERENCES intake_attempts
) STRICT;

CREATE TABLE IF NOT EXISTS intake_spans (
    job_id TEXT NOT NULL,
    candidate_id TEXT NOT NULL,
    attempt_id TEXT NOT NULL,
    span_id TEXT NOT NULL,
    page INTEGER,
    location_json TEXT CHECK (json_valid(location_json)),
    visibility TEXT NOT NULL,
    excluded_from_extraction INTEGER NOT NULL CHECK (excluded_from_extraction IN (0,1)),
    excluded_from_evidence INTEGER NOT NULL CHECK (excluded_from_evidence IN (0,1)),
    controlled_excerpt TEXT,
    PRIMARY KEY (job_id, candidate_id, attempt_id, span_id),
    FOREIGN KEY (job_id, candidate_id, attempt_id) REFERENCES intake_attempts
) STRICT;

CREATE TABLE IF NOT EXISTS intake_findings (
    job_id TEXT NOT NULL,
    candidate_id TEXT NOT NULL,
    attempt_id TEXT NOT NULL,
    finding_id TEXT NOT NULL,
    layer TEXT NOT NULL,
    category TEXT NOT NULL,
    attack_family TEXT,
    risk TEXT NOT NULL,
    detector TEXT NOT NULL,
    action_code TEXT NOT NULL,
    PRIMARY KEY (job_id, candidate_id, attempt_id, finding_id),
    FOREIGN KEY (job_id, candidate_id, attempt_id) REFERENCES intake_attempts
) STRICT;

CREATE TABLE IF NOT EXISTS intake_finding_spans (
    job_id TEXT NOT NULL,
    candidate_id TEXT NOT NULL,
    attempt_id TEXT NOT NULL,
    finding_id TEXT NOT NULL,
    span_id TEXT NOT NULL,
    PRIMARY KEY (job_id, candidate_id, attempt_id, finding_id, span_id),
    FOREIGN KEY (job_id, candidate_id, attempt_id, finding_id) REFERENCES intake_findings,
    FOREIGN KEY (job_id, candidate_id, attempt_id, span_id) REFERENCES intake_spans
) STRICT;

CREATE TABLE IF NOT EXISTS intake_review_dispositions (
    job_id TEXT NOT NULL,
    candidate_id TEXT NOT NULL,
    attempt_id TEXT NOT NULL,
    disposition_id TEXT NOT NULL,
    decision_no INTEGER NOT NULL,
    action TEXT NOT NULL CHECK (action IN ('clear_for_scoring','keep_on_hold')),
    reason TEXT NOT NULL CHECK (length(trim(reason)) > 0),
    actor_id TEXT NOT NULL REFERENCES actors,
    recorded_at TEXT NOT NULL,
    PRIMARY KEY (job_id, candidate_id, attempt_id, disposition_id),
    UNIQUE (job_id, candidate_id, attempt_id, decision_no),
    FOREIGN KEY (job_id, candidate_id, attempt_id) REFERENCES intake_attempts
) STRICT;

CREATE TABLE IF NOT EXISTS replacement_requests (
    job_id TEXT NOT NULL,
    candidate_id TEXT NOT NULL,
    request_id TEXT NOT NULL,
    attempt_id TEXT NOT NULL,
    requested_by TEXT NOT NULL REFERENCES actors,
    requested_at TEXT NOT NULL,
    state TEXT NOT NULL CHECK (state IN ('open','fulfilled','superseded')),
    replacement_document_id TEXT,
    closed_at TEXT,
    PRIMARY KEY (job_id, candidate_id, request_id),
    UNIQUE (job_id, request_id),
    FOREIGN KEY (job_id, candidate_id, attempt_id) REFERENCES intake_attempts,
    FOREIGN KEY (job_id, candidate_id, replacement_document_id) REFERENCES resume_documents,
    CHECK ((state = 'fulfilled') = (replacement_document_id IS NOT NULL))
) STRICT;
CREATE UNIQUE INDEX IF NOT EXISTS open_replacement ON replacement_requests(job_id, candidate_id)
    WHERE state = 'open';

CREATE TABLE IF NOT EXISTS scoring_bases (
    job_id TEXT NOT NULL REFERENCES jobs,
    basis_id TEXT NOT NULL,
    profile_version INTEGER NOT NULL,
    matcher_version TEXT NOT NULL,
    scorer_version TEXT NOT NULL,
    config_version TEXT NOT NULL,
    settings_json TEXT NOT NULL CHECK (json_valid(settings_json)),
    fingerprint TEXT NOT NULL,
    PRIMARY KEY (job_id, basis_id),
    UNIQUE (job_id, fingerprint),
    FOREIGN KEY (job_id, profile_version) REFERENCES requirement_profiles
) STRICT;

CREATE TABLE IF NOT EXISTS scoring_runs (
    job_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    basis_id TEXT NOT NULL,
    mode TEXT NOT NULL CHECK (mode IN ('full','incremental')),
    plan_fingerprint TEXT NOT NULL,
    requested_by TEXT NOT NULL REFERENCES actors,
    state TEXT NOT NULL CHECK (state IN ('queued','running','completed','completed_with_errors','failed')),
    error_json TEXT CHECK (json_valid(error_json)),
    created_at TEXT NOT NULL,
    started_at TEXT,
    finished_at TEXT,
    lease_token TEXT,
    lease_until TEXT,
    PRIMARY KEY (job_id, run_id),
    UNIQUE (job_id, run_id, basis_id),
    FOREIGN KEY (job_id, basis_id) REFERENCES scoring_bases
) STRICT;
CREATE UNIQUE INDEX IF NOT EXISTS active_run ON scoring_runs(job_id) WHERE state IN ('queued','running');
CREATE INDEX IF NOT EXISTS scoring_queue ON scoring_runs(state, lease_until, created_at);
CREATE INDEX IF NOT EXISTS run_history ON scoring_runs(job_id, created_at);

CREATE TABLE IF NOT EXISTS scoring_run_candidates (
    job_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    candidate_id TEXT NOT NULL,
    basis_id TEXT NOT NULL,
    attempt_id TEXT NOT NULL,
    clearance_id TEXT,
    planned_action TEXT NOT NULL CHECK (planned_action IN ('score','reuse','exclude')),
    excluded_reason TEXT,
    captured_risk TEXT,
    finding_summary_json TEXT NOT NULL CHECK (json_valid(finding_summary_json)),
    state TEXT NOT NULL CHECK (state IN ('pending','scored','reused','excluded','failed','skipped')),
    score_id TEXT,
    error_code TEXT,
    PRIMARY KEY (job_id, run_id, candidate_id),
    UNIQUE (job_id, run_id, candidate_id, attempt_id),
    FOREIGN KEY (job_id, run_id, basis_id) REFERENCES scoring_runs(job_id, run_id, basis_id),
    FOREIGN KEY (job_id, candidate_id, attempt_id) REFERENCES intake_attempts,
    FOREIGN KEY (job_id, candidate_id, attempt_id, clearance_id) REFERENCES intake_review_dispositions,
    FOREIGN KEY (job_id, candidate_id, attempt_id, basis_id, score_id)
        REFERENCES candidate_scores(job_id, candidate_id, attempt_id, basis_id, score_id),
    CHECK ((state IN ('scored','reused')) = (score_id IS NOT NULL))
) STRICT;

CREATE TABLE IF NOT EXISTS candidate_scores (
    job_id TEXT NOT NULL,
    score_id TEXT NOT NULL,
    origin_run_id TEXT NOT NULL,
    candidate_id TEXT NOT NULL,
    attempt_id TEXT NOT NULL,
    basis_id TEXT NOT NULL,
    fit_score REAL NOT NULL CHECK (fit_score BETWEEN 0 AND 1),
    tier TEXT NOT NULL CHECK (tier IN ('A','B','C')),
    mandatory_coverage REAL,
    counterfactual_delta REAL,
    links_json TEXT NOT NULL CHECK (json_valid(links_json)),
    explanations_json TEXT NOT NULL CHECK (json_valid(explanations_json)),
    created_at TEXT NOT NULL,
    PRIMARY KEY (job_id, score_id),
    UNIQUE (job_id, origin_run_id, candidate_id),
    UNIQUE (job_id, candidate_id, attempt_id, basis_id, score_id),
    FOREIGN KEY (job_id, origin_run_id, basis_id) REFERENCES scoring_runs(job_id, run_id, basis_id),
    FOREIGN KEY (job_id, origin_run_id, candidate_id, attempt_id)
        REFERENCES scoring_run_candidates(job_id, run_id, candidate_id, attempt_id),
    FOREIGN KEY (job_id, candidate_id, attempt_id) REFERENCES verified_profiles
) STRICT;

CREATE TABLE IF NOT EXISTS rankings (
    job_id TEXT NOT NULL,
    ranking_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    published_at TEXT NOT NULL,
    PRIMARY KEY (job_id, ranking_id),
    UNIQUE (job_id, run_id),
    UNIQUE (job_id, ranking_id, run_id),
    FOREIGN KEY (job_id, run_id) REFERENCES scoring_runs
) STRICT;
CREATE INDEX IF NOT EXISTS ranking_history ON rankings(job_id, published_at);

CREATE TABLE IF NOT EXISTS ranking_rows (
    job_id TEXT NOT NULL,
    ranking_id TEXT NOT NULL,
    candidate_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    rank INTEGER,
    unranked_reason TEXT,
    PRIMARY KEY (job_id, ranking_id, candidate_id),
    UNIQUE (job_id, ranking_id, rank),
    UNIQUE (job_id, ranking_id, candidate_id, rank),
    FOREIGN KEY (job_id, ranking_id, run_id) REFERENCES rankings(job_id, ranking_id, run_id),
    FOREIGN KEY (job_id, run_id, candidate_id) REFERENCES scoring_run_candidates,
    CHECK ((rank IS NOT NULL AND rank >= 1 AND unranked_reason IS NULL)
        OR (rank IS NULL AND unranked_reason IS NOT NULL))
) STRICT;

CREATE TABLE IF NOT EXISTS shortlist_proposals (
    job_id TEXT NOT NULL,
    proposal_id TEXT NOT NULL,
    ranking_id TEXT NOT NULL,
    proposal_no INTEGER NOT NULL,
    state TEXT NOT NULL CHECK (state IN ('draft','approved','superseded')),
    current_revision_no INTEGER NOT NULL,
    approval_baseline_id TEXT,
    previous_ranking_id TEXT,
    comparison_proposal_id TEXT,
    comparison_revision_no INTEGER,
    created_at TEXT NOT NULL,
    PRIMARY KEY (job_id, proposal_id),
    UNIQUE (job_id, ranking_id, proposal_no),
    UNIQUE (job_id, proposal_id, ranking_id),
    FOREIGN KEY (job_id, ranking_id) REFERENCES rankings,
    FOREIGN KEY (job_id, approval_baseline_id) REFERENCES shortlist_approvals,
    FOREIGN KEY (job_id, previous_ranking_id) REFERENCES rankings,
    FOREIGN KEY (job_id, comparison_proposal_id, comparison_revision_no) REFERENCES shortlist_proposal_revisions,
    FOREIGN KEY (job_id, proposal_id, current_revision_no) REFERENCES shortlist_proposal_revisions
        DEFERRABLE INITIALLY DEFERRED
) STRICT;
CREATE UNIQUE INDEX IF NOT EXISTS open_proposal ON shortlist_proposals(job_id, ranking_id) WHERE state = 'draft';

CREATE TABLE IF NOT EXISTS shortlist_proposal_revisions (
    job_id TEXT NOT NULL,
    proposal_id TEXT NOT NULL,
    revision_no INTEGER NOT NULL,
    size INTEGER CHECK (size >= 1),
    overrides_json TEXT NOT NULL CHECK (json_valid(overrides_json)),
    prior_override_decisions_json TEXT NOT NULL CHECK (json_valid(prior_override_decisions_json)),
    saved_by TEXT NOT NULL REFERENCES actors,
    saved_at TEXT NOT NULL,
    PRIMARY KEY (job_id, proposal_id, revision_no),
    FOREIGN KEY (job_id, proposal_id) REFERENCES shortlist_proposals
) STRICT;

CREATE TABLE IF NOT EXISTS shortlist_approvals (
    job_id TEXT NOT NULL,
    shortlist_id TEXT NOT NULL,
    ranking_id TEXT NOT NULL,
    proposal_id TEXT NOT NULL,
    revision_no INTEGER NOT NULL,
    supersedes_shortlist_id TEXT,
    effective_overrides_json TEXT NOT NULL CHECK (json_valid(effective_overrides_json)),
    approved_by TEXT NOT NULL REFERENCES actors,
    approved_at TEXT NOT NULL,
    PRIMARY KEY (job_id, shortlist_id),
    UNIQUE (job_id, proposal_id),
    UNIQUE (job_id, shortlist_id, ranking_id),
    FOREIGN KEY (job_id, proposal_id, ranking_id) REFERENCES shortlist_proposals(job_id, proposal_id, ranking_id),
    FOREIGN KEY (job_id, proposal_id, revision_no) REFERENCES shortlist_proposal_revisions,
    FOREIGN KEY (job_id, supersedes_shortlist_id) REFERENCES shortlist_approvals
) STRICT;
CREATE INDEX IF NOT EXISTS approval_history ON shortlist_approvals(job_id, approved_at);

CREATE TABLE IF NOT EXISTS shortlist_members (
    job_id TEXT NOT NULL,
    shortlist_id TEXT NOT NULL,
    candidate_id TEXT NOT NULL,
    ranking_id TEXT NOT NULL,
    rank INTEGER NOT NULL,
    PRIMARY KEY (job_id, shortlist_id, candidate_id),
    FOREIGN KEY (job_id, shortlist_id, ranking_id)
        REFERENCES shortlist_approvals(job_id, shortlist_id, ranking_id),
    FOREIGN KEY (job_id, ranking_id, candidate_id, rank)
        REFERENCES ranking_rows(job_id, ranking_id, candidate_id, rank)
) STRICT;

CREATE TABLE IF NOT EXISTS audit_events (
    job_id TEXT NOT NULL REFERENCES jobs,
    event_no INTEGER NOT NULL,
    event_id TEXT NOT NULL UNIQUE,
    event_type TEXT NOT NULL,
    actor_id TEXT NOT NULL REFERENCES actors,
    recorded_at TEXT NOT NULL,
    references_json TEXT NOT NULL CHECK (json_valid(references_json)),
    details_json TEXT NOT NULL CHECK (json_valid(details_json)),
    prev_hash TEXT NOT NULL,
    event_hash TEXT NOT NULL,
    PRIMARY KEY (job_id, event_no)
) STRICT;

CREATE TABLE IF NOT EXISTS idempotency_keys (
    actor_id TEXT NOT NULL REFERENCES actors,
    method TEXT NOT NULL,
    resource_path TEXT NOT NULL,
    idempotency_key TEXT NOT NULL,
    job_id TEXT REFERENCES jobs,
    request_hash TEXT NOT NULL,
    response_status INTEGER NOT NULL,
    location TEXT,
    response_json TEXT NOT NULL CHECK (json_valid(response_json)),
    recorded_at TEXT NOT NULL,
    PRIMARY KEY (actor_id, method, resource_path, idempotency_key)
) STRICT;

CREATE TABLE IF NOT EXISTS workflow_secrets (
    name TEXT PRIMARY KEY,
    value BLOB NOT NULL
) STRICT;

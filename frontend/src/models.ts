export interface Failure {
  code: string;
  message: string;
  retryable: boolean;
  field_errors?: { field: string; message: string }[];
}

export interface Action {
  allowed: boolean;
  reason_code: string | null;
}

export interface Page<T> {
  items: T[];
  next_cursor: string | null;
}

export type DraftStatus = "none" | "queued" | "running" | "ready" | "failed";

export interface Job {
  job_id: string;
  role_title: string | null;
  draft: { status: DraftStatus; href: string };
  active_profile: { profile_id: string; version: number } | null;
  ranking_state: "none" | "current" | "rescore_required";
  current_ranking_id: string | null;
  latest_ranking_id: string | null;
  latest_scoring_run_id: string | null;
  current_proposal_id: string | null;
  last_approved_shortlist_id: string | null;
  actions: Record<string, Action>;
}

export type RequirementKind = "skill" | "experience" | "education" | "leadership" | "achievement";
export type Provenance = "jd_qualification" | "jd_responsibility" | "model_suggestion" | "hr_added";
export interface Option {
  option_id?: string;
  text: string;
  preferred: boolean;
}
export interface FlagReview {
  flag_id: string;
  disposition: "addressed" | "acknowledged";
}
export interface Requirement {
  requirement_id?: string;
  statement: string;
  kind: RequirementKind;
  options: Option[];
  combine: "all_of" | "any_of";
  substitutes: Option[];
  min_years: number | null;
  min_evidence: number | null;
  priority: "M" | "P";
  weight: number;
  provenance: Provenance;
  quote: string | null;
  rationale: string | null;
  onet_source: string | null;
  status: "active" | "inactive";
  flags: { flag_id: string; message: string }[];
}
export interface RequirementContent {
  role_title: string;
  requirements: Requirement[];
  not_used_for_scoring: string[];
  flag_reviews: FlagReview[];
}
export type Draft = { status: "none" } | (
  { jd_id: string; revision: string; base_version: number | null; stage: string | null } & (
    { status: "queued" | "running" }
    | { status: "failed"; error: Failure; actions: Record<string, Action> }
    | ({ status: "ready" } & RequirementContent)
  )
);
export interface Profile extends RequirementContent {
  jd_id: string;
  version: number;
  profile_id: string;
  approved_by: string;
  approved_at: string;
}
export interface JobDescription {
  jd_id: string;
  media_type: string;
  text: string | null;
}

export type IntakeStatus = "processing" | "ready_to_score" | "review_required" | "quarantined";
export type Risk = "low" | "medium" | "high";
export interface Attempt {
  candidate_id: string;
  attempt_id: string;
  stage: string | null;
  status: IntakeStatus;
  overall_risk: Risk | null;
  hold_code: string | null;
  issue_codes: string[];
}
export interface Finding {
  finding_id: string;
  layer: string;
  category: string;
  attack_family: string | null;
  risk: Risk;
  action_code: string;
}
export interface Disposition {
  disposition_id: string;
  action: "clear_for_scoring" | "keep_on_hold";
  reason: string;
  actor_id: string;
  recorded_at: string;
}
export interface Candidate extends Attempt {
  revision: string;
  latest_disposition: Disposition | null;
  security_flags: Finding[];
  replacement_request_id: string | null;
  actions: Record<string, Action>;
}
export interface Batch {
  batch_id: string;
  confirmed_profile_version: number;
  uploaded_at: string;
  uploaded_by: string;
  status_counts: Partial<Record<IntakeStatus, number>>;
}
export interface BatchDetail extends Batch {
  files: (Attempt & { upload_position: number })[];
}
export interface SourceSpan {
  span_id: string;
  page: number | null;
  visibility: string;
  excerpt_href?: string;
}
export interface VerifiedFact {
  fact_id: string;
  statement: string;
  kind: string;
  evidence_level: number;
  dates: string[];
  spans: { location: string; quote: string }[];
}
export interface IntakeAudit extends Attempt {
  outcome: string | null;
  document_sha256: string;
  provenance: { stopped_at?: string; model?: string; l3_model?: string; pipeline_schema?: string };
  error: Failure | null;
  verified_profile: { facts: VerifiedFact[] } | null;
  findings: (Finding & { spans: SourceSpan[] })[];
  exclusions: SourceSpan[];
  dispositions: Disposition[];
}
export interface Excerpt {
  span_id: string;
  excerpt: string;
  page: number | null;
}

export interface ScorePlan {
  profile_version: number | null;
  role_title: string | null;
  mode: "full" | "incremental" | null;
  pool_count: number;
  ready_count: number;
  to_score_count: number;
  reuse_count: number;
  excluded_counts: Partial<Record<IntakeStatus, number>>;
  can_start: boolean;
  reason_code: string | null;
  plan_token: string | null;
}
export interface Run {
  run_id: string;
  ranking_id: string | null;
  status: "queued" | "running" | "completed" | "completed_with_errors" | "failed";
  mode: "full" | "incremental";
  profile_version: number;
  pool_count: number;
  ready_count: number;
  scored_count: number;
  reused_count: number;
  skipped_count: number;
  error_count: number;
  finished_at: string | null;
  error: Failure | null;
}
export interface Ranking {
  ranking_id: string;
  run_id: string;
  role_title: string;
  profile_version: number;
  matcher_version: string;
  config_version: string;
  published_at: string;
  finished_at: string;
  ranked_count: number;
  unranked_count: number;
  unranked_counts: Record<string, number>;
  security_finding_count: number;
  flagged_candidate_count: number;
}
export interface EvidenceCitation {
  quote: string | null;
  location: string | null;
  page?: number | null;
}
export interface CriterionResult extends EvidenceCitation {
  option_id: string;
  text: string;
  evidence_fact: string | null;
  declared_level: number | null;
  effective_level: number | null;
  adjustment: string | null;
  is_substitute: boolean;
  duration?: {
    reported: string;
    scope: string;
    evidence_fact: string;
    spans: EvidenceCitation[];
  } | null;
}
export interface Assessment {
  requirement_id: string;
  statement: string;
  priority: "M" | "P";
  assessment: "met" | "not_met" | "not_assessed";
  credit: number | null;
  via_substitute: boolean;
  gap: string | null;
  min_years?: number | null;
  evidence_met?: boolean | null;
  deciding: CriterionResult | null;
  criteria: CriterionResult[];
}
export type RankingCandidate = {
  candidate_id: string;
  attempt_id: string;
  overall_risk: Risk | null;
  security_flags: Finding[];
  audit_href?: string;
} & (
  { status: "unranked"; reason_code: string }
  | { status: "ranked"; rank: number; fit_score: number; tier: "A" | "B" | "C"; unassessed: string[];
      assessments?: Assessment[]; counterfactual_delta?: number; mandatory_coverage?: number | null }
);

export interface OverrideInput {
  candidate_id: string;
  action: "include" | "exclude";
  reason: string;
}
export interface Override extends OverrideInput {
  override_id: string;
  actor_id: string;
  recorded_at: string;
}
export interface OverrideDecision {
  override_id: string;
  decision: "reconfirm" | "drop";
}
export interface RankChange {
  candidate_id: string;
  old_rank: number | null;
  new_rank: number | null;
}
export interface MembershipDiff {
  entered: RankChange[];
  exited: RankChange[];
  rank_changes: RankChange[];
}
export interface Proposal {
  proposal_id: string;
  ranking_id: string;
  revision: string;
  status: "draft" | "approved" | "superseded";
  size: number | null;
  top_n_ids: string[] | null;
  near_cutoff_ids: string[] | null;
  effective_member_ids: string[] | null;
  shortlist_id: string | null;
  overrides: Override[];
  prior_override_decisions: OverrideDecision[];
  prior_overrides: (Override & { decision: OverrideDecision["decision"] | null; can_reconfirm: boolean })[];
  comparison: { ranking_id: string; shortlist_id: string | null; computed_top_n?: MembershipDiff;
    effective_membership?: MembershipDiff } | null;
}
export interface Shortlist {
  shortlist_id: string;
  ranking_id: string;
  profile_version: number;
  size: number;
  top_n_ids: string[];
  near_cutoff_ids: string[];
  approved_member_ids: string[];
  approved_by: string;
  approved_at: string;
  supersedes_shortlist_id: string | null;
  overrides: Override[];
  prior_override_decisions: OverrideDecision[];
  export_formats?: string[];
}
export interface AuditEvent {
  event_no: number;
  event_type: string;
  actor_id: string;
  recorded_at: string;
}
export interface RankingAudit {
  ranking: Ranking & { requested_by: string };
  run: Run;
  events: AuditEvent[];
  profile_approval: { approved_by: string; approved_at: string; version: number };
}
export interface ShortlistAudit {
  events: AuditEvent[];
  proposal_history: {
    revision_no: number; size: number | null; saved_by: string; saved_at: string;
    overrides: Override[]; prior_override_decisions: OverrideDecision[];
  }[];
}

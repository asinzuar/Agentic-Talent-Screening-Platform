import { useState } from "react";
import { Link, useSearchParams } from "react-router";
import { useQueryClient } from "@tanstack/react-query";
import {
  ArrowRight, ArrowUpRight, ChartNoAxesCombined, Check, ClipboardCheck, Download, FileUp, History,
  ListOrdered, LockKeyhole, Plus, Scale, ShieldAlert, SlidersHorizontal,
} from "lucide-react";
import { jobApi, jobKey, jobUrl, useCollection, useCommand, useJob, useRead, useSession } from "./api";
import { IntakeAudit } from "./IntakeAudit";
import type {
  Assessment, AuditEvent, CriterionResult, EvidenceCitation, MembershipDiff, OverrideDecision, OverrideInput, Proposal,
  Ranking, RankingAudit, RankingCandidate, RankChange, Shortlist, ShortlistAudit,
} from "./models";
import {
  actionReason, Button, ButtonLink, Chip, dateTime, Dialog, Empty, ErrorNotice, humanize,
  Loading, LoadMore, Notice, plural, StateChip,
} from "./ui";

const percent = (score: number) => `${(score * 100).toFixed(1).replace(/\.0$/, "")}%`;
type RankedRow = Extract<RankingCandidate, { status: "ranked" }>;

function Citation({ quote, page, location }: EvidenceCitation) {
  return <>
    {quote && <blockquote>{quote}</blockquote>}
    {(page || location) && <p className="small muted">{page ? `Page ${page}` : ""}
      {page && location ? " · " : ""}{location}</p>}
  </>;
}

function Evidence({ item }: { item: CriterionResult }) {
  const level = item.effective_level ?? item.declared_level;
  const duration = item.duration;
  return <div className="evidence-snippet">
    <div className="chip-row">{level != null && <Chip tone="blue">E{level}</Chip>}
      {item.is_substitute && <Chip tone="violet">Substitute</Chip>} <span className="small">{item.text}</span></div>
    {(!duration || duration.evidence_fact !== item.evidence_fact) && <Citation {...item} />}
    {duration && <>
      <p className="small">Self-reported duration: <strong>{duration.reported}</strong> · {duration.scope}</p>
      {duration.spans.map((span) => <Citation key={span.location} {...span} />)}
    </>}
    {item.adjustment && <small className="muted">{item.adjustment}</small>}
  </div>;
}

function AssessmentCard({ assessment }: { assessment: Assessment }) {
  const durationNotEstablished = assessment.assessment === "not_met" && assessment.evidence_met && assessment.min_years != null;
  return <section className="assessment-card stack">
    <div className="toolbar"><div><h3>{assessment.statement}</h3><p className="muted small">{assessment.priority === "M" ? "Mandatory" : "Preferred"}</p></div>
      {durationNotEstablished ? <Chip tone="amber">Duration not established</Chip> : <StateChip value={assessment.assessment} />}</div>
    {assessment.deciding && <Evidence item={assessment.deciding} />}
    {assessment.gap && <p className="small muted">{assessment.gap}</p>}
    {assessment.criteria.length > 1 && <details className="detail-disclosure"><summary>All options and substitutes</summary>
      {assessment.criteria.map((criterion) => <Evidence key={criterion.option_id} item={criterion} />)}
    </details>}
  </section>;
}

function CandidateEvidence({ ranking, row, onClose }: { ranking: Ranking; row: RankedRow; onClose: () => void }) {
  const job = useJob();
  const query = useRead<RankingCandidate>([...jobKey(job.job_id), "ranked-candidate", ranking.ranking_id, row.candidate_id],
    `${jobApi(job.job_id)}/rankings/${ranking.ranking_id}/candidates/${row.candidate_id}`);
  const [showAudit, setShowAudit] = useState(false);
  const result = query.data;
  return <Dialog title={`Candidate evidence · ${row.candidate_id}`}
    description={`Rank #${row.rank} · Requirements v${ranking.profile_version} · ${dateTime(ranking.published_at)}`}
    onClose={onClose} wide>
    {query.isPending ? <Loading>Loading requirement evidence…</Loading>
      : query.isError ? <ErrorNotice error={query.error} retry={() => void query.refetch()} />
      : result?.status === "ranked" && <div className="stack">
        <div className="evidence-summary">
          <div><strong>{percent(result.fit_score)}</strong><span>Fit score</span></div><Chip tone="violet">Tier {result.tier}</Chip>
          {result.mandatory_coverage != null && <Chip tone="blue">{percent(result.mandatory_coverage)} mandatory coverage</Chip>}
          {!!result.unassessed.length && <Chip tone="amber">{result.unassessed.length} not assessed</Chip>}
          {!!result.security_flags.length && <Chip tone="amber">{plural(result.security_flags.length, "finding")}</Chip>}
        </div>
        {result.counterfactual_delta != null && <Notice>Computed counterfactual score difference: {result.counterfactual_delta.toFixed(3)}</Notice>}
        {result.assessments?.map((assessment) => <AssessmentCard key={assessment.requirement_id} assessment={assessment} />)}
        {result.audit_href && <div className="stack">
          <Button variant="secondary" onClick={() => setShowAudit(!showAudit)} aria-expanded={showAudit}>
            <ShieldAlert size={16} /> {showAudit ? "Hide intake audit" : "View intake audit"}
          </Button>
          {showAudit && <IntakeAudit jobId={job.job_id} path={result.audit_href} />}
        </div>}
      </div>}
  </Dialog>;
}

function EventTrail({ events }: { events: AuditEvent[] }) {
  return events.length ? <ul className="timeline">{events.map((event) => <li key={event.event_no}>
    <strong>{humanize(event.event_type)}</strong><small>{event.actor_id} · {dateTime(event.recorded_at)}</small>
  </li>)}</ul> : <p className="muted small">No events are available for this snapshot.</p>;
}

function RunAudit({ ranking, onClose }: { ranking: Ranking; onClose: () => void }) {
  const job = useJob();
  const query = useRead<RankingAudit>([...jobKey(job.job_id), "ranking-audit", ranking.ranking_id],
    `${jobApi(job.job_id)}/rankings/${ranking.ranking_id}/audit`);
  return <Dialog title="Ranking audit" description={`Requirements v${ranking.profile_version} · ${dateTime(ranking.published_at)}`}
    onClose={onClose} wide>
    {query.isPending ? <Loading>Loading audit…</Loading> : query.isError
      ? <ErrorNotice error={query.error} retry={() => void query.refetch()} />
      : <div className="stack">
        <dl className="detail-grid">
          <div><dt>Ranking snapshot</dt><dd className="code-text break-text">{ranking.ranking_id}</dd></div>
          <div><dt>Run ID</dt><dd className="code-text break-text">{ranking.run_id}</dd></div>
          <div><dt>Run requested by</dt><dd>{query.data.ranking.requested_by}</dd></div>
          <div><dt>Requirements approved by</dt><dd>{query.data.profile_approval.approved_by} · {dateTime(query.data.profile_approval.approved_at)}</dd></div>
          <div><dt>Matcher</dt><dd className="break-text">{ranking.matcher_version}</dd></div>
          <div><dt>Scoring configuration</dt><dd>{ranking.config_version}</dd></div>
        </dl>
        <Notice>{query.data.run.scored_count} newly scored · {query.data.run.reused_count} reused · {query.data.run.error_count} errors</Notice>
        <h3>Run events</h3><EventTrail events={query.data.events} />
      </div>}
  </Dialog>;
}

function ShortlistSnapshot({ id, onClose, onRanking }: { id: string; onClose: () => void; onRanking: (snapshot: Shortlist) => void }) {
  const job = useJob();
  const { api } = useSession();
  const path = `${jobApi(job.job_id)}/shortlists/${id}`;
  const query = useRead<Shortlist>([...jobKey(job.job_id), "shortlist", id], path);
  const [showAudit, setShowAudit] = useState(false);
  const audit = useRead<ShortlistAudit>([...jobKey(job.job_id), "shortlist-audit", id], showAudit ? `${path}/audit` : null);
  const [downloadError, setDownloadError] = useState<unknown>(null);
  const [downloading, setDownloading] = useState(false);
  return <Dialog title="Approved shortlist" description="A saved snapshot of the reviewed membership." onClose={onClose} wide>
    {query.isPending ? <Loading>Loading the approved shortlist…</Loading> : query.isError
      ? <ErrorNotice error={query.error} retry={() => void query.refetch()} />
      : <div className="stack">
        <div className="chip-row"><Chip tone="green"><LockKeyhole size={13} /> Approved</Chip>
          <Chip tone="blue">Requirements v{query.data.profile_version}</Chip>{job.last_approved_shortlist_id !== id && <Chip>Historical</Chip>}</div>
        <div className="confirmation-context"><span><strong>{plural(query.data.approved_member_ids.length, "approved candidate")}</strong>
          <small>Approved by {query.data.approved_by} · {dateTime(query.data.approved_at)}</small></span></div>
        <dl className="detail-grid">
          <div><dt>Shortlist snapshot</dt><dd className="code-text break-text">{id}</dd></div>
          <div><dt>Ranking snapshot</dt><dd className="code-text break-text">{query.data.ranking_id}</dd></div>
        </dl>
        <section className="audit-section stack"><h3>Approved membership</h3>
          {query.data.approved_member_ids.length > 0 ? <div className="chip-row">
            {query.data.approved_member_ids.map((candidate) => <Chip key={candidate} tone="green">{candidate}</Chip>)}
          </div> : <p className="muted">No candidates were included in this approval.</p>}
        </section>
        {!!query.data.overrides.length && <section className="audit-section stack"><h3>Applied overrides</h3>
          <ul className="timeline">{query.data.overrides.map((item) => <li key={item.override_id}>
            <strong>{humanize(item.action)} {item.candidate_id}</strong><p>{item.reason}</p>
            <small>{item.actor_id} · {dateTime(item.recorded_at)}</small>
          </li>)}</ul></section>}
        <ErrorNotice error={downloadError} />
        <div className="button-row">
          <Button variant="secondary" onClick={() => onRanking(query.data)}>View matching ranking <ArrowUpRight size={15} /></Button>
          <Button variant="secondary" onClick={() => setShowAudit(!showAudit)} aria-expanded={showAudit}>
            <History size={16} /> {showAudit ? "Hide audit" : "View audit"}</Button>
          {query.data.export_formats?.includes("json") && <Button busy={downloading} onClick={async () => {
            setDownloading(true); setDownloadError(null);
            try {
              const blob = await api.download(`${path}/export?format=json`);
              const url = URL.createObjectURL(blob);
              const anchor = document.createElement("a");
              anchor.href = url; anchor.download = `shortlist-${id}.json`; anchor.click();
              setTimeout(() => URL.revokeObjectURL(url), 1000);
              if (showAudit) void audit.refetch();
            } catch (error) { setDownloadError(error); } finally { setDownloading(false); }
          }}><Download size={17} /> Export JSON</Button>}
        </div>
        {showAudit && <section className="audit-section stack">
          <h3>Shortlist audit</h3>
          {audit.isPending ? <Loading>Loading audit…</Loading> : audit.isError
            ? <ErrorNotice error={audit.error} retry={() => void audit.refetch()} />
            : <><EventTrail events={audit.data.events} />
              <h3>Proposal revisions</h3><ul className="timeline">{audit.data.proposal_history.map((edit) => <li key={edit.revision_no}>
                <strong>Revision {edit.revision_no}{edit.size != null && ` · Top ${edit.size}`}</strong>
                <small>{edit.saved_by} · {dateTime(edit.saved_at)}</small>
                {edit.overrides.map((item) => <p key={item.override_id}>{humanize(item.action)} {item.candidate_id} · {item.reason}</p>)}
                {edit.prior_override_decisions.length > 0 && <p className="muted">
                  {edit.prior_override_decisions.filter((item) => item.decision === "reconfirm").length} reconfirmed · {edit.prior_override_decisions.filter((item) => item.decision === "drop").length} dropped
                </p>}
              </li>)}</ul></>}
        </section>}
      </div>}
  </Dialog>;
}

function HistoryDialog({ onClose, onRanking, onShortlist }: {
  onClose: () => void; onRanking: (id: string) => void; onShortlist: (id: string) => void;
}) {
  const job = useJob();
  const path = jobApi(job.job_id);
  const rankings = useCollection<Ranking>([...jobKey(job.job_id), "rankings"], `${path}/rankings`);
  const shortlists = useCollection<Shortlist>([...jobKey(job.job_id), "shortlists"], `${path}/shortlists`);
  return <Dialog title="Ranking and shortlist history" description="Open a snapshot with the requirements version used at the time."
    onClose={onClose} wide>
    <div className="history-grid">
      <section className="stack"><h3>Rankings</h3>
        {rankings.isPending ? <Loading /> : rankings.isError ? <ErrorNotice error={rankings.error} retry={() => void rankings.refetch()} />
          : !rankings.items.length ? <p className="muted">No rankings published yet.</p>
          : <ul className="history-list">{rankings.items.map((rank) => <li key={rank.ranking_id}>
            <Button variant="ghost" className="history-item" onClick={() => onRanking(rank.ranking_id)}>
              <span><strong>Requirements v{rank.profile_version}</strong><small>{dateTime(rank.published_at)} · {rank.ranked_count} ranked</small></span>
              {rank.ranking_id === job.current_ranking_id ? <Chip tone="blue">Current</Chip> : <ArrowUpRight size={16} />}
            </Button></li>)}</ul>}
        <LoadMore query={rankings} />
      </section>
      <section className="stack"><h3>Approved shortlists</h3>
        {shortlists.isPending ? <Loading /> : shortlists.isError ? <ErrorNotice error={shortlists.error} retry={() => void shortlists.refetch()} />
          : !shortlists.items.length ? <p className="muted">No shortlist has been approved.</p>
          : <ul className="history-list">{shortlists.items.map((item) => <li key={item.shortlist_id}>
            <Button variant="ghost" className="history-item" onClick={() => onShortlist(item.shortlist_id)}>
              <span><strong>{plural(item.approved_member_ids.length, "candidate")} · v{item.profile_version}</strong><small>{dateTime(item.approved_at)} · {item.approved_by}</small></span>
              {item.shortlist_id === job.last_approved_shortlist_id ? <Chip tone="green">Latest</Chip> : <ArrowUpRight size={16} />}
            </Button></li>)}</ul>}
        <LoadMore query={shortlists} />
      </section>
    </div>
  </Dialog>;
}

function RankChanges({ diff, label }: { diff: MembershipDiff; label: string }) {
  const groups: { label: string; rows: RankChange[] }[] = [
    { label: "Entered", rows: diff.entered }, { label: "Left", rows: diff.exited }, { label: "Moved", rows: diff.rank_changes },
  ];
  return <details className="detail-disclosure comparison">
    <summary>{label} · {diff.entered.length} entered, {diff.exited.length} left</summary>
    <div className="comparison-groups">
      {groups.filter((group) => group.rows.length).map((group) => <div key={group.label}><h4>{group.label}</h4>
        <ul>{group.rows.map((row) => <li key={row.candidate_id}><span className="code-text">{row.candidate_id}</span>
          <span>{row.old_rank != null ? `#${row.old_rank}` : "Unranked"} <ArrowRight size={13} /> {row.new_rank != null ? `#${row.new_rank}` : "Unranked"}</span>
        </li>)}</ul></div>)}
      {!groups.some((group) => group.rows.length) && <p className="small muted">No changes from the comparison snapshot.</p>}
    </div>
  </details>;
}

interface ProposalEdit {
  size?: number;
  overrides?: OverrideInput[];
  decisions?: OverrideDecision[];
}

function ProposalPanel({ proposal, rankedCount, pending, error, onSave, onApprove, onApproval, onRevise }: {
  proposal: Proposal; rankedCount: number; pending: boolean; error: unknown;
  onSave: (edit: ProposalEdit) => Promise<boolean>; onApprove: () => void; onApproval: (id: string) => void; onRevise: () => void;
}) {
  const [size, setSize] = useState(proposal.size == null ? "" : String(proposal.size));
  const number = Number(size);
  const valid = Number.isSafeInteger(number) && number > 0;
  const unresolved = proposal.prior_overrides.filter((item) => !item.decision).length;
  const draft = proposal.status === "draft";
  return <section className="panel proposal-panel">
    <div className="panel-heading"><div className="heading-with-icon"><span className="section-icon"><ClipboardCheck size={18} /></span>
      <div><h2>{draft ? "Shortlist proposal" : "Approved shortlist"}</h2>
        <p>{draft ? "Review the computed cutoff and any membership overrides." : "This proposal has a saved approval."}</p></div></div></div>
    <div className="panel-body stack">
      {draft && <form className="size-form" onSubmit={(event) => { event.preventDefault(); if (valid) void onSave({ size: number }); }}>
        <label className="field">Shortlist size<input required min={1} step={1} type="number" value={size} onChange={(event) => setSize(event.target.value)} /></label>
        <Button type="submit" variant="secondary" busy={pending} disabled={!valid || number === proposal.size || !rankedCount}>
          {proposal.size == null ? "Set cutoff" : "Update cutoff"}
        </Button>
      </form>}
      {proposal.size != null ? <div className="shortlist-summary">
        <div><span>Computed top N</span><strong>{proposal.top_n_ids?.length ?? 0}</strong></div>
        <div><span>{draft ? "Proposed members" : "Approved members"}</span><strong>{proposal.effective_member_ids?.length ?? 0}</strong></div>
        <div><span>Near cutoff</span><strong>{proposal.near_cutoff_ids?.length ?? 0}</strong></div>
      </div> : <p className="muted small">{rankedCount ? "Choose a shortlist size to mark candidates in the ranked table." : "At least one ranked candidate is needed."}</p>}
      {proposal.comparison?.computed_top_n && <RankChanges diff={proposal.comparison.computed_top_n} label="Computed top N changes" />}
      {proposal.comparison?.effective_membership && <RankChanges diff={proposal.comparison.effective_membership} label="Membership changes" />}
      {!!proposal.prior_overrides.length && <section className="prior-overrides stack">
        <h3>Previous overrides {draft && unresolved > 0 && <Chip tone="amber">{unresolved} to review</Chip>}</h3>
        <p className="small muted">Review each previous decision against this ranking.</p>
        {proposal.prior_overrides.map((item) => <div className="override-card stack" key={item.override_id}>
          <div className="toolbar"><strong>{humanize(item.action)} {item.candidate_id}</strong>
            <Chip tone={item.decision ? "green" : "amber"}>{item.decision ? humanize(item.decision) : "Awaiting review"}</Chip></div>
          <p className="small">{item.reason}</p>
          <small className="muted">{item.actor_id} · {dateTime(item.recorded_at)}</small>
          {draft && <div className="button-row">
            {(["reconfirm", "drop"] as const).map((decision) => <Button key={decision} variant="secondary" disabled={pending || proposal.size == null ||
              item.decision === decision || (decision === "reconfirm" && !item.can_reconfirm)}
              onClick={() => void onSave({ decisions: [
                ...proposal.prior_override_decisions.filter((choice) => choice.override_id !== item.override_id),
                { override_id: item.override_id, decision },
              ] })}>{humanize(decision)}</Button>)}
          </div>}
          {draft && !item.can_reconfirm && <small className="muted">This candidate is no longer ranked.</small>}
        </div>)}
      </section>}
      <ErrorNotice error={error} />
      <div className="button-row">
        {draft ? <Button disabled={!proposal.size || !rankedCount || !!unresolved || number !== proposal.size || pending}
          onClick={onApprove}><LockKeyhole size={16} /> Approve shortlist</Button>
          : <><Button variant="secondary" disabled={!proposal.shortlist_id} onClick={() => proposal.shortlist_id && onApproval(proposal.shortlist_id)}>View approval</Button>
            <Button variant="secondary" busy={pending} onClick={onRevise}><Plus size={16} /> Revise shortlist</Button></>}
      </div>
      {draft && unresolved > 0 && <small className="muted">Resolve each previous override before approval.</small>}
    </div>
  </section>;
}

function OverrideDialog({ row, proposal, pending, error, onSave, onClose }: {
  row: RankedRow; proposal: Proposal; pending: boolean; error: unknown;
  onSave: (edit: ProposalEdit) => Promise<boolean>; onClose: () => void;
}) {
  const existing = proposal.overrides.find((item) => item.candidate_id === row.candidate_id);
  const [action, setAction] = useState<OverrideInput["action"]>(existing?.action ??
    (proposal.effective_member_ids?.includes(row.candidate_id) ? "exclude" : "include"));
  const [reason, setReason] = useState(existing?.reason ?? "");
  const overrides = proposal.overrides.filter((item) => item.candidate_id !== row.candidate_id);
  const reconfirmed = proposal.prior_overrides.some((item) => item.candidate_id === row.candidate_id && item.decision === "reconfirm");
  return <Dialog title={`Override · ${row.candidate_id}`} description={`Rank #${row.rank} · Fit score ${percent(row.fit_score)}`}
    onClose={onClose} busy={pending}>
    <form className="stack" onSubmit={async (event) => {
      event.preventDefault();
      if (reason.trim() && !reconfirmed && await onSave({ overrides: [...overrides, { candidate_id: row.candidate_id, action, reason: reason.trim() }] })) onClose();
    }}>
      <p>Change proposed membership while keeping the computed rank and score.</p>
      <label className="field">Membership action<select value={action} onChange={(event) => setAction(event.target.value as typeof action)}>
        <option value="include">Include in shortlist</option><option value="exclude">Exclude from shortlist</option>
      </select></label>
      <label className="field">Override reason<textarea required maxLength={4000} value={reason} rows={3}
        onChange={(event) => setReason(event.target.value)} /></label>
      {reconfirmed && <Notice tone="amber">Drop this candidate’s reconfirmed override before creating a new one.</Notice>}
      <ErrorNotice error={error} />
      <div className="dialog-actions">
        {existing && <Button variant="ghost" disabled={pending} onClick={async () => {
          if (await onSave({ overrides })) onClose();
        }}>Remove override</Button>}
        <Button variant="secondary" disabled={pending} onClick={onClose}>Cancel</Button>
        <Button type="submit" busy={pending} disabled={!reason.trim() || reconfirmed}>Save override</Button>
      </div>
    </form>
  </Dialog>;
}

function RankingView({ id, approvedId, onShortlist }: { id: string; approvedId: string | null; onShortlist: (id: string) => void }) {
  const job = useJob();
  const cache = useQueryClient();
  const path = `${jobApi(job.job_id)}/rankings/${id}`;
  const current = job.current_ranking_id === id;
  const proposalId = current && !approvedId ? job.current_proposal_id : null;
  const proposalKey = [...jobKey(job.job_id), "proposal", proposalId];
  const ranking = useRead<Ranking>([...jobKey(job.job_id), "ranking", id], path);
  const proposalQuery = useRead<Proposal>(proposalKey, proposalId ? `${jobApi(job.job_id)}/shortlist-proposals/${proposalId}` : null);
  const approvedQuery = useRead<Shortlist>([...jobKey(job.job_id), "shortlist", approvedId],
    approvedId ? `${jobApi(job.job_id)}/shortlists/${approvedId}` : null);
  const [view, setView] = useState<"ranked" | "unranked">("ranked");
  const rows = useCollection<RankingCandidate>([...jobKey(job.job_id), "ranking-rows", id, view], `${path}/candidates?status=${view}`);
  const [detail, setDetail] = useState<RankedRow | null>(null);
  const [override, setOverride] = useState<RankedRow | null>(null);
  const [audit, setAudit] = useState(false);
  const [approvalRevision, setApprovalRevision] = useState<string | null>(null);
  const command = useCommand(job.job_id);
  if (ranking.isPending) return <Loading>Loading ranking…</Loading>;
  if (ranking.isError) return <ErrorNotice error={ranking.error} retry={() => void ranking.refetch()} />;
  const data = ranking.data;
  const proposal = proposalQuery.data?.ranking_id === id ? proposalQuery.data : null;
  const approvalChanged = proposal?.revision !== approvalRevision || proposal?.status !== "draft";
  const approved = approvedQuery.data?.ranking_id === id ? approvedQuery.data : null;
  const members = new Set(approved?.approved_member_ids ?? proposal?.effective_member_ids ?? []);
  const near = new Set(approved?.near_cutoff_ids ?? proposal?.near_cutoff_ids ?? []);
  const overridden = new Set([
    ...(approved?.overrides ?? proposal?.overrides ?? []).map((item) => item.candidate_id),
    ...(proposal?.prior_overrides.filter((item) => item.decision === "reconfirm").map((item) => item.candidate_id) ?? []),
  ]);
  const isApproved = Boolean(approved) || proposal?.status === "approved";
  const proposalPath = proposal ? `${jobApi(job.job_id)}/shortlist-proposals/${proposal.proposal_id}` : "";
  async function saveProposal(edit: ProposalEdit) {
    if (!proposal || proposal.status !== "draft") return false;
    const size = edit.size ?? proposal.size;
    if (size == null) return false;
    const overrides = (edit.overrides ?? proposal.overrides).map(({ candidate_id, action, reason }) => ({ candidate_id, action, reason }));
    const prior = (edit.decisions ?? proposal.prior_override_decisions).map(({ override_id, decision }) => ({ override_id, decision }));
    const result = await command.submit<Proposal>("PUT", proposalPath,
      { expected_revision: proposal.revision, size, overrides, prior_override_decisions: prior }, false);
    if (result) {
      cache.setQueryData(proposalKey, result);
    } else {
      void proposalQuery.refetch();
    }
    return Boolean(result);
  }
  return <div className="page-stack">
    <section className="panel ranking-header">
      <div className="toolbar">
        <div className="heading-with-icon"><span className="section-icon"><ChartNoAxesCombined size={19} /></span>
          <div><h2>{data.role_title}</h2><p>Published {dateTime(data.published_at)}</p></div></div>
        <div className="chip-row"><Chip tone="blue">Requirements v{data.profile_version}</Chip>
          <Chip tone={current ? "green" : "amber"}>{current ? "Current ranking" : "Historical ranking"}</Chip>
          <Button variant="ghost" onClick={() => setAudit(true)}><History size={16} /> Run audit</Button></div>
      </div>
      <div className="ranking-stat-grid">
        <div><strong>{data.ranked_count}</strong><span>Ranked</span></div>
        <div><strong>{data.unranked_count}</strong><span>Unranked</span></div>
        <div><strong>{data.security_finding_count}</strong><span>Security findings</span></div>
        <div><strong>{data.flagged_candidate_count}</strong><span>Flagged candidates</span></div>
      </div>
    </section>
    {approved && <Notice tone="green">Showing approved membership from this exact snapshot. <Button variant="ghost" onClick={() => onShortlist(approved.shortlist_id)}>Open approval</Button></Notice>}
    {approvedQuery.error && <ErrorNotice error={approvedQuery.error} retry={() => void approvedQuery.refetch()} />}
    {proposalQuery.error && <ErrorNotice error={proposalQuery.error} retry={() => void proposalQuery.refetch()} />}
    <div className={`ranking-columns ${proposal ? "with-proposal" : ""}`}>
      <section className="panel ranking-table-panel">
        <div className="panel-heading toolbar"><div><h2>Candidate results</h2><p>Rank and score come from verified evidence.</p></div>
          <div className="segmented" aria-label="Result status">
            <button className={view === "ranked" ? "active" : ""} type="button" aria-pressed={view === "ranked"} onClick={() => setView("ranked")}>Ranked</button>
            <button className={view === "unranked" ? "active" : ""} type="button" aria-pressed={view === "unranked"} onClick={() => setView("unranked")}>Unranked</button>
          </div></div>
        {view === "unranked" && Object.keys(data.unranked_counts).length > 0 && <div className="reason-summary">
          {Object.entries(data.unranked_counts).map(([reason, count]) => <Chip key={reason} tone="amber">{humanize(reason)} · {count}</Chip>)}
        </div>}
        {rows.isPending ? <Loading>Loading results…</Loading> : rows.isError ? <ErrorNotice error={rows.error} retry={() => void rows.refetch()} />
          : rows.items.length === 0 ? <Empty icon={<ListOrdered size={24} />} title={view === "ranked" ? "No candidates ranked" : "No unranked candidates"}
            message={view === "ranked" ? "Check the unranked view for reasons and next actions." : "Every candidate in this snapshot has a score."} />
          : <div className="table-wrap"><table>
            <thead>{view === "ranked" ? <tr><th>Rank</th><th>Candidate</th><th>Fit</th><th>Tier</th><th>Shortlist</th>
              {proposal?.status === "draft" && proposal.size != null && <th className="numeric"><span className="visually-hidden">Override</span></th>}</tr>
              : <tr><th>Candidate</th><th>Reason unranked</th><th className="numeric">Intake</th></tr>}</thead>
            <tbody>{rows.items.map((row) => row.status === "ranked" ? <tr key={row.candidate_id}>
              <td><strong className="rank-number">#{row.rank}</strong></td>
              <td><Button variant="ghost" className="text-button code-text" onClick={() => setDetail(row)}>{row.candidate_id}</Button>
                <div className="cell-chips">
                  {!!row.security_flags.length && <Chip tone="amber">{plural(row.security_flags.length, "finding")}</Chip>}
                  {!!row.unassessed.length && <Chip tone="amber">{row.unassessed.length} not assessed</Chip>}
                </div></td>
              <td><strong className="score-number">{percent(row.fit_score)}</strong></td>
              <td><Chip tone={row.tier === "A" ? "green" : row.tier === "B" ? "blue" : "neutral"}>{row.tier}</Chip></td>
              <td><div className="cell-chips">
                {members.has(row.candidate_id) && <Chip tone={isApproved ? "green" : "blue"}>{isApproved ? "Approved" : "Proposed"}</Chip>}
                {near.has(row.candidate_id) && <Chip tone="violet">Near cutoff</Chip>}
                {overridden.has(row.candidate_id) && <Chip tone="amber">Override</Chip>}
                {!members.has(row.candidate_id) && !near.has(row.candidate_id) && !overridden.has(row.candidate_id) && <span className="muted">—</span>}
              </div></td>
              {proposal?.status === "draft" && proposal.size != null && <td className="numeric">
                <Button variant="ghost" aria-label={`Override ${row.candidate_id}`} onClick={() => { command.reset(); setOverride(row); }}>
                  <SlidersHorizontal size={16} /></Button></td>}
            </tr> : <tr key={row.candidate_id}>
              <td className="code-text">{row.candidate_id}</td><td><Chip tone="amber">{humanize(row.reason_code)}</Chip>
                {["input_changed", "scoring_failed"].includes(row.reason_code) && <small className="cell-note">{actionReason(row.reason_code)}</small>}</td>
              <td className="numeric"><Link className="row-link align-right" to={`${jobUrl(job.job_id)}/resumes?candidate=${row.candidate_id}`}>
                View intake <ArrowUpRight size={15} /></Link></td>
            </tr>)}</tbody></table></div>}
        <LoadMore query={rows} />
      </section>
      {proposal && <ProposalPanel key={proposal.revision} proposal={proposal} rankedCount={data.ranked_count}
        pending={command.pending} error={command.error} onSave={saveProposal}
        onApprove={() => { command.reset(); setApprovalRevision(proposal.revision); }}
        onApproval={onShortlist} onRevise={() => void command.submit("POST", `${path}/shortlist-proposals`, {})} />}
    </div>
    {detail && <CandidateEvidence ranking={data} row={detail} onClose={() => setDetail(null)} />}
    {audit && <RunAudit ranking={data} onClose={() => setAudit(false)} />}
    {override && proposal && <OverrideDialog row={override} proposal={proposal} pending={command.pending} error={command.error} onSave={saveProposal}
      onClose={() => { setOverride(null); command.reset(); }} />}
    {approvalRevision && proposal && <Dialog title="Approve this shortlist?"
      description={`Requirements v${data.profile_version} · ${plural(proposal.effective_member_ids?.length ?? 0, "candidate")}`}
      onClose={() => { setApprovalRevision(null); command.reset(); }} busy={command.pending}>
      <div className="stack">
        <p>This records the reviewed membership for this ranking. Earlier approvals remain available in history.</p>
        {approvalChanged && <Notice tone="amber">The proposal has changed. Close this dialog and review the updated membership.</Notice>}
        <ErrorNotice error={command.error} />
        <div className="dialog-actions"><Button variant="secondary" disabled={command.pending} onClick={() => setApprovalRevision(null)}>Cancel</Button>
          <Button busy={command.pending} disabled={approvalChanged} onClick={async () => {
            const result = await command.submit<Shortlist>("POST", `${proposalPath}/approve`, { expected_revision: approvalRevision });
            if (result) { setApprovalRevision(null); onShortlist(result.shortlist_id); }
            else void proposalQuery.refetch();
          }}><Check size={17} /> Confirm shortlist approval</Button></div>
      </div>
    </Dialog>}
  </div>;
}

export function Rankings() {
  const job = useJob();
  const [search, setSearch] = useSearchParams();
  const [history, setHistory] = useState(false);
  const [snapshot, setSnapshot] = useState<string | null>(null);
  const id = search.get("ranking") || job.current_ranking_id;
  const approvedId = search.get("approved");
  const resumes = `${jobUrl(job.job_id)}/resumes`;
  const latestApproval = job.last_approved_shortlist_id;
  return <div className="page-stack">
    <div className="toolbar tab-heading">
      <div><p className="eyebrow">DECISION REVIEW</p><h2>Rankings &amp; Shortlist</h2><p>Review evidence, compare pool results, and record shortlist decisions.</p></div>
      <div className="button-row">
        <Button variant="secondary" onClick={() => setHistory(true)}><History size={17} /> History</Button>
        {job.actions.upload_resumes?.allowed && <ButtonLink to={`${resumes}?upload=1`} primary><FileUp size={17} /> Add resumes</ButtonLink>}
      </div>
    </div>
    {latestApproval && <div className="toolbar revision-banner">
      <div><strong>Last approved shortlist is available</strong><p>A new run or proposal leaves the approved membership intact.</p></div>
      <Button variant="secondary" onClick={() => setSnapshot(latestApproval)}>View last approval <ArrowUpRight size={15} /></Button>
    </div>}
    {id && (id !== job.current_ranking_id || approvedId) && job.current_ranking_id && <div className="end-actions">
      <Button variant="secondary" onClick={() => setSearch({})}>Return to current ranking</Button>
    </div>}
    {id ? <RankingView key={`${id}:${approvedId ?? ""}`} id={id} approvedId={approvedId} onShortlist={setSnapshot} />
      : <section className="panel"><Empty icon={<Scale size={26} />}
        title={job.ranking_state === "rescore_required" ? "A full rescore is required" : "No ranking published yet"}
        message={job.ranking_state === "rescore_required"
          ? "Score the eligible pool on the active requirements before reviewing a current ranking. Earlier snapshots are available in history."
          : "Start a scoring run from Resumes once at least one candidate is ready."}>
        <ButtonLink to={resumes} primary>Go to Resumes <ArrowRight size={16} /></ButtonLink>
      </Empty></section>}
    {history && <HistoryDialog onClose={() => setHistory(false)} onRanking={(rankingId) => {
      setHistory(false); setSearch(rankingId === job.current_ranking_id ? {} : { ranking: rankingId });
    }} onShortlist={(shortlistId) => { setHistory(false); setSnapshot(shortlistId); }} />}
    {snapshot && <ShortlistSnapshot key={snapshot} id={snapshot} onClose={() => setSnapshot(null)} onRanking={(item) => {
      setSnapshot(null); setSearch({ ranking: item.ranking_id, approved: item.shortlist_id });
    }} />}
  </div>;
}

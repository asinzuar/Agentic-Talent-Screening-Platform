import { useState } from "react";
import { Link, useNavigate, useSearchParams } from "react-router";
import { ArrowRight, ArrowUpRight, FileCheck2, FileStack, FileUp, History, ListChecks, ShieldAlert, UsersRound, Zap } from "lucide-react";
import { jobApi, jobKey, jobUrl, useCollection, useCommand, useJob, useRead } from "./api";
import type { Batch, BatchDetail, Candidate, IntakeStatus, Job, ScorePlan } from "./models";
import { IntakeAudit } from "./IntakeAudit";
import {
  actionReason, Button, ButtonLink, Chip, dateTime, Dialog, Empty, ErrorNotice, FilePicker,
  humanize, Loading, LoadMore, Notice, plural, RESUME_FILES, StateChip,
} from "./ui";

const stageLabels: Record<string, string> = {
  queued: "Queued", document_safety: "Document safety", injection_detection: "Injection detection",
  extraction: "Extraction", evidence_verification: "Evidence verification",
};
const intakeFilters: { value: IntakeStatus | "all"; label: string }[] = [
  { value: "all", label: "All candidates" }, { value: "processing", label: "Processing" },
  { value: "ready_to_score", label: "Ready to score" }, { value: "review_required", label: "Review required" },
  { value: "quarantined", label: "Quarantined" },
];

function ResumeUpload({ job, onSubmitted, replacementRequestId }: {
  job: Job; onSubmitted: (batch: BatchDetail) => void; replacementRequestId?: string;
}) {
  const command = useCommand(job.job_id);
  const [files, setFiles] = useState<File[]>([]);
  const [version, setVersion] = useState(job.active_profile?.version);
  const versionChanged = version !== job.active_profile?.version;
  return <form className="stack" onSubmit={async (event) => {
    event.preventDefault();
    if (version == null || versionChanged || !files.length || command.pending) return;
    const body = new FormData();
    body.append("profile_version", String(version));
    if (replacementRequestId) body.append("replacement_request_id", replacementRequestId);
    for (const file of files) body.append("files", file);
    const result = await command.submit<BatchDetail>("POST", `${jobApi(job.job_id)}/resume-batches`, body);
    if (result) { setFiles([]); onSubmitted(result); }
  }}>
    <div className="confirmation-context">
      <span><strong>{job.role_title}</strong><small>Target job</small></span>
      <Chip tone="blue">Requirements v{version}</Chip>
    </div>
    {versionChanged && <Notice tone="amber">The active requirements changed. Confirm the new version before uploading.
      <Button variant="ghost" onClick={() => { setVersion(job.active_profile?.version); command.reset(); }}>Use active version</Button>
    </Notice>}
    <FilePicker files={files} onChange={setFiles} disabled={command.pending}
      rules={replacementRequestId ? { ...RESUME_FILES, maxFiles: 1 } : RESUME_FILES} />
    <small className="muted">{replacementRequestId ? "One file, up to 10 MiB." : "Up to 32 files, 10 MiB per file, 32 MiB total."} Upload starts intake only.</small>
    <ErrorNotice error={command.error} />
    <div className="dialog-actions"><Button type="submit" busy={command.pending} disabled={!files.length || versionChanged || !job.actions.upload_resumes?.allowed}>
      <FileUp size={17} /> {replacementRequestId ? "Upload replacement" : "Upload resumes"}
    </Button></div>
  </form>;
}

function CandidateReview({ candidateId, onClose, onBatch }: {
  candidateId: string; onClose: () => void; onBatch: (batch: BatchDetail) => void;
}) {
  const job = useJob();
  const path = `${jobApi(job.job_id)}/candidates/${encodeURIComponent(candidateId)}`;
  const candidate = useRead<Candidate>([...jobKey(job.job_id), "candidate", candidateId], path,
    (row) => row?.status === "processing" ? 2500 : false);
  const command = useCommand(job.job_id);
  const [reason, setReason] = useState("");
  const [feedback, setFeedback] = useState("");
  const row = candidate.data;
  const inputs = row ? { expected_revision: row.revision, attempt_id: row.attempt_id } : null;
  return <Dialog title={`Intake review · ${candidateId}`} description="Review the current intake attempt and its eligibility for scoring."
    onClose={onClose} wide busy={command.pending}>
    {candidate.isPending ? <Loading>Loading candidate…</Loading>
      : candidate.isError || !row ? <ErrorNotice error={candidate.error} retry={() => void candidate.refetch()} />
      : <div className="stack">
        <div className="chip-row"><StateChip value={row.status} />{row.overall_risk && <Chip tone={row.overall_risk === "high" ? "red" : "neutral"}>{humanize(row.overall_risk)} risk</Chip>}
          {!!row.security_flags.length && <Chip tone="amber">{plural(row.security_flags.length, "finding")}</Chip>}</div>
        {row.hold_code && <Notice tone="amber">{actionReason(row.hold_code)}</Notice>}
        {row.status === "processing" && <Loading>{stageLabels[row.stage ?? "queued"] ?? "Processing resume"}</Loading>}
        {row.status === "quarantined" && <Notice tone="amber">This file cannot proceed. Request a replacement to keep this applicant’s blind ID.</Notice>}
        {row.latest_disposition && <div className="note-block">
          <strong>{humanize(row.latest_disposition.action)}</strong><p>{row.latest_disposition.reason}</p>
          <small>{row.latest_disposition.actor_id} · {dateTime(row.latest_disposition.recorded_at)}</small>
        </div>}
        <IntakeAudit jobId={job.job_id} path={`${path}/intake-attempts/${encodeURIComponent(row.attempt_id)}/audit`} />
        {(row.actions.keep_on_hold?.allowed || row.actions.clear_for_scoring?.allowed) && <section className="audit-section stack">
          <h3>Record a review disposition</h3><p className="small muted">Clearance uses only verified evidence. Existing findings and exclusions stay on record.</p>
          <label className="field">Review reason
            <textarea rows={3} value={reason} required maxLength={4000} placeholder="Explain why you’re taking this action."
              onChange={(event) => setReason(event.target.value)} />
          </label>
          <div className="button-row">
            {(["keep_on_hold", "clear_for_scoring"] as const).filter((action) => row.actions[action]?.allowed).map((action) =>
              <Button key={action} variant={action === "clear_for_scoring" ? "primary" : "secondary"} busy={command.pending} disabled={!reason.trim()}
                onClick={async () => {
                  const result = await command.submit<Candidate>("POST", `${path}/review-dispositions`, { ...inputs, action, reason: reason.trim() });
                  if (result) { setReason(""); setFeedback(action === "clear_for_scoring" ? "Cleared for scoring." : "Hold reason recorded."); }
                }}>{humanize(action)}</Button>)}
          </div>
        </section>}
        <ErrorNotice error={command.error} retry={command.error ? () => { command.reset(); void candidate.refetch(); } : undefined} />
        {feedback && <Notice tone="green">{feedback}</Notice>}
        {(row.actions.retry?.allowed || row.actions.request_replacement?.allowed) && <div className="button-row">
          {row.actions.retry?.allowed && <Button variant="secondary" busy={command.pending}
            onClick={async () => { if (await command.submit("POST", `${path}/intake-retries`, inputs!)) setFeedback("Intake retry queued."); }}>
            Retry intake</Button>}
          {row.actions.request_replacement?.allowed && <Button variant="secondary" busy={command.pending}
            onClick={async () => { if (await command.submit("POST", `${path}/replacement-requests`, inputs!)) setFeedback("Replacement request recorded."); }}>
            Request replacement</Button>}
        </div>}
        {row.replacement_request_id && job.actions.upload_resumes?.allowed && <section className="audit-section stack">
          <h3>Upload a replacement</h3><p className="small muted">This upload starts a fresh attempt for {candidateId}.</p>
          <ResumeUpload job={job} replacementRequestId={row.replacement_request_id} onSubmitted={onBatch} />
        </section>}
      </div>}
  </Dialog>;
}

function BatchStatus({ batchId, onCandidate }: { batchId: string; onCandidate: (id: string) => void }) {
  const job = useJob();
  const batch = useRead<BatchDetail>([...jobKey(job.job_id), "batch", batchId],
    `${jobApi(job.job_id)}/resume-batches/${batchId}`,
    (data) => data?.files.some((file) => file.status === "processing") ? 2500 : false);
  if (batch.isPending) return <Loading>Loading batch…</Loading>;
  if (batch.isError) return <ErrorNotice error={batch.error} retry={() => void batch.refetch()} />;
  return <ul className="batch-rows">{batch.data.files.map((file) => <li key={file.attempt_id}>
    <div><small className="muted">File {file.upload_position}</small>
      <Button variant="ghost" className="text-button code-text" onClick={() => onCandidate(file.candidate_id)}>{file.candidate_id}</Button></div>
    <StateChip value={file.status} />
    <p>{file.stage ? stageLabels[file.stage] ?? humanize(file.stage) :
      file.issue_codes.length ? file.issue_codes.map(humanize).join(" · ") : "Intake complete"}</p>
  </li>)}</ul>;
}

function ConfirmScore({ plan, onClose, refresh }: { plan: ScorePlan; onClose: () => void; refresh: () => Promise<void> }) {
  const job = useJob();
  const navigate = useNavigate();
  const command = useCommand(job.job_id);
  const excluded = Object.entries(plan.excluded_counts).filter(([, count]) => count > 0);
  return <Dialog title="Confirm scoring run" description={`Use Requirements v${plan.profile_version} for this pool.`}
    onClose={onClose} busy={command.pending}>
    <div className="stack">
      <div className="confirmation-context"><span><strong>{plan.role_title}</strong><small>{plural(plan.pool_count, "uploaded resume")}</small></span>
        <Chip tone="blue">{plan.mode === "incremental" ? "Incremental run" : "Full run"}</Chip></div>
      <dl className="review-counts">
        <div><dt>New scores to compute</dt><dd>{plan.to_score_count}</dd></div>
        <div><dt>Previous scores to reuse</dt><dd>{plan.reuse_count}</dd></div>
        {excluded.map(([status, count]) => <div key={status}><dt>Excluded · {humanize(status)}</dt><dd>{count}</dd></div>)}
      </dl>
      <p className="small muted">The run captures ready candidates at confirmation time. The previous ranking remains available while it runs.</p>
      <ErrorNotice error={command.error} />
      <div className="dialog-actions">
        <Button variant="secondary" onClick={onClose} disabled={command.pending}>Cancel</Button>
        {command.error ? <Button onClick={async () => { await refresh(); command.reset(); }}>Refresh confirmation</Button>
          : <Button busy={command.pending} disabled={!plan.can_start || !plan.plan_token} onClick={async () => {
            if (await command.submit("POST", `${jobApi(job.job_id)}/scoring-runs`, { plan_token: plan.plan_token })) {
              navigate(`${jobUrl(job.job_id)}/rankings`);
            }
          }}><Zap size={16} /> Start scoring</Button>}
      </div>
      {!plan.can_start && <Notice tone="amber">{actionReason(plan.reason_code)}</Notice>}
    </div>
  </Dialog>;
}

export function Resumes() {
  const job = useJob();
  const path = jobApi(job.job_id);
  const [search, setSearch] = useSearchParams();
  const [filter, setFilter] = useState<IntakeStatus | "all">("all");
  const [batchId, setBatch] = useState<string | null>(null);
  const [confirmation, setConfirmation] = useState<ScorePlan | null>(null);
  const planQuery = useRead<ScorePlan>([...jobKey(job.job_id), "score-plan"], `${path}/score-plan`, () => 3500);
  const candidates = useCollection<Candidate>([...jobKey(job.job_id), "candidates", filter],
    `${path}/candidates${filter === "all" ? "" : `?status=${filter}`}`, true);
  const batches = useCollection<Batch>([...jobKey(job.job_id), "batches"], `${path}/resume-batches`, true);
  const plan = planQuery.data;
  const uploadOpen = search.get("upload") === "1";
  const candidateId = search.get("candidate");
  const uploadAllowed = job.actions.upload_resumes?.allowed ?? false;
  const scoreLabel = plan?.mode === "incremental" ? "Score new ready resumes" : "Score ready resumes";
  const metrics = [
    { label: "Uploaded", value: plan?.pool_count, icon: FileStack, tone: "blue" },
    { label: "Ready to score", value: plan?.ready_count, icon: FileCheck2, tone: "green" },
    { label: "Processing", value: plan ? plan.excluded_counts.processing ?? 0 : undefined, icon: History, tone: "blue" },
    { label: "Review required", value: plan ? plan.excluded_counts.review_required ?? 0 : undefined, icon: ShieldAlert, tone: "amber" },
  ];
  function selectCandidate(id: string) { setSearch({ candidate: id }); }
  function showBatch(batch: BatchDetail) { setSearch({}); setBatch(batch.batch_id); setFilter("all"); }
  return <div className="page-stack">
    <div className="toolbar tab-heading">
      <div><p className="eyebrow">APPLICANT POOL</p><h2>Resume intake</h2><p>Track verification and review which candidates are ready for scoring.</p></div>
      <div className="button-row">
        <Button variant="secondary" disabled={!uploadAllowed} onClick={() => setSearch({ upload: "1" })}>
          <FileUp size={17} /> {plan?.pool_count ? "Add resumes" : "Upload resumes"}
        </Button>
        <Button disabled={!plan?.can_start} onClick={() => plan && setConfirmation(plan)}><Zap size={17} /> {scoreLabel}</Button>
      </div>
    </div>
    {!uploadAllowed && <Notice tone="amber">{actionReason(job.actions.upload_resumes?.reason_code)}
      {!job.active_profile && <> <Link to={`${jobUrl(job.job_id)}/requirements`}>Review requirements <ArrowRight size={13} /></Link></>}</Notice>}
    {plan && !plan.can_start && job.active_profile && <p className="small muted action-hint">{actionReason(plan.reason_code)}</p>}
    <ErrorNotice error={planQuery.error} retry={planQuery.error ? () => void planQuery.refetch() : undefined} />
    <div className="metric-grid">
      {metrics.map(({ label, value, icon: Icon, tone }) => <div className="metric" key={label}>
        <span className={`section-icon section-${tone}`}><Icon size={18} /></span>
        <div><strong>{value ?? "—"}</strong><span>{label}</span></div>
      </div>)}
    </div>
    <section className="panel">
      <div className="panel-heading toolbar"><div><h2>Candidates</h2><p>Screening uses blind candidate IDs.</p></div>
        <select aria-label="Filter by intake status" value={filter} onChange={(event) => setFilter(event.target.value as typeof filter)}>
          {intakeFilters.map(({ value, label }) => <option key={value} value={value}>{label}</option>)}
        </select></div>
      {candidates.isPending ? <Loading>Loading candidates…</Loading> : candidates.isError
        ? <ErrorNotice error={candidates.error} retry={() => void candidates.refetch()} />
        : candidates.items.length === 0 ? <Empty icon={<UsersRound size={25} />} title={filter === "all" ? "No resumes yet" : "No matching candidates"}
          message={filter === "all" ? "Upload resumes to start document checks and evidence verification." : "Choose another intake status to see different candidates."}>
          {filter === "all" && uploadAllowed && <Button onClick={() => setSearch({ upload: "1" })}><FileUp size={16} /> Upload resumes</Button>}
        </Empty> : <div className="table-wrap"><table>
          <thead><tr><th>Candidate</th><th>Intake status</th><th>Risk</th><th>Findings</th><th className="numeric">Action</th></tr></thead>
          <tbody>{candidates.items.map((row) => <tr key={row.candidate_id}>
            <td><Button variant="ghost" className="text-button code-text" onClick={() => selectCandidate(row.candidate_id)}>{row.candidate_id}</Button></td>
            <td><StateChip value={row.status} />{row.stage && <small className="cell-note">{stageLabels[row.stage] ?? humanize(row.stage)}</small>}</td>
            <td>{row.overall_risk ? <StateChip value={row.overall_risk} /> : "—"}</td>
            <td>{row.security_flags.length ? <Chip tone="amber">{plural(row.security_flags.length, "finding")}</Chip> : <span className="muted">—</span>}</td>
            <td className="numeric"><Button variant="ghost" onClick={() => selectCandidate(row.candidate_id)}>
              {row.status === "review_required" ? "Review" : "Details"} <ArrowUpRight size={15} />
            </Button></td>
          </tr>)}</tbody></table></div>}
      <LoadMore query={candidates} />
    </section>
    {(batches.isError || batches.items.length > 0) && <section className="panel">
      <div className="panel-heading"><div className="heading-with-icon"><span className="section-icon"><ListChecks size={18} /></span>
        <div><h2>Upload batches</h2><p>Track each file from the original submission.</p></div></div></div>
      <div className="panel-body stack">
        <ErrorNotice error={batches.error} retry={batches.error ? () => void batches.refetch() : undefined} />
        {batches.items.map((batch) => <div className="batch-card" key={batch.batch_id}>
          <Button className="batch-heading" variant="ghost" onClick={() => setBatch(batchId === batch.batch_id ? null : batch.batch_id)}
            aria-expanded={batchId === batch.batch_id}>
            <FileStack size={18} /><span>{dateTime(batch.uploaded_at)}</span>
            <span className="muted">{plural(Object.values(batch.status_counts).reduce((a, b) => a + b, 0), "file")} · v{batch.confirmed_profile_version}</span>
            {!!batch.status_counts.processing && <Chip tone="blue">{batch.status_counts.processing} processing</Chip>}
          </Button>
          {batchId === batch.batch_id && <BatchStatus batchId={batch.batch_id} onCandidate={selectCandidate} />}
        </div>)}
      </div><LoadMore query={batches} />
    </section>}
    {job.current_ranking_id && <div className="end-actions"><ButtonLink to={`${jobUrl(job.job_id)}/rankings`}>View current ranking <ArrowRight size={16} /></ButtonLink></div>}
    {uploadOpen && uploadAllowed && <Dialog title="Upload resumes" description="Confirm the job and requirements version before adding files."
      onClose={() => setSearch({})}><ResumeUpload job={job} onSubmitted={showBatch} /></Dialog>}
    {candidateId && <CandidateReview key={candidateId} candidateId={candidateId} onClose={() => setSearch({})} onBatch={showBatch} />}
    {confirmation && <ConfirmScore plan={confirmation} onClose={() => setConfirmation(null)} refresh={async () => {
      const updated = await planQuery.refetch();
      if (updated.data) setConfirmation(updated.data);
    }} />}
  </div>;
}

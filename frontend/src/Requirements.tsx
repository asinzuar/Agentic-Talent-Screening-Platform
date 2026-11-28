import { useRef, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { BookOpen, ChevronDown, ClipboardCheck, FileText, LockKeyhole, Plus, RotateCcw, Save, Trash2 } from "lucide-react";
import { jobApi, jobKey, useCommand, useJob, useRead } from "./api";
import type { Draft, FlagReview, JobDescription, Option, Profile, Requirement, RequirementKind } from "./models";
import {
  Button, Chip, Dialog, Empty, ErrorNotice, FilePicker, JD_FILES, Loading, Notice, StateChip, dateTime, humanize, plural,
} from "./ui";

const sourceNames: Record<Requirement["provenance"], string> = {
  jd_qualification: "JD qualification", jd_responsibility: "Inferred from JD",
  model_suggestion: "Model suggestion", hr_added: "HR added",
};
const kinds: RequirementKind[] = ["skill", "experience", "education", "leadership", "achievement"];
const levels = ["Listed", "Weak or indirect", "Applied in a role or project", "Specific demonstrated outcome"];
type ReadyDraft = Extract<Draft, { status: "ready" }>;

function Source({ jobId, jdId }: { jobId: string; jdId: string }) {
  const source = useRead<JobDescription>([...jobKey(jobId), "jd", jdId], `${jobApi(jobId)}/job-descriptions/${jdId}`);
  return <aside className="panel document-panel">
    <div className="panel-heading"><div className="heading-with-icon"><span className="section-icon"><BookOpen size={17} /></span>
      <div><h2>Source job description</h2><p>Text used to draft and validate criteria</p></div></div></div>
    {source.isPending ? <Loading>Loading source…</Loading>
      : source.isError ? <ErrorNotice error={source.error} retry={() => void source.refetch()} />
      : <div className="document-text" tabIndex={0} aria-label="Job description text">{source.data.text || "No extracted text is available."}</div>}
  </aside>;
}

function OptionList({ options, group, combine, onChange }: {
  options: Option[]; group: "options" | "substitutes"; combine: Requirement["combine"]; onChange: (next: Option[]) => void;
}) {
  const regular = group === "options";
  const label = regular ? "Option" : "Substitute";
  return <div className="option-list">
    <div className="split-heading"><strong>{regular ? "Criteria options" : "Substitutes"}</strong>
      <Button variant="ghost" onClick={() => onChange([...options, { text: "", preferred: false }])}><Plus size={15} /> Add {label.toLowerCase()}</Button></div>
    {!options.length && <small className="muted">No substitutes specified.</small>}
    {options.map((item, index) => <div className="option-row" key={item.option_id ?? index}>
      <input aria-label={`${label} ${index + 1}`} required value={item.text} placeholder={`${label} ${index + 1}`}
        onChange={(event) => onChange(options.map((entry, i) => i === index ? { ...entry, text: event.target.value } : entry))} />
      {regular && combine === "any_of" && <label className="check-label">
        <input type="checkbox" checked={item.preferred}
          onChange={(event) => onChange(options.map((entry, i) => ({ ...entry, preferred: i === index && event.target.checked })))} /> Preferred
      </label>}
      <Button variant="ghost" className="icon-button" aria-label={`Remove ${label.toLowerCase()} ${index + 1}`}
        disabled={regular && options.length === 1}
        onClick={() => onChange(options.filter((_, i) => i !== index))}><Trash2 size={16} /></Button>
    </div>)}
  </div>;
}

function CriterionCard({ criterion, index, onChange, onRemove, reviews, onReview }: {
  criterion: Requirement; index: number; onChange?: (criterion: Requirement) => void; onRemove?: () => void;
  reviews: FlagReview[]; onReview?: (id: string, decision: FlagReview["disposition"] | "") => void;
}) {
  const editable = Boolean(onChange);
  const fromJD = criterion.provenance === "jd_qualification" || criterion.provenance === "jd_responsibility";
  const change = (partial: Partial<Requirement>) => onChange?.({ ...criterion, ...partial });
  return <details className="criterion-card" open={index === 0 || !criterion.requirement_id}>
    <summary>
      <span className="criterion-index">{String(index + 1).padStart(2, "0")}</span>
      <span className="criterion-title"><strong>{criterion.statement || "Untitled criterion"}</strong>
        <small>{sourceNames[criterion.provenance]}{criterion.flags.length ? ` · ${plural(criterion.flags.length, "flag")}` : ""}</small></span>
      <span className="criterion-chips"><Chip tone={criterion.priority === "M" ? "violet" : "neutral"}>
        {criterion.priority === "M" ? "Mandatory" : "Preferred"}</Chip><StateChip value={criterion.status} /></span>
      <ChevronDown size={17} className="chevron" />
    </summary>
    <div className="criterion-body stack">
      {editable ? <>
        <label className="field">Criterion statement<input required value={criterion.statement} onChange={(event) => change({ statement: event.target.value })} /></label>
        <div className="field-grid">
          <label className="field">Kind<select value={criterion.kind} onChange={(event) => {
            const kind = event.target.value as RequirementKind;
            change({ kind, min_years: kind === "experience" ? criterion.min_years : null });
          }}>{kinds.map((kind) => <option key={kind} value={kind}>{humanize(kind)}</option>)}</select></label>
          <label className="field">Priority<select value={criterion.priority} onChange={(event) => change({ priority: event.target.value as Requirement["priority"] })}>
            <option value="M">Mandatory</option><option value="P">Preferred</option></select></label>
          <label className="field">Weight<input required type="number" min="0" max="10" step="any"
            value={criterion.weight || ""} onChange={(event) => change({ weight: Number(event.target.value) })} /></label>
          <label className="field">Minimum evidence<select value={criterion.min_evidence ?? ""}
            onChange={(event) => change({ min_evidence: event.target.value ? Number(event.target.value) : null })}>
            <option value="">Any usable evidence</option>{levels.map((level, i) => <option key={level} value={i + 1}>E{i + 1} · {level}</option>)}
          </select></label>
          <label className="field">Match options<select value={criterion.combine} onChange={(event) => {
            const combine = event.target.value as Requirement["combine"];
            change({ combine, options: combine === "all_of" ? criterion.options.map((item) => ({ ...item, preferred: false })) : criterion.options });
          }}><option value="all_of">All options</option><option value="any_of">Any option</option></select></label>
          {criterion.kind === "experience" && <label className="field">Minimum years
            <input type="number" min="0" step="any" value={criterion.min_years ?? ""}
              onChange={(event) => change({ min_years: event.target.value ? Number(event.target.value) : null })} />
          </label>}
        </div>
        <OptionList options={criterion.options} group="options" combine={criterion.combine} onChange={(options) => change({ options })} />
        <OptionList options={criterion.substitutes} group="substitutes" combine={criterion.combine} onChange={(substitutes) => change({ substitutes })} />
        <label className="field">{fromJD ? "Verbatim source quote" : "Rationale"}
          <textarea required rows={2} value={(fromJD ? criterion.quote : criterion.rationale) ?? ""}
            onChange={(event) => change(fromJD ? { quote: event.target.value } : { rationale: event.target.value })} />
        </label>
      </> : <>
        <dl className="detail-grid">
          <div><dt>Kind</dt><dd>{humanize(criterion.kind)}</dd></div>
          <div><dt>Weight</dt><dd>{criterion.weight}</dd></div>
          <div><dt>Evidence minimum</dt><dd>{criterion.min_evidence ? `E${criterion.min_evidence}` : "Any usable evidence"}</dd></div>
          {criterion.min_years != null && <div><dt>Minimum years</dt><dd>{criterion.min_years}</dd></div>}
        </dl>
        <div><strong>{criterion.combine === "all_of" ? "All of" : "Any of"}</strong>
          <ul className="compact-list">{criterion.options.map((item, i) => <li key={item.option_id ?? i}>{item.text}{item.preferred && " (preferred)"}</li>)}</ul></div>
        {!!criterion.substitutes.length && <div><strong>Substitutes</strong><ul className="compact-list">
          {criterion.substitutes.map((item, i) => <li key={item.option_id ?? i}>{item.text}</li>)}</ul></div>}
        {(criterion.quote || criterion.rationale) && <blockquote>{criterion.quote || criterion.rationale}</blockquote>}
      </>}
      {criterion.onet_source && <div><strong>O*NET source</strong><p className="small muted">{criterion.onet_source}</p></div>}
      {!!criterion.flags.length && <div className="review-flags"><strong>Reviewer flags</strong>
        {criterion.flags.map((flag) => {
          const disposition = reviews.find((review) => review.flag_id === flag.flag_id)?.disposition;
          return <div key={flag.flag_id} className="flag-row"><span>{flag.message}</span>
            {onReview ? <select aria-label={`Disposition for ${flag.message}`} value={disposition ?? ""}
              onChange={(event) => onReview(flag.flag_id, event.target.value as FlagReview["disposition"] | "")}>
              <option value="">Needs review</option><option value="acknowledged">Acknowledged</option><option value="addressed">Addressed</option>
            </select> : <Chip tone={disposition ? "green" : "amber"}>{disposition ? humanize(disposition) : "Needs review"}</Chip>}
          </div>;
        })}
      </div>}
      {editable && <div className="criterion-actions">
        <label className="check-label"><input type="checkbox" checked={criterion.status === "active"}
          onChange={(event) => change({ status: event.target.checked ? "active" : "inactive" })} /> Use for scoring</label>
        <Button variant="ghost" className="remove-action" onClick={onRemove}><Trash2 size={15} /> Remove criterion</Button>
      </div>}
    </div>
  </details>;
}

function NonScoring({ excerpts, onChange }: { excerpts: string[]; onChange?: (excerpts: string[]) => void }) {
  return <div className="panel non-scoring">
    <div className="split-heading"><div><h2>Not used for scoring</h2><p className="muted small">Source passages excluded from the requirements.</p></div>
      {onChange && <Button variant="ghost" onClick={() => onChange([...excerpts, ""])}><Plus size={16} /> Add passage</Button>}</div>
    {!excerpts.length && <p className="small muted">No passages listed.</p>}
    {excerpts.map((excerpt, i) => onChange
      ? <div className="option-row" key={i}><textarea required rows={2} aria-label={`Non-scoring passage ${i + 1}`} value={excerpt}
        onChange={(event) => onChange(excerpts.map((item, index) => index === i ? event.target.value : item))} />
        <Button variant="ghost" className="icon-button" aria-label={`Remove passage ${i + 1}`}
          onClick={() => onChange(excerpts.filter((_, index) => index !== i))}><Trash2 size={16} /></Button></div>
      : <blockquote key={i}>{excerpt}</blockquote>)}
  </div>;
}

function newCriterion(): Requirement {
  return {
    statement: "", kind: "skill", options: [{ text: "", preferred: false }], combine: "all_of", substitutes: [],
    min_years: null, min_evidence: null, priority: "P", weight: 5, provenance: "hr_added", quote: null,
    rationale: "", onet_source: null, status: "inactive", flags: [],
  };
}

function DraftEditor({ draft, reload }: { draft: ReadyDraft; reload: () => Promise<Draft | undefined> }) {
  const job = useJob();
  const cache = useQueryClient();
  const path = `${jobApi(job.job_id)}/requirement-draft`;
  const [base, setBase] = useState(draft);
  const [title, setTitle] = useState(draft.role_title);
  const [criteria, setCriteria] = useState(() => draft.requirements.map((item, index) => ({ key: index, item })));
  const [nextKey, setNextKey] = useState(criteria.length);
  const [excerpts, setExcerpts] = useState(draft.not_used_for_scoring);
  const [reviews, setReviews] = useState(draft.flag_reviews);
  const [confirm, setConfirm] = useState(false);
  const [saved, setSaved] = useState(false);
  const [validation, setValidation] = useState("");
  const command = useCommand(job.job_id);
  const form = useRef<HTMLFormElement>(null);
  const requirements = criteria.map(({ item }) => item);
  const activeCount = requirements.filter((item) => item.status === "active").length;
  const activeLabel = `${activeCount} active ${activeCount === 1 ? "criterion" : "criteria"}`;
  const flagIds = new Set(requirements.flatMap((item) => item.flags.map((flag) => flag.flag_id)));
  const remaining = [...flagIds].filter((flagId) => !reviews.some((review) => review.flag_id === flagId)).length;
  const dirty = title !== base.role_title || JSON.stringify(requirements) !== JSON.stringify(base.requirements)
    || JSON.stringify(excerpts) !== JSON.stringify(base.not_used_for_scoring) || JSON.stringify(reviews) !== JSON.stringify(base.flag_reviews);
  const stale = draft.revision !== base.revision;

  function useSnapshot(next: ReadyDraft) {
    setBase(next);
    setTitle(next.role_title);
    setCriteria(next.requirements.map((item, index) => ({ key: index, item })));
    setNextKey(next.requirements.length);
    setExcerpts(next.not_used_for_scoring);
    setReviews(next.flag_reviews);
    setValidation("");
  }

  async function save() {
    setSaved(false);
    setValidation("");
    if (!form.current?.reportValidity()) {
      setValidation("Complete the required fields before saving.");
      return undefined;
    }
    if (requirements.some((item) => item.weight <= 0 || (item.min_years !== null && item.min_years <= 0))) {
      setValidation("Weights and minimum years must be greater than zero.");
      return undefined;
    }
    const before = new Set(base.requirements.filter((item) => item.status === "inactive").map((item) => item.requirement_id));
    const body = {
      expected_revision: base.revision,
      role_title: title.trim(), requirements, not_used_for_scoring: excerpts,
      flag_reviews: reviews.filter((review) => flagIds.has(review.flag_id)),
      activate_requirement_ids: requirements.flatMap((item) =>
        item.requirement_id && item.status === "active" && before.has(item.requirement_id) ? [item.requirement_id] : []),
    };
    const result = await command.submit<ReadyDraft>("PUT", path, body, false);
    if (result) {
      useSnapshot(result);
      cache.setQueryData([...jobKey(job.job_id), "draft"], result);
      void cache.invalidateQueries({ queryKey: jobKey(job.job_id), exact: true });
      setSaved(true);
    }
    return result;
  }

  const nextVersion = (job.active_profile?.version ?? 0) + 1;
  return <div className="workspace-columns">
    <div className="stack">
      <section className="panel">
        <div className="panel-heading"><div className="heading-with-icon"><span className="section-icon"><ClipboardCheck size={18} /></span>
          <div><h2>{draft.base_version ? `Revision of Requirements v${draft.base_version}` : "Review draft requirements"}</h2>
            <p>{activeLabel} · {remaining ? `${plural(remaining, "flag")} to review` : "No outstanding flags"}</p></div>
        </div></div>
        <form ref={form} onSubmit={(event) => { event.preventDefault(); if (dirty && !stale) void save(); }}>
          <fieldset disabled={command.pending} className="draft-fields stack">
            <label className="field">Role title<input required value={title} onChange={(event) => { setTitle(event.target.value); setSaved(false); }} /></label>
            {criteria.map(({ item, key }, index) => <CriterionCard key={key} criterion={item} index={index} reviews={reviews}
              onChange={(criterion) => { setCriteria(criteria.map((entry) => entry.key === key ? { key, item: criterion } : entry)); setSaved(false); }}
              onRemove={() => { setCriteria(criteria.filter((entry) => entry.key !== key)); setSaved(false); }}
              onReview={(flagId, disposition) => {
                setReviews([...reviews.filter((item) => item.flag_id !== flagId), ...(disposition ? [{ flag_id: flagId, disposition }] : [])]);
                setSaved(false);
              }} />)}
            <Button className="add-criterion" variant="secondary" onClick={() => {
              setCriteria([...criteria, { key: nextKey, item: newCriterion() }]); setNextKey(nextKey + 1); setSaved(false);
            }}><Plus size={17} /> Add criterion</Button>
            <NonScoring excerpts={excerpts} onChange={(value) => { setExcerpts(value); setSaved(false); }} />
          </fieldset>
        </form>
        <div className="panel-actions stack">
          {stale && <Notice tone="amber">This draft changed in another session. Reload it before saving.</Notice>}
          <ErrorNotice error={command.error || validation} />
          {saved && !dirty && <Notice tone="green">Draft saved.</Notice>}
          <div className="toolbar">
            <div>{dirty && <span className="muted small">Unsaved changes</span>}</div>
            <div className="button-row">
              {(stale || command.error) && <Button variant="ghost" onClick={async () => {
                if (!dirty || window.confirm("Discard unsaved draft edits and reload?")) {
                  const refreshed = await reload();
                  if (refreshed?.status === "ready") { useSnapshot(refreshed); command.reset(); setSaved(false); }
                }
              }}><RotateCcw size={15} /> Reload draft</Button>}
              <Button variant="secondary" busy={command.pending && !confirm} disabled={!dirty || stale} onClick={() => void save()}><Save size={16} /> Save draft</Button>
              <Button disabled={command.pending || stale || !activeCount || !!remaining || job.actions.approve_requirements?.allowed === false}
                onClick={() => setConfirm(true)}><LockKeyhole size={16} /> Approve &amp; lock</Button>
            </div>
          </div>
          {(!activeCount || remaining > 0) && <small className="muted">Approval needs at least one active criterion and a disposition for every flag.</small>}
        </div>
      </section>
    </div>
    <Source jobId={job.job_id} jdId={draft.jd_id} />
    {confirm && <Dialog title={`Approve Requirements v${nextVersion}?`}
      description={`Lock ${activeLabel} for this job.`}
      onClose={() => { setConfirm(false); command.reset(); }} busy={command.pending}>
      <div className="stack">
        <p>{job.active_profile
          ? "Future scoring will require a full rescore of the eligible pool. Earlier rankings and shortlist decisions stay available in history."
          : "Approval unlocks resume uploads. The approved requirements are read-only; later changes need a revision."}</p>
        {dirty && <Notice>Current edits will be saved before approval.</Notice>}
        {stale && <Notice tone="amber">The draft has changed. Close this dialog and reload it before approving.</Notice>}
        <ErrorNotice error={command.error || validation} />
        <div className="dialog-actions">
          <Button variant="secondary" disabled={command.pending} onClick={() => setConfirm(false)}>Cancel</Button>
          <Button busy={command.pending} disabled={stale} onClick={async () => {
            const ready = dirty ? await save() : base;
            if (!ready) return;
            const approved = await command.submit<Profile>("POST", `${path}/approve`, { expected_revision: ready.revision });
            if (approved) setConfirm(false);
          }}><LockKeyhole size={16} /> Approve Requirements v{nextVersion}</Button>
        </div>
      </div>
    </Dialog>}
  </div>;
}

function Locked({ version, revisionOpen }: { version: number; revisionOpen: boolean }) {
  const job = useJob();
  const path = jobApi(job.job_id);
  const profile = useRead<Profile>([...jobKey(job.job_id), "profile", version], `${path}/requirement-profiles/${version}`);
  const command = useCommand(job.job_id);
  if (profile.isPending) return <Loading>Loading approved requirements…</Loading>;
  if (profile.isError) return <ErrorNotice error={profile.error} retry={() => void profile.refetch()} />;
  const data = profile.data;
  return <div className="workspace-columns">
    <div className="stack">
      <section className="panel">
        <div className="panel-heading toolbar"><div className="heading-with-icon">
          <span className="section-icon"><LockKeyhole size={17} /></span><div><h2>Requirements v{version}</h2>
            <p>Approved by {data.approved_by} · {dateTime(data.approved_at)}</p></div></div>
          {!revisionOpen && <Button variant="secondary" busy={command.pending} disabled={!job.actions.create_revision?.allowed}
            onClick={() => void command.submit("POST", `${path}/requirement-draft`, { base_version: version })}>
            <Plus size={16} /> Create revision</Button>}
        </div>
        <div className="panel-body stack">
          <ErrorNotice error={command.error} />
          <div className="locked-strip"><Chip tone="green"><LockKeyhole size={13} /> Approved &amp; locked</Chip>
            <span>{data.requirements.filter((item) => item.status === "active").length} active of {data.requirements.length} criteria</span></div>
          {data.requirements.map((criterion, index) =>
            <CriterionCard key={criterion.requirement_id ?? index} criterion={criterion} index={index} reviews={data.flag_reviews} />)}
        </div>
      </section>
      <NonScoring excerpts={data.not_used_for_scoring} />
    </div>
    <Source jobId={job.job_id} jdId={data.jd_id} />
  </div>;
}

function DraftFailure({ draft }: { draft: Extract<Draft, { status: "failed" }> }) {
  const job = useJob();
  const command = useCommand(job.job_id);
  const [replace, setReplace] = useState(false);
  const [files, setFiles] = useState<File[]>([]);
  async function retry(file?: File) {
    const body = new FormData();
    body.append("expected_revision", draft.revision);
    if (file) body.append("file", file);
    await command.submit("POST", `${jobApi(job.job_id)}/requirement-draft/retry`, body);
  }
  return <section className="panel panel-body stack">
    <h2>The draft could not be generated</h2><ErrorNotice error={draft.error} /><ErrorNotice error={command.error} />
    <div className="button-row">
      {draft.actions.retry?.allowed && <Button variant="secondary" busy={command.pending} onClick={() => void retry()}>
        <RotateCcw size={16} /> Retry drafting</Button>}
      {draft.actions.replace_jd?.allowed && <Button onClick={() => { setReplace(true); command.reset(); }}>Replace job description</Button>}
    </div>
    {replace && <Dialog title="Replace job description" description="Submit a readable file to try drafting again."
      onClose={() => setReplace(false)} busy={command.pending}>
      <div className="stack"><FilePicker files={files} onChange={setFiles} rules={JD_FILES} disabled={command.pending} />
        <ErrorNotice error={command.error} /><div className="dialog-actions"><Button variant="secondary" onClick={() => setReplace(false)}>Cancel</Button>
          <Button disabled={!files[0]} busy={command.pending} onClick={() => void retry(files[0])}>Upload and retry</Button></div></div>
    </Dialog>}
  </section>;
}

export function Requirements() {
  const job = useJob();
  const [showLocked, setShowLocked] = useState(false);
  const query = useRead<Draft>([...jobKey(job.job_id), "draft"], `${jobApi(job.job_id)}/requirement-draft`,
    (draft) => !draft || draft.status === "queued" || draft.status === "running" || draft.status !== job.draft.status ? 2000 : false);
  const draft = query.data;
  if (query.isPending) return <Loading>Loading requirements…</Loading>;
  if (query.isError || !draft) return <ErrorNotice error={query.error} retry={() => void query.refetch()} />;
  const version = job.active_profile?.version;
  const revisionOpen = draft.status !== "none";
  return <div className="stack">
    {version && revisionOpen && <div className="toolbar revision-banner">
      <div><strong>Requirements v{version} remains active</strong><p>The revision takes effect after it is approved.</p></div>
      <Button variant="secondary" onClick={() => setShowLocked(!showLocked)}>
        {showLocked ? "Back to draft revision" : `View active v${version}`}
      </Button>
    </div>}
    {version && (draft.status === "none" || showLocked) ? <Locked version={version} revisionOpen={revisionOpen} />
      : draft.status === "ready" ? <DraftEditor key={`${draft.jd_id}:${draft.base_version}`} draft={draft}
        reload={async () => (await query.refetch()).data} />
      : draft.status === "failed" ? <DraftFailure draft={draft} />
      : draft.status === "queued" || draft.status === "running" ? <section className="panel">
        <Empty icon={<FileText size={26} />} title="Preparing your requirements"
          message="You’ll review the draft and its source passages before resumes can be uploaded.">
          <Loading>{draft.stage === "extract_jd" ? "Reading job description" : draft.stage === "draft_requirements" ? "Drafting requirements" : "Queued for drafting"}</Loading>
        </Empty>
      </section> : <Loading>Refreshing approved requirements…</Loading>}
  </div>;
}

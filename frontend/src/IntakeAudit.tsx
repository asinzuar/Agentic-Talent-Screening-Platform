import { useState } from "react";
import { Eye, FileCheck2, ShieldAlert } from "lucide-react";
import { ApiError, jobKey, useRead, useSession } from "./api";
import type { Excerpt, IntakeAudit as Packet, SourceSpan } from "./models";
import { Button, Chip, ErrorNotice, Loading, StateChip, dateTime, humanize, plural } from "./ui";

function ControlledExcerpt({ span }: { span: SourceSpan }) {
  const { api } = useSession();
  const [excerpt, setExcerpt] = useState<Excerpt | null>(null);
  const [error, setError] = useState<unknown>(null);
  const [busy, setBusy] = useState(false);
  return <div className="excluded-span stack">
    <div className="toolbar"><span className="small muted">{span.page ? `Page ${span.page} · ` : ""}{span.span_id} · {humanize(span.visibility)}</span>
      {span.excerpt_href && <Button variant="ghost" busy={busy} onClick={async () => {
        if (excerpt) { setExcerpt(null); return; }
        setBusy(true); setError(null);
        try { setExcerpt(await api.read<Excerpt>(span.excerpt_href!)); } catch (failure) { setError(failure); } finally { setBusy(false); }
      }}><Eye size={16} /> {excerpt ? "Hide excerpt" : "Show excerpt"}</Button>}</div>
    <ErrorNotice error={error} />
    {excerpt && <blockquote className="excluded-quote">{excerpt.excerpt}</blockquote>}
  </div>;
}

export function IntakeAudit({ jobId, path }: { jobId: string; path: string }) {
  const audit = useRead<Packet>([...jobKey(jobId), "intake-audit", path], path,
    (data) => data?.status === "processing" ? 3500 : false);
  if (audit.isPending) return <Loading>Loading the intake record…</Loading>;
  if (audit.isError) {
    return <ErrorNotice error={audit.error instanceof ApiError && audit.error.status === 403
      ? "Your account cannot access the intake audit packet for this job." : audit.error}
      retry={audit.error instanceof ApiError && audit.error.status === 403 ? undefined : () => void audit.refetch()} />;
  }
  const packet = audit.data;
  const facts = packet.verified_profile?.facts ?? [];
  const reportedSpans = new Set(packet.findings.flatMap((finding) => finding.spans.map((span) => span.span_id)));
  const otherExclusions = packet.exclusions.filter((span) => !reportedSpans.has(span.span_id));
  return <div className="stack">
    <section className="audit-section stack">
      <div className="heading-with-icon"><span className="section-icon section-green"><FileCheck2 size={18} /></span>
        <div><h3>Verified evidence</h3><p className="small muted">Visible, span-verified facts retained by intake.</p></div></div>
      <ErrorNotice error={packet.error} />
      {facts.length ? <ul className="evidence-list">{facts.map((fact) => <li key={fact.fact_id}>
        <div className="toolbar"><strong>{fact.statement}</strong><Chip tone="blue">E{fact.evidence_level} · {humanize(fact.kind)}</Chip></div>
        {fact.spans.map((span) => <div key={span.location}><blockquote>{span.quote}</blockquote><small className="muted">{span.location}</small></div>)}
        {!!fact.dates.length && <p className="small muted">Dates shown on resume: {fact.dates.join(", ")}</p>}
      </li>)}</ul> : <p className="muted">{packet.status === "processing" ? "Evidence verification is in progress." : "No usable verified profile is available."}</p>}
    </section>
    <section className="audit-section excluded-section stack">
      <div className="heading-with-icon"><span className="section-icon section-amber"><ShieldAlert size={18} /></span>
        <div><h3>Findings and excluded content</h3><p className="small muted">Excluded content is unavailable for scoring. Excerpt access is logged.</p></div></div>
      {!packet.findings.length && !packet.exclusions.length && <p className="muted">No findings recorded for this attempt.</p>}
      {packet.findings.map((finding) => <div className="finding-card stack" key={finding.finding_id}>
        <div className="toolbar"><strong>{humanize(finding.category)}</strong><div className="chip-row">
          <Chip>{finding.layer}{finding.attack_family ? ` · ${finding.attack_family}` : ""}</Chip><StateChip value={finding.risk} />
        </div></div>
        <p className="small muted">{humanize(finding.action_code)}</p>
        {finding.spans.map((span) => <ControlledExcerpt key={span.span_id} span={span} />)}
      </div>)}
      {!!otherExclusions.length && <details>
        <summary>{plural(otherExclusions.length, "other excluded span")}</summary>
        {otherExclusions.map((span) => <ControlledExcerpt key={span.span_id} span={span} />)}
      </details>}
    </section>
    <details className="detail-disclosure">
      <summary>Intake provenance and review history</summary>
      <dl className="detail-grid">
        <div><dt>Intake outcome</dt><dd>{packet.outcome ? humanize(packet.outcome) : "In progress"}</dd></div>
        <div><dt>Stopped at</dt><dd>{packet.provenance.stopped_at ? humanize(packet.provenance.stopped_at) : "—"}</dd></div>
        <div><dt>Pipeline schema</dt><dd>{packet.provenance.pipeline_schema || "—"}</dd></div>
        <div><dt>Intake model</dt><dd>{packet.provenance.model || "—"}</dd></div>
        <div><dt>Evidence verifier</dt><dd>{packet.provenance.l3_model || "—"}</dd></div>
        <div><dt>Document SHA-256</dt><dd className="code-text break-text">{packet.document_sha256}</dd></div>
      </dl>
      {!!packet.dispositions.length && <ul className="timeline">{packet.dispositions.map((item) => <li key={item.disposition_id}>
        <strong>{humanize(item.action)}</strong><p>{item.reason}</p><small>{item.actor_id} · {dateTime(item.recorded_at)}</small>
      </li>)}</ul>}
    </details>
  </div>;
}

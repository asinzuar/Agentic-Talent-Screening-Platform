import { useEffect, useId, useRef, useState } from "react";
import type { ButtonHTMLAttributes, ReactNode } from "react";
import { Link } from "react-router";
import { AlertCircle, Check, FileUp, LoaderCircle, X } from "lucide-react";
import { ApiError } from "./api";
import type { Failure } from "./models";

export function humanize(value: string) {
  return value.toLowerCase().replaceAll("_", " ").replace(/^\w/, (letter) => letter.toUpperCase());
}

export function dateTime(value: string | null | undefined) {
  return value ? new Intl.DateTimeFormat(undefined, { dateStyle: "medium", timeStyle: "short" }).format(new Date(value)) : "—";
}

export const plural = (value: number, label: string) => `${value} ${label}${value === 1 ? "" : "s"}`;

export function actionReason(code: string | null | undefined) {
  const reasons: Record<string, string> = {
    NO_READY_CANDIDATES: "No resumes are ready yet. Complete intake or review held profiles.",
    NO_NEW_WORK: "All ready candidates already have reusable scores.",
    REQUIREMENTS_NOT_APPROVED: "Approve a requirements profile to unlock this action.",
    RUN_IN_PROGRESS: "A scoring run is already in progress.",
    PERMISSION_REQUIRED: "Your account does not have permission for this action.",
    CLEARANCE_REQUIRED: "A verified profile is available and needs your clearance.",
    NO_USABLE_VERIFIED_PROFILE: "Intake did not produce a usable verified profile.",
    INPUT_CHANGED: "The candidate input changed during scoring.",
    SCORING_FAILED: "Matching or scoring could not finish for this candidate.",
  };
  return code ? reasons[code.toUpperCase()] ?? humanize(code) : "";
}

export function Chip({ children, tone = "neutral" }: {
  children: ReactNode; tone?: "neutral" | "blue" | "green" | "amber" | "red" | "violet";
}) {
  return <span className={`chip chip-${tone}`}>{children}</span>;
}

export function StateChip({ value }: { value: string }) {
  const tone = ["ready", "ready_to_score", "approved", "completed", "met", "active", "low"].includes(value) ? "green"
    : ["failed", "quarantined", "high", "not_met"].includes(value) ? "red"
    : ["review_required", "medium", "completed_with_errors", "not_assessed", "rescore_required"].includes(value) ? "amber"
    : ["processing", "queued", "running", "draft"].includes(value) ? "blue" : "neutral";
  const labels: Record<string, string> = {
    ready_to_score: "Ready to score", rescore_required: "Rescore required",
    completed_with_errors: "Completed with errors", not_assessed: "Not assessed", not_met: "Not met",
  };
  return <Chip tone={tone}><span className="status-dot" />{labels[value] ?? humanize(value)}</Chip>;
}

export function Button({ variant = "primary", busy, className = "", children, disabled, ...props }:
  ButtonHTMLAttributes<HTMLButtonElement> & { variant?: "primary" | "secondary" | "ghost" | "danger"; busy?: boolean }) {
  return <button type="button" {...props} className={`button button-${variant} ${className}`}
    disabled={disabled || busy} aria-busy={busy}>
    {busy && <LoaderCircle size={16} className="spin" />}{children}
  </button>;
}

export function ButtonLink({ to, children, primary = false }: { to: string; children: ReactNode; primary?: boolean }) {
  return <Link className={`button button-${primary ? "primary" : "secondary"}`} to={to}>{children}</Link>;
}

export function ErrorNotice({ error, retry }: { error: unknown; retry?: () => void }) {
  if (!error) return null;
  const fields = error instanceof ApiError ? error.failure.field_errors : undefined;
  const message = typeof error === "string" ? error : (error as Error | Failure).message ?? "Something went wrong.";
  return <div className="notice notice-error" role="alert">
    <AlertCircle size={18} aria-hidden="true" />
    <div className="notice-content">
      {message}
      {fields?.length ? <ul>{fields.map((field, i) => <li key={`${field.field}-${i}`}>{field.field || "Request"}: {field.message}</li>)}</ul> : null}
    </div>
    {retry && <Button variant="secondary" onClick={retry}>Retry</Button>}
  </div>;
}

export function Notice({ children, tone = "blue" }: { children: ReactNode; tone?: "blue" | "green" | "amber" }) {
  return <div className={`notice notice-${tone}`} role="status">
    {tone === "green" ? <Check size={18} /> : <AlertCircle size={18} />}
    <div className="notice-content">{children}</div>
  </div>;
}

export function Loading({ children = "Loading…" }: { children?: ReactNode }) {
  return <div className="loading" role="status"><LoaderCircle size={19} className="spin" />{children}</div>;
}

export function Empty({ icon, title, message, children }: {
  icon: ReactNode; title: string; message: string; children?: ReactNode;
}) {
  return <div className="empty-state">
    <div className="empty-icon">{icon}</div><h2>{title}</h2><p>{message}</p>{children}
  </div>;
}

export function Dialog({ title, description, children, onClose, wide = false, busy = false }: {
  title: string; description?: string; children: ReactNode; onClose: () => void; wide?: boolean; busy?: boolean;
}) {
  const dialog = useRef<HTMLDialogElement>(null);
  const titleId = useId();
  const descriptionId = useId();
  useEffect(() => {
    const element = dialog.current;
    if (element && !element.open) element.showModal();
    return () => { if (element?.open) element.close(); };
  }, []);
  return <dialog ref={dialog} className={`dialog ${wide ? "dialog-wide" : ""}`} aria-labelledby={titleId}
    aria-describedby={description ? descriptionId : undefined}
    onCancel={(event) => { event.preventDefault(); if (!busy) onClose(); }}>
    <div className="dialog-header">
      <div><h2 id={titleId}>{title}</h2>{description && <p id={descriptionId}>{description}</p>}</div>
      <Button variant="ghost" className="icon-button" aria-label="Close dialog" disabled={busy} onClick={onClose}><X size={20} /></Button>
    </div>
    <div className="dialog-body">{children}</div>
  </dialog>;
}

export interface FileRules {
  extensions: string[];
  maxFiles: number;
}
export const JD_FILES: FileRules = { extensions: [".pdf", ".txt"], maxFiles: 1 };
export const RESUME_FILES: FileRules = { extensions: [".pdf", ".docx", ".txt"], maxFiles: 32 };

export function FilePicker({ files, onChange, rules, disabled = false }: {
  files: File[]; onChange: (files: File[]) => void; rules: FileRules; disabled?: boolean;
}) {
  const id = useId();
  const [error, setError] = useState("");
  const [dragging, setDragging] = useState(false);
  function choose(next: File[]) {
    const accepted = next.every((file) => rules.extensions.some((suffix) => file.name.toLowerCase().endsWith(suffix)));
    const failure = !accepted ? `Choose ${rules.extensions.map((suffix) => suffix.slice(1).toUpperCase()).join(", ")} files.`
      : next.length > rules.maxFiles ? `Choose at most ${plural(rules.maxFiles, "file")}.`
      : next.some((file) => file.size > 10 * 1024 ** 2) ? "Each file must be 10 MiB or smaller."
      : next.reduce((size, file) => size + file.size, 0) > 32 * 1024 ** 2 ? "The batch must be 32 MiB or smaller." : "";
    setError(failure);
    onChange(failure ? [] : next);
  }
  return <div className="file-picker">
    <label className={`dropzone ${dragging ? "dragging" : ""}`} htmlFor={id}
      onDragOver={(event) => { event.preventDefault(); if (!disabled) setDragging(true); }}
      onDragLeave={() => setDragging(false)}
      onDrop={(event) => { event.preventDefault(); setDragging(false); if (!disabled) choose(Array.from(event.dataTransfer.files)); }}>
      <input id={id} className="visually-hidden" type="file" accept={rules.extensions.join(",")} disabled={disabled}
        multiple={rules.maxFiles > 1} onChange={(event) => { choose(Array.from(event.target.files ?? [])); event.target.value = ""; }} />
      <FileUp size={23} /><strong>{rules.maxFiles === 1 ? "Choose a file" : "Choose files"}</strong>
      <span>or drag and drop · {rules.extensions.map((suffix) => suffix.slice(1).toUpperCase()).join(", ")}</span>
    </label>
    <ErrorNotice error={error} />
    {files.length > 0 && <ul className="chosen-files">
      {files.map((file, i) => <li key={`${file.name}-${i}`}>
        <span>{file.name}</span><small>{(file.size / 1024).toFixed(1)} KiB</small>
        <Button variant="ghost" className="icon-button" aria-label={`Remove ${file.name}`} disabled={disabled}
          onClick={() => onChange(files.filter((_, index) => index !== i))}><X size={16} /></Button>
      </li>)}
    </ul>}
  </div>;
}

export function LoadMore({ query }: {
  query: { hasNextPage: boolean; isFetchingNextPage: boolean; fetchNextPage: () => unknown };
}) {
  return query.hasNextPage
    ? <div className="table-footer"><Button variant="secondary" busy={query.isFetchingNextPage} onClick={() => void query.fetchNextPage()}>Load more</Button></div>
    : null;
}

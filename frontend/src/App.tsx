import { useEffect, useRef, useState } from "react";
import { Link, NavLink, Navigate, Outlet, Route, Routes, useLocation, useNavigate, useParams } from "react-router";
import { useQueryClient } from "@tanstack/react-query";
import {
  ArrowDownRight, ArrowLeft, ArrowRight, ArrowUpRight, BriefcaseBusiness, Check, ChevronRight,
  ClipboardCheck, FileCheck2, FileText, Fingerprint, ListOrdered, LockKeyhole, LogOut, Plus,
  ScanSearch, Search, ShieldCheck, UsersRound,
} from "lucide-react";
import { JOBS, jobApi, jobKey, jobUrl, useCollection, useCommand, useRead, useSession } from "./api";
import type { Job, Run } from "./models";
import {
  Button, ButtonLink, Chip, Dialog, Empty, ErrorNotice, FilePicker, JD_FILES, Loading,
  LoadMore, Notice, StateChip, plural,
} from "./ui";
import { Requirements } from "./Requirements";
import { Resumes } from "./Resumes";
import { Rankings } from "./Rankings";

function Brand() {
  return <div className="brand"><img src="/mark.svg" alt="" width={36} height={36} />
    <div><strong>Talent Screener</strong><span>Hiring workspace</span></div>
  </div>;
}

function Connect() {
  const { connect, expired } = useSession();
  const [token, setToken] = useState("");
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<unknown>(null);
  return <main className="connect-layout">
    <section className="connect-feature">
      <Brand />
      <div className="connect-intro">
        <div className="connect-eyebrow"><span /> EVIDENCE-BASED SCREENING</div>
        <h1>A clearer path<br />to the shortlist.</h1>
        <p>Set the criteria, verify the evidence, and review every ranking before you decide who moves forward.</p>
      </div>
      <div className="connect-stages">
        <span><ClipboardCheck size={17} /> Approve criteria</span>
        <span><ShieldCheck size={17} /> Verify evidence</span>
        <span><ListOrdered size={17} /> Review ranks</span>
      </div>
    </section>
    <section className="connect-panel">
      <form className="connect-form stack" onSubmit={async (event) => {
        event.preventDefault();
        if (pending || !token.trim()) return;
        setPending(true);
        setError(null);
        try { await connect(token); } catch (failure) { setError(failure); } finally { setPending(false); }
      }}>
        <div className="section-icon"><LockKeyhole size={22} /></div>
        <div><p className="eyebrow">REVIEWER ACCESS</p><h2>Connect to your workspace</h2>
          <p className="muted">Enter the access token configured for your account on the job API.</p></div>
        {expired && <Notice tone="amber">Your workspace session expired. Connect again to continue.</Notice>}
        <label className="field">Workspace access token
          <input type="password" value={token} required autoComplete="off" placeholder="Enter access token"
            onChange={(event) => { setToken(event.target.value); setError(null); }} />
        </label>
        <ErrorNotice error={error} />
        <Button type="submit" busy={pending} disabled={!token.trim()}>Connect <ArrowRight size={17} /></Button>
        <small className="muted">Access is scoped to the jobs assigned to your account.</small>
      </form>
    </section>
  </main>;
}

function Shell() {
  const { disconnect } = useSession();
  const location = useLocation();
  const workspacePage = location.pathname.startsWith("/jobs/");
  return <div className="app-shell">
    <a className="skip-link" href="#content">Skip to content</a>
    <aside className="sidebar">
      <Link to="/jobs" className="brand-link" aria-label="Talent Screener home"><Brand /></Link>
      <nav aria-label="Main navigation">
        <div className="nav-label">WORKSPACE</div>
        <NavLink to="/jobs" className={({ isActive }) => `side-link ${isActive ? "selected" : ""}`}>
          <BriefcaseBusiness size={18} /> Jobs <ChevronRight className="side-indicator" size={16} />
        </NavLink>
      </nav>
      <div className="sidebar-bottom">
        <div className="sidebar-note">
          <ShieldCheck size={19} /><div><strong>Human-reviewed</strong><p>Every shortlist needs a reviewer’s approval.</p></div>
        </div>
        <Button className="disconnect" variant="ghost" onClick={disconnect} title="Disconnect workspace">
          <LogOut size={17} /> <span>Disconnect</span>
        </Button>
      </div>
    </aside>
    <div className="app-main">
      <header className="topbar">
        <div className="breadcrumbs"><Link to="/jobs">Workspace</Link><ChevronRight size={14} />
          {workspacePage ? <Link to="/jobs">Jobs</Link> : <span>Jobs</span>}
          {workspacePage && <><ChevronRight size={14} /><span>Job details</span></>}
        </div>
        <span className="connection"><span /> API connected</span>
      </header>
      <main id="content" className="main-content" tabIndex={-1}><Outlet /></main>
    </div>
  </div>;
}

function CreateJob({ onClose }: { onClose: () => void }) {
  const [files, setFiles] = useState<File[]>([]);
  const command = useCommand();
  const navigate = useNavigate();
  return <Dialog title="Create a job" description="Upload a job description to begin a requirements draft."
    onClose={onClose} busy={command.pending}>
    <form className="stack" onSubmit={async (event) => {
      event.preventDefault();
      if (!files[0] || command.pending) return;
      const body = new FormData();
      body.append("file", files[0]);
      const job = await command.submit<Job>("POST", JOBS, body);
      if (job) navigate(`${jobUrl(job.job_id)}/requirements`);
    }}>
      <FilePicker files={files} onChange={setFiles} rules={JD_FILES} disabled={command.pending} />
      <p className="muted small">Use a text-based PDF or UTF-8 TXT file, up to 10 MiB.</p>
      <ErrorNotice error={command.error} />
      <div className="dialog-actions"><Button variant="secondary" disabled={command.pending} onClick={onClose}>Cancel</Button>
        <Button type="submit" busy={command.pending} disabled={!files.length}>Start drafting <ArrowRight size={16} /></Button>
      </div>
    </form>
  </Dialog>;
}

function jobState(job: Job) {
  if (job.ranking_state === "current") return "Ranked";
  if (job.ranking_state === "rescore_required") return "Rescore required";
  if (job.active_profile) return "Requirements approved";
  if (job.draft.status === "failed") return "Draft failed";
  return job.draft.status === "ready" ? "Awaiting approval" : "Drafting";
}

function Jobs() {
  const jobs = useCollection<Job>(["jobs"], JOBS, true);
  const [create, setCreate] = useState(false);
  const [search, setSearch] = useState("");
  const [status, setStatus] = useState("all");
  const filtered = jobs.items.filter((job) => {
    const name = job.role_title ?? "Untitled job";
    return name.toLowerCase().includes(search.trim().toLowerCase())
      && (status === "all" || (status === "approved" ? Boolean(job.active_profile) : !job.active_profile));
  });
  return <div className="page-stack">
    <div className="page-heading">
      <div><p className="eyebrow">HIRING WORKSPACE</p><h1>Jobs</h1>
        <p>Manage screening criteria, applicant pools, and approved shortlists.</p></div>
      <Button onClick={() => setCreate(true)}><Plus size={19} /> New job</Button>
    </div>
    <div className="overview-banner">
      <div className="overview-copy"><div className="overview-icon"><ScanSearch size={26} /></div>
        <div><p className="eyebrow">REVIEW WORKFLOW</p><h2>From job description to shortlist</h2>
          <p>Evidence and intake findings are available at every stage of review.</p></div>
      </div>
      <div className="overview-flow" aria-label="Workflow steps">
        <span><ClipboardCheck size={17} /> Requirements</span><ArrowDownRight size={14} /><span><ShieldCheck size={17} /> Intake</span>
        <ArrowDownRight size={14} /><span><ListOrdered size={17} /> Ranking</span>
      </div>
    </div>
    <section className="panel">
      <div className="panel-heading toolbar">
        <div><h2>All jobs</h2><p>Open a job to pick up where you left off.</p></div>
        {jobs.items.length > 0 && <div className="toolbar-controls">
          <label className="search-field"><Search size={16} /><input value={search} onChange={(event) => setSearch(event.target.value)}
            placeholder="Search jobs…" aria-label="Search jobs" /></label>
          <select value={status} onChange={(event) => setStatus(event.target.value)} aria-label="Filter jobs by approval status">
            <option value="all">All statuses</option><option value="draft">Needs approval</option>
            <option value="approved">Requirements approved</option>
          </select>
        </div>}
      </div>
      {jobs.isPending ? <Loading>Loading jobs…</Loading> : jobs.isError ? <ErrorNotice error={jobs.error} retry={() => void jobs.refetch()} />
        : jobs.items.length === 0 ? <Empty icon={<BriefcaseBusiness size={25} />} title="Create your first job"
          message="Upload a job description and review the criteria before adding resumes.">
          <Button onClick={() => setCreate(true)}><Plus size={17} /> New job</Button>
        </Empty> : filtered.length === 0 ? <Empty icon={<Search size={24} />} title="No matching jobs"
          message={jobs.hasNextPage ? "Try a different search or load more jobs." : "Try a different search or status filter."} />
        : <div className="table-wrap"><table>
          <thead><tr><th>Role</th><th>Screening status</th><th>Active profile</th><th className="numeric"><span className="visually-hidden">Open</span></th></tr></thead>
          <tbody>{filtered.map((job) => <tr key={job.job_id}>
            <td><Link className="row-link" to={`${jobUrl(job.job_id)}/requirements`}>
              <span className="row-icon"><BriefcaseBusiness size={18} /></span>{job.role_title ?? "Untitled job"}
            </Link></td>
            <td><Chip tone={job.draft.status === "failed" ? "red" : job.ranking_state === "rescore_required" || !job.active_profile ? "amber" : "green"}>
              <span className="status-dot" />{jobState(job)}
            </Chip></td>
            <td>{job.active_profile ? `Requirements v${job.active_profile.version}` : "—"}</td>
            <td className="numeric"><Link className="icon-link" to={`${jobUrl(job.job_id)}/requirements`}
              aria-label={`Open ${job.role_title ?? "job"}`}><ArrowUpRight size={18} /></Link></td>
          </tr>)}</tbody>
        </table></div>}
      <LoadMore query={jobs} />
    </section>
    {create && <CreateJob onClose={() => setCreate(false)} />}
  </div>;
}

function RunProgress({ job }: { job: Job }) {
  const queryClient = useQueryClient();
  const notified = useRef<string | null>(null);
  const runId = job.latest_scoring_run_id;
  const runQuery = useRead<Run>([...jobKey(job.job_id), "run", runId],
    runId ? `${jobApi(job.job_id)}/scoring-runs/${runId}` : null,
    (run) => !run || run.status === "queued" || run.status === "running" ? 2200 : false);
  const run = runQuery.data;
  const inProgress = run?.status === "queued" || run?.status === "running";
  useEffect(() => {
    if (run && !inProgress && notified.current !== run.run_id) {
      notified.current = run.run_id;
      void queryClient.invalidateQueries({ queryKey: jobKey(job.job_id) });
    }
  }, [run, inProgress, queryClient, job.job_id]);
  if (!run && !runQuery.error) return null;
  if (runQuery.error) return <ErrorNotice error={runQuery.error} retry={() => void runQuery.refetch()} />;
  if (!run) return null;
  if (inProgress) {
    const done = run.scored_count + run.reused_count + run.error_count + run.skipped_count;
    return <div className="run-progress" role="status">
      <div><StateChip value={run.status} /> <strong>Scoring against Requirements v{run.profile_version}</strong></div>
      <div className="progress-track" aria-hidden="true"><div style={{ width: `${run.ready_count ? Math.min(100, 100 * done / run.ready_count) : 0}%` }} /></div>
      <span>{done} of {run.ready_count} eligible</span>
    </div>;
  }
  if (run.status === "failed") return <ErrorNotice error={run.error ?? "The scoring run failed. Start a new run from Resumes."} />;
  return run.status === "completed_with_errors"
    ? <Notice tone="amber">{plural(run.error_count, "scoring error")} in the last run. Successful results are available under Rankings &amp; Shortlist.</Notice>
    : null;
}

function Workspace() {
  const { jobId = "" } = useParams();
  const query = useRead<Job>(jobKey(jobId), jobApi(jobId), () => 4000);
  if (query.isPending) return <Loading>Opening job workspace…</Loading>;
  if (query.isError) return <div className="stack"><ButtonLink to="/jobs"><ArrowLeft size={16} /> All jobs</ButtonLink>
    <ErrorNotice error={query.error} retry={() => void query.refetch()} /></div>;
  const job = query.data;
  const ready = Boolean(job.active_profile);
  const ranking = job.ranking_state === "current";
  const stages = [
    { title: "Approve requirements", done: ready, current: !ready },
    { title: "Verify resumes", done: ranking, current: ready && !ranking },
    { title: "Review shortlist", done: false, current: ranking },
  ];
  return <div className="page-stack workspace">
    <header className="workspace-header">
      <div className="page-heading"><div><Link className="back-link" to="/jobs"><ArrowLeft size={15} /> All jobs</Link>
        <h1>{job.role_title ?? "Untitled job"}</h1><p>Review criteria, applicant evidence, and shortlist decisions for this role.</p></div>
        {job.active_profile ? <Chip tone="blue"><LockKeyhole size={13} /> Requirements v{job.active_profile.version}</Chip>
          : <Chip tone="amber"><ClipboardCheck size={14} /> Requirements awaiting approval</Chip>}
      </div>
      <ol className="workflow-steps" aria-label="Screening workflow">
        {stages.map((stage, i) => <li key={stage.title} className={stage.done ? "done" : stage.current ? "current" : ""}
          aria-current={stage.current ? "step" : undefined}>
          <span className="step-number">{stage.done ? <Check size={15} /> : `0${i + 1}`}</span>{stage.title}
        </li>)}
      </ol>
    </header>
    <nav className="tabs" aria-label="Job workspace">
      <NavLink to="requirements"><FileText size={17} /> Requirements</NavLink>
      <NavLink to="resumes"><UsersRound size={17} /> Resumes</NavLink>
      <NavLink to="rankings"><ListOrdered size={17} /> Rankings &amp; Shortlist</NavLink>
    </nav>
    <RunProgress job={job} />
    <Outlet context={job} />
  </div>;
}

export function App() {
  const { connected } = useSession();
  if (!connected) return <Connect />;
  return <Routes>
    <Route element={<Shell />}>
      <Route index element={<Navigate to="/jobs" replace />} />
      <Route path="jobs" element={<Jobs />} />
      <Route path="jobs/:jobId" element={<Workspace />}>
        <Route index element={<Navigate to="requirements" replace />} />
        <Route path="requirements" element={<Requirements />} />
        <Route path="resumes" element={<Resumes />} />
        <Route path="rankings" element={<Rankings />} />
      </Route>
      <Route path="*" element={<Empty icon={<Fingerprint size={25} />} title="Page not found"
        message="Use the jobs list to reopen your workspace."><ButtonLink to="/jobs"><FileCheck2 size={17} /> Jobs</ButtonLink></Empty>} />
    </Route>
  </Routes>;
}

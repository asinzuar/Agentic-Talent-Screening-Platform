# Web UI Flows: From Job Description to Shortlist

> Design spec for the target HR workflow, not a description of the implemented UI. See [the project spec](../SPEC.md#10-web-ui-and-hr-review-dashboard), [resume intake](../RESUME_INTAKE_SPEC.md), and [ranking](RANKING_SPEC.md) for pipeline rules.

## 1. Job workspace

HR opens a job from **Jobs**. Each job has one applicant pool and three tabs: **Requirements**, **Resumes**, and **Rankings & Shortlist**. The header identifies the role and the active requirement profile version.

**Upload JD → Review and approve requirements → Upload and verify resumes → Score ready candidates → Review and approve shortlist**

HR can return to **Add resumes** at any time after requirements approval. Upload starts intake, not scoring. The approved requirement profile governs matching and scoring for the job but is never sent to the resume intake lane.

## 2. Draft and approve requirements

1. From **Jobs → New job**, HR uploads a JD. The workspace shows drafting progress or an actionable error if the JD can't be processed. Resume uploads stay disabled until a profile is approved.
2. In **Requirements**, HR reviews the source JD alongside the draft. Each criterion shows its source quote or rationale, active status, priority (mandatory or preferred), weight, and any reviewer flags. HR can add, edit, or remove criteria and review details such as substitutes and minimum evidence. Inferred requirements and model suggestions remain inactive unless HR explicitly activates them. The screen also shows which JD content isn't used for scoring.
3. **Approve & lock** creates v1, records who approved it and when, and unlocks **Upload resumes**. The locked profile is read-only. **Create revision** opens a new draft while v1 stays active. Approving v2 requires rescoring the whole eligible pool before showing a current ranking for v2. Older results remain labeled with their original version; the dashboard never mixes versions.

## 3. Upload and verify resumes

HR selects one or more files in **Resumes → Upload resumes**. The upload dialog confirms the job and its active requirements version. The batch view shows per-file intake progress through document safety checks, injection detection, extraction, and evidence verification. After intake, the screening list uses blind candidate IDs.

| Status HR sees | Meaning and next action |
|---|---|
| **Processing** | Intake is running; the candidate is not yet eligible for scoring. |
| **Ready to score** | A Verified Profile is available and any required review has been cleared. Non-blocking and cleared flags stay visible. |
| **Review required** | Intake could not produce a usable Verified Profile, or a verified but flagged profile requires clearance. The candidate is held out of matching and scoring. |
| **Quarantined** | This file cannot proceed. HR can request a replacement, not release the quarantined file. |

**Review required** opens a review screen with the reason for the hold and the verified evidence, if any, shown separately from excluded content. Authorized HR can see a controlled excerpt and its page location when available. **The same authorized HR user who uploaded the resume can clear the hold**; a second reviewer is not required. Available actions:

- **Clear for scoring** an existing Verified Profile, with a required reason. This clears the hold without restoring excluded evidence or erasing flags.
- **Keep on hold** with a required reason. If there is no usable Verified Profile, retry when safe or request a replacement. There is no manual bypass of verification or quarantine.

The review disposition, actor, reason, and timestamp are recorded. A hold, clearance, or quarantine is an intake decision, not a hiring decision. Raw resume text and excluded spans do not pass to matching or scoring.

## 4. Score and review the shortlist

### Start a run

In **Resumes**, HR selects **Score ready resumes**. **HR can run scoring before the whole batch finishes intake.** The confirmation names the approved profile version and shows what is included and excluded: for example, *Score 18 of 24 uploaded resumes against Requirements v1. 4 are processing; 2 need review.* If none are ready, the disabled action explains why.

While scoring runs, show progress without replacing any previous results. When it finishes, show how many candidates were ranked and how many are still unranked, with reasons and a link back to the affected records. A scoring error leaves that candidate unranked without discarding successful results. Processing, held, quarantined, and failed records never receive a placeholder zero or a shortlist position.

### Review results

- **Ranked table:** rank, blind candidate ID, fit score, tier, shortlist status, and security and not-assessed flags. Show the role, requirement version, run time, ranked versus unranked counts, and security flag count.
- **Candidate detail:** show evidence by requirement, including evidence level, verified quote and location, and gaps. Distinguish **not met** from **not assessed**. An authorized audit view shows the risk, finding category, detection layer, and what was excluded. Show a counterfactual score difference only if one was actually computed (see [project spec Q26](../SPEC.md#12-open-questions)).
- **Shortlist:** HR chooses a size (N). The top N form the proposed shortlist; mark the next few candidates as just below the cutoff. **Approve shortlist** records the reviewed list. **Override** changes membership with a required reason without changing the computed score or rank. Audit and export actions identify the snapshot they refer to.

## 5. Add resumes and update the same pool

1. HR selects **Add resumes** from the dashboard. The new files use the same intake and review flow. Nothing is scored automatically when verification finishes or a hold is cleared.
2. HR selects **Score new ready resumes**. If the requirements, matcher, and scoring configuration are unchanged, score only candidates without a reusable result and keep the rest of the scores. Newly ready candidates include those cleared since the previous run, not just those recently uploaded. If the scoring basis changed, rescore the whole eligible pool instead.
3. Re-rank the entire pool of scored candidates and generate a new proposed top N. Show who entered or left the shortlist and their rank changes. For example, when a new candidate enters at #3, the previous #10 drops to #11 and leaves a top-10 shortlist. The last approved shortlist remains intact while HR reviews the new proposal.
4. **HR explicitly reconfirms or drops previous overrides.** Show each prior override and its reason alongside the new proposal; don't carry it forward automatically or discard it silently. Once HR resolves the overrides and approves, the new shortlist supersedes the previous one, which stays available as a historical snapshot.

## 6. Open questions

- **Approval and export permissions:** The same authorized HR user can upload and clear an intake hold. Whether a hiring manager must sign off on requirements remains open ([project spec Q13](../SPEC.md#12-open-questions)), as do the permissions and formats for shortlist exports.
- **Enforcement:** The workflow requires a logged clearance before a held profile can enter matching. Where that gate is enforced is still open ([project spec Q32](../SPEC.md#12-open-questions)).
- **Screen details:** Tab layout and action labels are _(inferred)_ UI choices. The wireframes, file preview affordances, and the exact retry and recovery controls are not specified here.

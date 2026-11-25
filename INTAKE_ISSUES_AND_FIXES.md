# Resume Intake: Issues Encountered and Fixes

This note captures the substantive issues found while exercising the resume-intake pipeline on clean and adversarial resume variants. It records the current behaviour and rationale; it is not an evaluation report or a claim that the security controls are production complete.

## 1. Semantic security judgement could block clean resumes

**Issue.** L1 has a deterministic detector and an always-on semantic LLM judge. Smaller local models sometimes returned an internally inconsistent structured judgement (for example, a medium/high risk with no meaningful evidence). The pipeline then safely returned `manual_review`, even when deterministic detection found no attack.

**Resolution.** The semantic judge remains intentionally enabled for every L0-approved resume: attacks can be paraphrased in ways rules cannot enumerate. Its output is schema-validated and every finding must quote text that exists in the source. Invalid/unavailable semantic output follows a safe fallback:

- if deterministic L1 found medium/high risk, sanitise its grounded spans, retain the flag, and continue on the clean remainder;
- if deterministic L1 found no signal, return `manual_review` rather than incorrectly treating the resume as clean.

The default local model was moved to `qwen2.5:3b-instruct` to improve structured-output reliability. The every-resume semantic-judge policy is documented in `RESUME_INTAKE_SPEC.md`.

## 2. Attacked resumes needed safe-remainder processing, not automatic rejection

**Issue.** It was unclear whether a detected injection should terminate intake. Automatically discarding every attacked resume would let an attacker deny consideration of otherwise legitimate evidence and would conflict with the project’s human-in-the-loop commitment.

**Resolution.** The implemented policy is:

1. Unsafe files (resource exhaustion, invalid type, unsafe parser condition) are quarantined before L2.
2. Suspicious text spans are excluded from L2 and evidence use.
3. If safe spans remain, L2 extracts only from the sanitised remainder.
4. High-risk security events are retained and require human review; they are never an automatic hiring or rejection decision.
5. If no usable clean evidence remains, the result is `manual_review` and is not eligible for ranking.

Every `PipelineResult` now includes `sanitised_span_ids`, so the exact spans removed before extraction are auditable.

## 3. L1 span matching could over-sanitise legitimate evidence

**Issue.** L1 mapped a detected excerpt to a source span using both forward and reverse substring matching. A short, normal span could be contained within a long malicious excerpt, causing unrelated legitimate evidence to be removed.

**Resolution.** Span resolution now only accepts spans that contain the full detected excerpt. Findings retain their resolved span IDs, and a semantic finding that cannot be grounded to L0 spans still routes safely to review rather than silently removing broad content.

## 4. Security findings in hidden text could not be actioned correctly

**Issue.** L0 correctly identified hidden or non-visible resume content, but the original grounding policy only allowed visible spans to support an L1 security finding. An attack contained only in hidden text could therefore be detected but could not be tied to an auditable span for a safe-remainder decision.

**Resolution.** Security findings may now be grounded to either visible or L0-excluded spans. This makes hidden-text attacks reviewable and ensures their span IDs appear in the security assessment. The boundary remains strict: non-visible spans are never supplied to L2, never become candidate evidence, and can never cross into ranking.

## 5. Multiple deterministic findings were not combined in final risk

**Issue.** The final risk decision initially considered individual grounded findings but not the deterministic detector's aggregate score. Several medium-confidence rule findings could therefore be represented as less severe than the combined signal warranted.

**Resolution.** Risk aggregation now includes the deterministic aggregate score. Multiple corroborating rule findings can produce a high-risk, human-review-required result, while still allowing L2 to process the safe remainder when usable evidence exists.

## 6. L2 output shape varied across clean and attacked variants

**Issue.** The local model sometimes copied datamarked span labels such as `marker:SPAN:p1:s0005` instead of returning the required canonical ID `p1:s0005`. One malformed claim could invalidate the entire L2 response, even if other candidate facts were usable.

**Resolution.** L2 now performs deterministic structural repair before strict validation:

- extracts canonical L0 span IDs from marker-decorated references;
- assigns a deterministic claim ID when only the claim ID is malformed;
- drops only claims with no recoverable source reference;
- never invents a fact, quote, or source span.

Repair and drop actions are recorded as events such as `L2_SOURCE_ID_NORMALISED` and `L2_CLAIM_DROPPED`. A genuinely unusable L2 model response routes to `manual_review`, not file quarantine.

## 7. L2-only evaluation needed an explicit safe mode

**Issue.** L3 evidence verification is currently too literal for normal model paraphrases: it requires the proposed claim text to appear verbatim in its cited span. This made clean resumes frequently end in `manual_review` while the team is still measuring L2 extraction quality.

**Resolution.** `--stop-after-l2` was added as a development-only command-line mode. It runs L0, L1, and schema-validated L2, then returns `l2_output` for extraction evaluation. It is explicitly not a ranking input and must not cross the trust boundary. Normal pipeline execution still requires L3 before producing a verified profile.

## 8. Intake and ranking used incompatible profile contracts

**Issue.** The intake lane emitted `VerifiedCandidateProfile` facts with `p#:s####` source IDs and an opaque blind ID, while ranking required `VerifiedProfile`, `p#:l#` locations, canonical skill IDs, and `C-####` candidate IDs. There was no explicit safe handoff.

**Resolution.** A deterministic adapter in `resume_intake/ranking_adapter.py` now bridges the contracts.

- Pipeline-generated blind IDs use the `C-####` format.
- Ranking accepts visible intake locations in either `p#:l#` or `p#:s####` form.
- Only L3-verified skill facts with an allow-listed, caller-supplied canonical skill mapping cross into ranking.
- Roles, projects, education, and location remain missing until they have an equivalent deterministic intake contract; ranking renormalises missing components rather than treating them as zero.
- Security flags are translated to ranking-compatible flags without forwarding raw resume text.

## 9. PII could be copied into extraction claims

**Issue.** Although L2 separates an administrative identity envelope from claims, a model could still copy contact information, links, location, or institution-bearing education text into a claim. That would violate the PII-blind ranking boundary.

**Resolution.** L3 excludes claims containing identity-envelope values, email addresses, phone numbers, URLs, or institution-bearing education text. The ranking adapter adds a second fail-closed PII check and only transfers allow-listed skill facts.

## 10. Machine-oriented events were unclear to human reviewers

**Issue.** Raw pipeline events such as `L0_FORENSICS_PASSED` were technically accurate but misleading: a file could pass forensic processing while still contain hidden spans that L0 excluded. The compact event codes also made it difficult for a reviewer to tell what happened, what was removed, and whether action was required.

**Resolution.** L0 now distinguishes a clean completion (`L0_FORENSICS_COMPLETED_CLEAN`) from a completion with exclusions (`L0_FORENSICS_COMPLETED_WITH_EXCLUSIONS`). `PipelineResult` also includes `event_details`, a reviewer-facing companion to the stable machine event codes. Each detail has a stage, severity, plain-language message, and relevant span IDs. This makes it clear when a security flag requires human review, when safe-remainder extraction continued, and when L2 output was validated.

## 11. Hosted model calls received contact PII

**Issue.** L1's semantic judge and L2 extraction received labelled source-span text. Although downstream ranking was PII-blind, those hosted model calls could include a candidate's email address, telephone number, contact location, or personal links.

**Resolution.** L0 now derives a separate LLM-safe span view while preserving the canonical source spans for deterministic processing and authorised audit. Email, phone, contact-location, and personal-link values are replaced with typed tokens before any LLM call; name is the explicit permitted exception. L1 semantic evidence is grounded against the safe view using the same stable span IDs, L2 discards any hallucinated contact fields, and the pipeline emits a non-sensitive `PII_REDACTED_BEFORE_MODEL` event when redaction occurred.

## Verification

The current automated suite covers L0, L1/L2/L3 hooks, ranking, the structural-repair path, and the new intake-to-ranking adapter.

```text
147 passed
```

Remaining planned work includes replacing literal-only L3 entailment with an evaluated span/NLI approach, defining complete deterministic role/project/education contracts for ranking, and measuring paired clean-versus-attacked resume consistency across the A1–A10 evaluation set.

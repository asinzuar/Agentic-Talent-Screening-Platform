# Worked examples: draft requirement profiles

Illustrative **draft** requirement profiles (see [SPEC §6.2](../../SPEC.md#62-requirement-profile)) for two sample JDs from the team's sample data set, which is kept outside this repo. An LLM drafted them in the role of the Job Analysis Agent (LLM-first, no O*NET). They show what a human reviewer would see **before** approval. Weights, minimum evidence levels and suggestions are starting points for the reviewer, not decisions.

**Legend**

- **Kind:** `skill` · `experience` · `education` · `leadership` · `achievement` (practices such as code review or Agile count as `skill`)
- **Pri:** M = Mandatory, P = Preferred
- **Wt:** H / M / L
- **Min E:** minimum evidence level (E1–E4, SPEC §6.1)
- **Prov:** **Q** = JD qualification · **R** = JD responsibility / summary · **S** = model suggestion
- **Status:** Active / *Inactive* (R and S items start inactive until a reviewer accepts them)

---

## JD 1: Senior Staff AI Engineer

Source: `Senior-staff-ai-engineer.txt`

| ID | Requirement | Kind | Pri | Wt | Yrs | Min E | Prov | Status | Flags |
|---|---|---|---|---|---|---|---|---|---|
| M1 | General-purpose language, any of: Python *(preferred)*, Java, Go | experience | M | H | 12–15 | E3 | Q: "12-15 years of experience with at least one general purpose programming language (Python [preferred], Java, or Go)" | Active | ⚠ Upper bound on years may be an age proxy |
| M2 | Front-end, any of: JavaScript, TypeScript | skill | M | M | | E3 | Q: "front-end experience with JavaScript/TypeScript" | Active | |
| M3 | Shipped production apps incorporating LLMs, RAG, MCP and AI services at scale | achievement | M | H | | E4 | Q: "Proven track record shipping production applications that incorporate LLMs, RAG, MCPs and AI services at scale" | Active | |
| M4 | Led technical teams; guided external contractors / vendors | leadership | M | H | | E3 | Q: "Demonstrated experience leading technical teams and providing guidance to external contractors or vendors" | Active | |
| M5 | Cloud platform, any of: AWS, GCP, Azure | skill | M | M | | E3 | Q: "Strong experience with cloud platforms (AWS, GCP, or Azure)" | Active | |
| M6 | Containerisation, all of: Docker, Kubernetes | skill | M | M | | E3 | Q: "advanced containerization (Docker, Kubernetes)" | Active | "all of" vs "any of" is ambiguous; reviewer to confirm |
| M7 | Technical oversight, code review, QA methodologies | skill | M | M | | E3 | Q: "Experience with technical oversight, code review processes, and quality assurance methodologies" | Active | |
| M8 | Bachelor's in CS, Computer Engineering or related | education | M | L | | — | Q: "Bachelor's degree in computer science, Computer Engineering, or a related technical field" | Active | ⚠ No "or equivalent experience" substitute |
| P1 | AI/ML architecture design and system optimisation | skill | P | M | | E3 | Q: "Thorough experience in AI/ML, including architectural design, technical leadership, and system optimization" | Active | |
| P2 | Front-end framework, any of: React, Vue, Angular; state management | skill | P | L | | E3 | Q: "modern front-end frameworks (React, Vue, or Angular) and advanced state management" | Active | |
| P3 | Vector DB, any of: Milvus, Pinecone, Weaviate, pgvector; embedding optimisation | skill | P | M | | E3 | Q: "vector databases (e.g. Milvus, Pinecone, Weaviate, pgvector) and embedding model optimization" | Active | |
| P4 | Multiple LLM providers; orchestration frameworks, any of: LangChain, LlamaIndex | skill | P | M | | E3 | Q: "multiple LLM providers and advanced AI-orchestration frameworks (e.g. LangChain, LlamaIndex)" | Active | |
| P5 | Enterprise RAG, semantic search, conversation memory at scale | skill | P | M | | E3 | Q: "enterprise RAG architectures, semantic search optimization, and conversation memory management at scale" | Active | Overlaps M3 |
| P6 | AI-powered features for 100k+ users | achievement | P | M | | E4 | Q: "building and scaling AI-powered features for 100k+ users" | Active | |
| R1 | Microservices and REST API design | skill | P | M | | E3 | R: "Ship resilient REST APIs and microservices." | *Inactive* | |
| R2 | Agentic workflows / agents | skill | P | M | | E3 | R: "Design agentic workflows for autonomous tasks." | *Inactive* | |
| R3 | Multimodal LLM features | skill | P | L | | E3 | R: "Build RAG, Multimodal LLM features and Agents." | *Inactive* | |
| R4 | Prompt and model-performance testing / evaluation | skill | P | M | | E3 | R: "establish testing for prompts and model performance" | *Inactive* | |
| R5 | Observability for AI features | skill | P | L | | E3 | R: "implement observability for AI features" | *Inactive* | |
| R6 | Cloud cost optimisation for AI systems | skill | P | L | | E3 | R: "balancing reliability, performance, and cloud cost for AI systems" | *Inactive* | |
| R7 | Agile delivery | skill | P | L | | E2 | R: "drive Agile practices" | *Inactive* | |
| S1 | LLM guardrails / safety (prompt-injection defence, output validation) | skill | P | L | | E3 | S: standard concern when shipping enterprise LLM features; not mentioned in the JD | *Inactive* | |

**Not used for scoring:** "We value autonomy, continuous learning, and measurable impact." These are culture statements that can't be judged from a resume.

---

## JD 2: Senior Software Engineer, AI/ML (Google Search)

Source: `Sr-software-engg-AIML.txt`

| ID | Requirement | Kind | Pri | Wt | Yrs | Min E | Prov | Status | Flags |
|---|---|---|---|---|---|---|---|---|---|
| M1 | Software development, any language | experience | M | H | ≥ 5 | E3 | Q: "5 years of experience with software development in one or more programming languages" | Active | |
| M2 | Testing, maintaining or launching software products | experience | M | M | ≥ 3 | E3 | Q: "3 years of experience testing, maintaining, or launching software products" | Active | |
| M3 | Software design and architecture | experience | M | M | ≥ 1 | E3 | Q: "1 year of experience with software design and architecture" | Active | |
| M4 | Specialised ML area, any of: speech/audio, reinforcement learning, ML infrastructure, *other ML field* | experience | M | H | ≥ 3 | E3 | Q: "3 years of experience with one or more of the following: speech/audio…, reinforcement learning…, ML infrastructure, or specialization in another ML field" | Active | ⚠ Open-ended option; reviewer to narrow to the team's area |
| M5 | End-to-end ML: deployment, evaluation, data processing, debugging | experience | M | H | ≥ 3 | E3 | Q: "3 years of experience with end-to-end machine learning (e.g., model deployment, model evaluation, data processing, debugging)" | Active | |
| M6 | Bachelor's degree; **substitute:** equivalent practical experience | education | M | L | | — | Q: "Bachelor's degree or equivalent practical experience" | Active | |
| P1 | Master's or PhD in CS or related | education | P | L | | — | Q: "Master's degree or PhD in Computer Science or related technical field" | Active | |
| P2 | Data structures and algorithms | experience | P | L | ≥ 5 | E2 | Q: "5 years of experience with data structures and algorithms" | Active | ⚠ Hard to evidence from a resume |
| P3 | Technical leadership role | leadership | P | M | ≥ 1 | E3 | Q: "1 year of experience in a technical leadership role" | Active | |
| P4 | Accessible technologies | skill | P | L | | E3 | Q: "Experience developing accessible technologies" | Active | |
| R1 | Design and code reviews; engineering best practices | skill | P | L | | E3 | R: "Collaborate with peers and stakeholders through design and code reviews" | *Inactive* | |
| R2 | Production triage and debugging | skill | P | L | | E3 | R: "Triage product or system issues and debug/track/resolve" | *Inactive* | Overlaps M5 |
| R3 | Technical documentation | skill | P | L | | E2 | R: "Contribute to existing documentation or educational content" | *Inactive* | |
| S1 | Information retrieval / search ranking | skill | P | M | | E3 | S: the role sits on the Google Search team | *Inactive* | |
| S2 | Large-scale distributed systems | skill | P | M | | E3 | S: "handle information at massive scale"; distributed computing listed among focus areas | *Inactive* | |
| S3 | NLP / LLMs | skill | P | M | | E3 | S: "reimagining what it means to search"; NLP listed among focus areas | *Inactive* | |

**Not used for scoring:**

- "this position requires in-person interviews" (interview logistics)
- the "About the job" company and team description (boilerplate; used only as context for S1–S3)
- the equal-opportunity / EEO statement

---

## What these examples surfaced

These findings drove the requirement fields in SPEC §6.2 and Open Questions #16 and #18–20:

- Requirements come in several kinds, not just skills.
- "Any of" / "all of" option groups are common, sometimes with a preferred option and sometimes open-ended.
- Years of experience need computing from resume dates; the evidence ladder doesn't cover them.
- Vague wording ("strong", "proven") maps naturally to a minimum evidence level.
- Substitutions such as "or equivalent practical experience".
- Requirements hidden in responsibilities, which need their own provenance label.
- JD content that shouldn't affect scoring.
- Overlapping requirements that risk scoring the same evidence twice.
- Real discriminatory-requirement candidates: a years upper bound, and a degree requirement with no "or equivalent".

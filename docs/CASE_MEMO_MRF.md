# Case Memo: The Framework That Became a Contour

*How an external reasoning pipeline was superseded as a product while its control concerns resurfaced in later safety work*

## 1. Executive Summary

**Retrospective interpretation.** Modular Reasoning Framework (MRF) began as an external pipeline for making inexpensive language-model responses more structured: specify a task, plan it, execute steps, verify results, reflect, and synthesize an answer. It did not become a continuing product. The differentiation of the generic pipeline diminished as model-native reasoning improved and mature agent runtimes supplied similar orchestration facilities.

Its durable value was different from its original product claim. MRF made several control concerns visible: scope can be bounded, intermediate state can be observed, arithmetic and tools can be checked outside generated prose, dependencies can be represented, and a result should leave evidence. Later work treats those concerns not as a generic reasoning product but as execution governance. MRF therefore failed as a continuing standalone product hypothesis but succeeded as an architectural ancestor and governance lesson.

This memo is a historical and technical closeout. It does not establish current runtime behavior, model quality, clinical safety, security, or production suitability. MRF is inactive, unmaintained, and not recommended for production or clinical use.

## 2. Epistemic Status

This memo uses five labels to keep evidence and interpretation separate.

- **Observed** means visible in the repository, the restored source, or the documented closeout checks.
- **Author recollection** means a historical account supplied by the project author and not independently reconstructed from repository artifacts.
- **Inference** means a constrained explanation drawn from the available observations; it is not a proven historical fact.
- **Retrospective interpretation** means a portfolio-level reading made after the fact, not a claim of direct causality or code reuse.
- **External research context** identifies claims attributed to the cited literature rather than to MRF.

## 3. Context

**Observed.** The restored historical source presented an external sequence of specification, planning, execution, verification, reflection, and synthesis. It belongs to the period in which prompting and multi-call orchestration were commonly used to improve structured reasoning from comparatively fast and inexpensive models. The source’s own reference list places the project beside Chain-of-Thought Prompting, Self-Consistency, Tree of Thoughts, and STaR as intellectual context. That context does not establish novelty, priority, or equivalent results.

**Retrospective interpretation.** The useful historical question is not whether a six-stage loop was a permanently differentiated product. It is what the attempt revealed about where a dependable boundary belongs when a model participates in an action path. MRF made the model’s intermediate work more visible, but visibility alone did not make the overall system independently reliable.

## 4. Original Hypothesis

**Author recollection.** MRF began as an attempt to improve the quality of fast, inexpensive language models by decomposing problem solving into explicit external stages. The aim was to impose structure around model interactions and to use external computation or tools where available rather than relying entirely on generated text. The intended payoff was a more inspectable and disciplined workflow for tasks that appeared multi-step.

**Observed.** The source reflects this hypothesis in its named stages, structured intermediate objects, evaluator, provider interface, and verification-oriented components. Some checks were external to generated prose; other verification was model-mediated. The latter was not formal verification, and a staged representation did not by itself provide an independent proof of correctness.

## 5. Evolution of the Product Hypothesis

### Phase 1: Improving fast and inexpensive model responses

**Author recollection.** The first phase focused on using a scaffold to improve answers from cost-sensitive models. The architecture separated task description, planning, tool or computation execution, checking, reflection, and final synthesis. This was a reasonable response to the practical limits of short, inexpensive model calls at the time, but it introduced multiple calls, intermediate parsing, and orchestration overhead.

### Phase 2: Local and clinical deployment hypothesis

**Author recollection.** The project was later explored conceptually as a possible architecture for privacy-sensitive, on-premise clinical environments, where locally hosted small models and constrained compute could matter. This remained a hypothesis about an architectural setting, not an implementation in care delivery.

**Observed boundary.** MRF was never integrated into a hospital. No clinical workflow, privacy assessment, security assessment, regulatory validation, clinical evaluation, or deployment was completed. Nothing in this repository supports clinical use.

### Phase 3: Reasoning moved inside the model

**Retrospective interpretation.** As model-native reasoning capabilities and agent runtimes matured, the value of a generic external multi-call reasoning scaffold declined. More reasoning could be represented inside model inference or supplied by established runtime components, while an external pipeline continued to impose latency, integration burden, and its own failure modes. This did not make MRF historically meaningless; it made its original product boundary less durable.

### Phase 4: Small and local models remained relevant

**External research context.** The continuing relevance of on-premise inference is a field-level point, not validation of MRF. Sommer, Till, and Matthes describe a fully local, two-stage pipeline for medical CRF filling using MedGemma-27B and report their own task results; their work is evidence that privacy-oriented local clinical NLP remains active research, not evidence that MRF was clinically valid. The MedGemma 1.5 technical report likewise documents active development of an open medical foundation model. Its reported benchmark improvements do not establish clinical safety, deployment readiness, or a link to MRF.

### Phase 5: Recurrent inference and termination control

**External research context.** Reusing model layers or increasing inference-time recurrence can improve reasoning depth in some small models. Zhu et al. present looped latent computation as a scaling direction for smaller language models and report strong benchmark results for their own models; those results were not independently reproduced for this memo. Kohli et al. study recurrent-depth transformers, including depth extrapolation through additional inference-time recurrence, and identify a reported limitation they call *overthinking*, where excessive recurrence can degrade predictions.

**Inference.** Bounded recurrent inference is not the same thing as an external software loop with no termination guard. Either design can require controls over latency, diminishing returns, compute budget, and stopping behavior. Excessive recurrence can produce overthinking, unstable stopping behavior, or budget exhaustion; it need not imply an infinite loop. The product direction that follows is not to revive MRF’s generic loop, but to make stopping rules and action controls explicit around model execution.

## 6. What Was Actually Built

**Observed.** The restored historical `mrf.py` snapshot is 1,096 lines and compiles successfully. It contains an AST-gated arithmetic and logical evaluator, JSONL execution tracing, plan-dependency handling, a tool registry, provider dependency injection, a mock-provider pattern, and structured intermediate objects. The code also represents the six-stage architecture in named program structures.

These components show that the project contained more than a prompt template. They do not show that its stages were independently validated, that providers behaved consistently, or that the system performed reliably under real operational conditions. In particular, a model-mediated verification step cannot be promoted to formal verification merely because it has a verification label.

## 7. Evidence Boundary

**Observed.** No automated test suite exists in the historical repository. No benchmarks, clinical-validation artifacts, security audit, or production-deployment receipts were found during closeout. The historical source compiled during the restore checks. The runtime demo was not executed during closeout because `pydantic` was unavailable in the audit environment; dependencies were not installed in order to preserve the scope of the audit.

This boundary matters for both favorable and unfavorable conclusions. The repository supports statements about source structure and history. It does not support quantified claims about quality, safety, security, cost, reliability, clinical usefulness, or comparative model performance.

## 8. The Final-Snapshot Failure Signal

**Observed.** The final cleanup snapshot replaced `mrf.py` with a shorter `mrf_v1.1.py`. That replacement contained typographic quotation marks, literal Markdown fences, and a damaged entry point. The final snapshot was therefore a failure signal: it was not a trustworthy representation of the preceding historical source.

The closeout restored the exact prior `mrf.py` blob as a separate historical-source commit. This is source preservation, not a claim that the historical code is ready to run or that it should be returned to active maintenance.

## 9. Probable Transfer Mechanism

**Author recollection.** Code was manually copied and pasted into GitHub during the cleanup period.

**Inference.** The final repository snapshot contained typographic quotation marks, literal Markdown fences, and a damaged entry point. The author recalls manually copying code into GitHub during this period. Manual transfer contamination is therefore a plausible explanation, but the exact insertion point was not independently reconstructed.

This is intentionally a bounded account. The repository gives a visible failure signal and the author supplies a recollection about the transfer process; neither establishes the precise mechanism or time of every corruption.

## 10. What MRF Got Right

**Retrospective interpretation.** Several design instincts remain useful even though the product boundary did not endure.

- Useful work can be decomposed into observable stages.
- Arithmetic and tool execution should not rely exclusively on model-generated text.
- Dependencies and traces matter when an output will affect downstream work.
- Provider abstraction and mocks can improve testability.
- Verification needs external artifacts rather than a self-description that something was checked.

These are not claims that MRF solved the associated problems. They are design constraints that became clearer through the project’s limitations.

## 11. What MRF Misidentified

**Retrospective interpretation.** MRF treated the generic reasoning pipeline as the durable product boundary. That overestimated what multiple stages could guarantee. More stages did not automatically create independent verification; reflection did not guarantee correction; and additional reasoning did not guarantee reliable action. A generic multi-call scaffold also created latency and termination complexity of its own.

The more durable question is not “how can a model reason through one more loop?” It is “what may this system do, under which evidence, within which capability and stopping limits, and with whose authorization?”

## 12. Surviving Technical Primitives

**Observed and retrospective interpretation.** The surviving primitives are not the historical pipeline as a unit. They are explicit task boundaries, structured intermediate objects, external evaluators, tool registration, provider seams, mock patterns, dependency handling, and trace records. Each supports inspection or controlled substitution. None should be mistaken for a complete assurance case by itself.

These primitives are useful when reassembled under stronger contracts: code-backed checks instead of model agreement, recorded evidence instead of untraceable completion, and explicit scope limits instead of an assumed general-purpose reasoning agent.

## 13. From Reasoning Orchestration to Execution Governance

**Retrospective interpretation.** Later work shifted the center of gravity from reasoning orchestration to execution governance. The control contour includes bounded scope, deterministic gates, evidence contracts, execution receipts, dependency review, capability limits, stopping rules, and explicit human authorization.

This shift does not require a claim that later projects directly reuse MRF code. It is a portfolio interpretation of how the problem framing changed: internal reasoning can remain uncertain, while actions can still be constrained by explicit contracts and independently checkable evidence.

## 14. Portfolio Descendants

**Retrospective interpretation.** Two later portfolio artifacts illustrate this later emphasis, without proving direct causality or code lineage:

- [EBAC-T4 — Deterministic Trace-Bound Authorization for High-Risk EHR Writes](https://doi.org/10.5281/zenodo.21281597)
- [Clinical Verifiable Gates (CVG)](https://doi.org/10.5281/zenodo.21328660)

They are cited as separately identified portfolio artifacts. Their existence does not establish that MRF supplied their implementation, that their controls are equivalent, or that they validate MRF.

## 15. Product Decision

**Retrospective interpretation.** MRF should remain a closed historical prototype rather than be revived as a standalone product. The reason is not that external structure became useless, or that small and local models ceased to matter. It is that the generic multi-call reasoning loop was not the right durable product boundary, and the repository lacks evidence needed to support active use.

The appropriate disposition is archival review: preserve the restored source, retain the historical documentation, keep the evidence boundary visible, and avoid claims that the project reached production or clinical readiness.

## 16. Lesson for Vector

**Retrospective interpretation.** “The durable system was not the reasoning loop itself. It was the control contour around execution: bounded scope, observable state, deterministic checks, receipts, and explicit human authorization.”

For future readers of Vector, the central lesson is that improvement in reasoning depth is not a substitute for governance. A system can allow sophisticated internal inference while still requiring external evidence, bounded authority, and a known stopping condition before it acts.

“MRF began as a system for making models think longer. Its descendants became systems for deciding what models are allowed to do after thinking.”

## 17. References

1. Wei, Jason, et al. “Chain-of-Thought Prompting Elicits Reasoning in Large Language Models.” arXiv:2201.11903.
   https://arxiv.org/abs/2201.11903
2. Wang, Xuezhi, et al. “Self-Consistency Improves Chain of Thought Reasoning in Language Models.” arXiv:2203.11171.
   https://arxiv.org/abs/2203.11171
3. Yao, Shunyu, et al. “Tree of Thoughts: Deliberate Problem Solving with Large Language Models.” arXiv:2305.10601.
   https://arxiv.org/abs/2305.10601
4. Zelikman, Eric, et al. “STaR: Bootstrapping Reasoning With Reasoning.” arXiv:2203.14465.
   https://arxiv.org/abs/2203.14465
5. Zhu, Rui-Jie, et al. “[Scaling Latent Reasoning via Looped Language Models](https://arxiv.org/abs/2510.25741).” arXiv:2510.25741. Used only for the attributed claim that looped latent computation is being explored as a scaling direction for smaller language models; reported benchmark results are the authors’ results.
6. Kohli, Harsh, et al. “[Loop, Think, & Generalize: Implicit Reasoning in Recurrent-Depth Transformers](https://arxiv.org/abs/2604.07822).” arXiv:2604.07822. Used for recurrent-depth reasoning, depth extrapolation, and the authors’ reported overthinking limitation.
7. Sommer, Katharina, Tristan Till, and Florian Matthes. “[sebis at CRF Filling 2026: A Two-Stage Local LLM Pipeline for Medical CRF Filling](https://arxiv.org/abs/2606.13082).” arXiv:2606.13082. Used as external context that fully local, privacy-oriented clinical NLP pipelines remain active research; its reported performance is attributed to the authors.
8. Sellergren, Andrew, et al. “[MedGemma 1.5 Technical Report](https://arxiv.org/abs/2604.05081).” arXiv:2604.05081. Used only as broader evidence that smaller open medical foundation models remain under active development, not as evidence of clinical safety or deployment readiness.

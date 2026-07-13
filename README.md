# Modular Reasoning Framework (MRF)

## Status: Historical closeout — July 2026

MRF is inactive and unmaintained. It is retained as a historical source artifact and is not recommended for production or clinical use.

## What MRF Was

MRF began as an attempt to improve the quality of fast, inexpensive language models by decomposing problem solving into explicit external stages. The project was later explored as a possible architecture for privacy-sensitive, on-premise clinical environments where locally hosted small models and constrained compute could matter. No hospital integration, clinical workflow, privacy assessment, regulatory validation, or clinical deployment was completed.

## Historical Architecture

The historical design used the following six-stage architecture:

```
SPECIFICATION → PLANNING → EXECUTION → VERIFICATION → REFLECTION → SYNTHESIS
```

This architecture is preserved as historical context. It combined structured model interactions with external checks where available; it did not establish a formal-verification system.

## Observed Technical Assets

The restored historical source contains an AST-gated arithmetic and logical evaluator, JSONL execution tracing, plan-dependency validation, a tool registry, provider dependency injection, a mock-provider pattern, and structured intermediate objects. These are observed code assets, not evidence of production readiness or validated performance.

## Evidence Boundary and Known Limitations

This repository contains no automated tests, benchmarks, clinical validation, security audit, or production-deployment receipts. The historical source is preserved for inspection, but this closeout does not establish current runtime behavior, safety, security, accuracy, or suitability for any deployment. In particular, MRF must not be used as a basis for clinical decisions or high-risk production workflows.

## Why Active Development Stopped

Native model reasoning and mature agent runtimes reduced the value of a generic external multi-call reasoning scaffold. Iterative and recurrent inference also moved more reasoning inside the model: additional inference can improve some small-model outputs, but it introduces latency, diminishing returns, compute-budget, and termination-control concerns. The resulting direction was not to revive MRF, but to place explicit stopping rules, deterministic gates, receipts, and capability controls around model execution. That change does not mean that small or local models lost useful roles; it means this particular generalized architecture was no longer the project’s active direction.

## Project Heritage

Within this portfolio, MRF is preserved as the first substantial self-directed AI engineering project and an early implementation of external orchestration principles. Its durable contribution was not the six-stage reasoning loop itself, but the transition toward execution-contour governance: bounded scope, observable state, deterministic checks, receipts, dependency review, and explicit human authorization.

## Portfolio Lineage

The connection between MRF and later projects is a retrospective interpretation by the author, not evidence of proven causal lineage. Related later portfolio artifacts include:

- [EBAC-T4 — Deterministic Trace-Bound Authorization for High-Risk EHR Writes](https://doi.org/10.5281/zenodo.21281597)
- [Clinical Verifiable Gates (CVG)](https://doi.org/10.5281/zenodo.21328660)

## Case Memo

The full historical and technical analysis is documented in the [MRF case memo](docs/CASE_MEMO_MRF.md), including the evolution from fast-model reasoning support to local clinical deployment hypotheses, model-native reasoning, recurrent inference, termination risks, and execution governance.

## License, Author, and Acknowledgments

Licensed under the [MIT License](LICENSE).

**Author:** Sergey Morev — project initiator, architecture, implementation, and project leadership.

**Acknowledgments:** AI-assisted development and review involved GPT, Claude, and Gemini systems.

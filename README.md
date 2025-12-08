# Modular Reasoning Framework (MRF)

External reasoning orchestration system for fast language models.

## Problem

Fast LLMs (Gemini Flash, GPT-4o-mini, Claude Haiku) are cost-effective but lack structured reasoning capabilities. Native chain-of-thought is either unavailable or expensive.

## Solution

MRF provides an external reasoning scaffold that forces structured problem-solving:

```
SPECIFICATION → PLANNING → EXECUTION → VERIFICATION → REFLECTION → SYNTHESIS
```

The model never "thinks out loud" — all reasoning happens through controlled JSON stages with external verification.

## Key Features

- **Safe math evaluation** — AST-based, no `eval()`, Decimal precision
- **Self-consistency** — Multiple candidates with voting
- **DAG validation** — Dependency checking, cycle detection
- **Tool registry** — Pluggable external functions
- **Quality gates** — Definition of Done (DoD) checks before output
- **Experiment tracking** — Lightweight JSONL logger, no external dependencies

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     User Query                               │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  SPEC: Define problem structure, constraints, outputs        │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  PLAN: Break into steps, identify dependencies (DAG)         │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  SOLVE: Execute steps (math via SafeEvaluator, tools)        │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  VERIFY: Check results against constraints                   │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  REFLECT: Model critiques own plan, patches if needed        │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  EMIT: Synthesize verified results into final answer         │
└─────────────────────────────────────────────────────────────┘
```

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```python
from mrf import ReasoningOrchestrator, Config

config = Config(
    candidates=3,           # Self-consistency voting
    max_reflect_loops=2,    # Reflection iterations
    timeout_seconds=30
)

orchestrator = ReasoningOrchestrator(llm_provider, config)
result = await orchestrator.solve("What is 15% of 847?")
```

## Use Cases

- **Math problems** — With external verification, not model arithmetic
- **Multi-step reasoning** — Complex queries broken into dependency graph
- **Batch processing** — Cost-effective reasoning at scale
- **Education** — Explainable reasoning traces

## Limitations

- Adds latency (multiple LLM calls per query)
- JSON parsing can fail on malformed model output
- Not suitable for creative/open-ended tasks

## Prior Art & Novelty

A systematic search was conducted across arXiv, GitHub, and patent databases (November 2024).

**Related work:**

| Project | Similarity | Difference |
|---------|------------|------------|
| ReWOO (LangGraph) | Plan-Worker-Solver pattern | Not optimized for fast/cheap models |
| OpenR | Test-time compute | Focused on RL training, not scaffolding |
| Self-Consistency | Voting mechanism | No structured spec→plan→execute pipeline |
| ART | Automatic reasoning steps | General purpose, not cost-optimized |

**What MRF adds:**

- **Cost-first design** — Explicitly targets Gemini Flash / GPT-4o-mini tier
- **External safe evaluation** — Math computed via Python AST, not in model's "head"
- **DoD quality gates** — Definition of Done checks before output
- **Production focus** — Built for batch processing, not research demos

**Novelty assessment:** 6.5-7.5/10 — Novel combination of established techniques for a specific underserved problem (cost-quality tradeoff in reasoning).

**Patents:** No direct patents found on this approach.

## References

- Wei et al. (2022): Chain-of-Thought Prompting
- Wang et al. (2023): Self-Consistency Improves Chain of Thought
- Yao et al. (2023): Tree of Thoughts
- Xu et al. (2023): ReWOO — Reasoning WithOut Observation

## License

MIT

## Author

[Your Name] — Architecture, Implementation

With contributions from collaborative AI development process.

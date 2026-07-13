#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modular Reasoning Framework (MRF) v1.1
========================================
An external reasoning orchestration system for fast language models
that implements structured chain-of-thought without model-internal reasoning traces.

Authors:
  - Segey Morev - Architecture, Implementation, Orchestration
  - Igor (GPT) - Orchestrator design, Safety improvements, Patches
  - Andrey (Claude Opus) - Production hardening, Research-grade refactoring

License: MIT

Architecture:
    Specification → Planning → Execution → Verification → Reflection → Synthesis
    with self-consistency voting, formal verification gates, and DAG validation.

References:
    - Wei et al. (2022): Chain-of-Thought Prompting
    - Wang et al. (2023): Self-Consistency Improves Chain of Thought
    - Yao et al. (2023): Tree of Thoughts
    - Zelikman et al. (2022): STaR: Self-Taught Reasoner
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import hashlib
import os
import pathlib
import re
import ast
import operator as op
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import List, Dict, Any, Optional, Callable, TypeVar, Union
from collections import Counter
from decimal import Decimal, getcontext, DivisionByZero, InvalidOperation

from pydantic import BaseModel, Field, field_validator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Type definitions
T = TypeVar('T')


# ============================================================================
# CORE ABSTRACTIONS
# ============================================================================

class ReasoningStage(Enum):
    """Enumeration of reasoning pipeline stages."""
    SPECIFICATION = "specification"
    PLANNING = "planning"
    EXECUTION = "execution"
    VERIFICATION = "verification"
    REFLECTION = "reflection"
    SYNTHESIS = "synthesis"


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    async def complete(self, 
                      messages: List[Dict[str, str]], 
                      temperature: float = 0.0,
                      max_tokens: Optional[int] = None) -> str:
        """Generate completion from messages."""
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Return model metadata."""
        pass


# ============================================================================
# PYDANTIC MODELS (v2 compatible)
# ============================================================================

class TaskSpecification(BaseModel):
    """Formal specification of a reasoning task."""
    goal: str = Field(description="Primary objective")
    outputs: List[str] = Field(default_factory=list, description="Expected output formats")
    knowns: List[Dict[str, Any]] = Field(default_factory=list, description="Given information")
    unknowns: List[str] = Field(default_factory=list, description="Variables to solve for")
    constraints: List[str] = Field(default_factory=list, description="Mathematical/logical constraints")
    success_criteria: List[str] = Field(default_factory=list, description="Testable conditions")

    @field_validator('knowns', mode='before')
    @classmethod
    def coerce_knowns(cls, v):
        """Ensure knowns is a list of dicts."""
        if isinstance(v, dict):
            return [v]
        return v or []


class ReasoningStep(BaseModel):
    """Single step in a reasoning plan."""
    step_type: str = Field(description="compute|derive|lookup|verify|tool")
    expression: str = Field(default="", description="Mathematical expression if applicable")
    description: str = Field(description="What this step does")
    dependencies: List[int] = Field(default_factory=list, description="Indices of prerequisite steps")
    expected_output: str = Field(default="", description="Description of result")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    tool: Optional[str] = Field(default=None, description="Tool name if step_type is 'tool'")
    args: Dict[str, Any] = Field(default_factory=dict, description="Tool arguments")


class ReasoningPlan(BaseModel):
    """Complete reasoning plan with verification tests."""
    steps: List[ReasoningStep] = Field(default_factory=list)
    verification_tests: List[Dict[str, str]] = Field(default_factory=list)
    estimated_complexity: str = Field(default="O(n)")
    plan_confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class ExecutionResult(BaseModel):
    """Results from plan execution."""
    step_results: Dict[int, Any] = Field(default_factory=dict)
    verification_outcomes: List[Dict[str, Any]] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    success_rate: float = Field(default=0.0)


# ============================================================================
# SAFE MATH EVALUATOR (AST-based, no eval!)
# ============================================================================

class SafeMathEvaluator:
    """
    Safe arithmetic evaluator using AST parsing.
    Supports: +, -, *, /, //, %, **, comparisons, and/or/not, parentheses.
    Uses Decimal for precision.
    """
    
    SAFE_OPS = {
        ast.Add: op.add,
        ast.Sub: op.sub,
        ast.Mult: op.mul,
        ast.Div: op.truediv,
        ast.FloorDiv: op.floordiv,
        ast.Mod: op.mod,
        ast.Pow: op.pow,
        ast.USub: lambda x: -x,
        ast.UAdd: lambda x: +x,
        # Comparisons
        ast.Eq: op.eq,
        ast.NotEq: op.ne,
        ast.Lt: op.lt,
        ast.LtE: op.le,
        ast.Gt: op.gt,
        ast.GtE: op.ge,
    }
    
    SAFE_FUNCS = {
        'sqrt': lambda x: Decimal(x).sqrt(),
        'abs': abs,
        'min': min,
        'max': max,
        'round': round,
        'int': int,
        'float': float,
    }
    
    def __init__(self, precision: int = 28):
        getcontext().prec = precision
        self.vars: Dict[str, Decimal] = {}
    
    def set_vars(self, variables: Dict[str, Any]):
        """Set variables for evaluation."""
        self.vars = {k: Decimal(str(v)) for k, v in variables.items()}
    
    def evaluate(self, expr: str) -> Union[Decimal, bool]:
        """
        Safely evaluate a mathematical expression.
        Returns Decimal for numeric results, bool for comparisons.
        """
        try:
            tree = ast.parse(expr, mode='eval')
            return self._eval_node(tree.body)
        except Exception as e:
            raise ValueError(f"Cannot evaluate '{expr}': {e}")
    
    def _eval_node(self, node: ast.AST) -> Union[Decimal, bool]:
        """Recursively evaluate AST nodes."""
        
        # Numbers
        if isinstance(node, ast.Constant):
            return Decimal(str(node.value))
        
        # Older Python compatibility
        if isinstance(node, ast.Num):
            return Decimal(str(node.n))
        
        # Variables
        if isinstance(node, ast.Name):
            if node.id in self.vars:
                return self.vars[node.id]
            if node.id == 'True':
                return True
            if node.id == 'False':
                return False
            raise ValueError(f"Unknown variable: {node.id}")
        
        # Unary operators
        if isinstance(node, ast.UnaryOp):
            operand = self._eval_node(node.operand)
            op_type = type(node.op)
            if op_type == ast.Not:
                return not operand
            if op_type in self.SAFE_OPS:
                return self.SAFE_OPS[op_type](operand)
            raise ValueError(f"Unsupported unary operator: {op_type}")
        
        # Binary operators
        if isinstance(node, ast.BinOp):
            left = self._eval_node(node.left)
            right = self._eval_node(node.right)
            op_type = type(node.op)
            if op_type in self.SAFE_OPS:
                return self.SAFE_OPS[op_type](left, right)
            raise ValueError(f"Unsupported binary operator: {op_type}")
        
        # Comparisons
        if isinstance(node, ast.Compare):
            left = self._eval_node(node.left)
            for op_node, comparator in zip(node.ops, node.comparators):
                right = self._eval_node(comparator)
                op_type = type(op_node)
                if op_type not in self.SAFE_OPS:
                    raise ValueError(f"Unsupported comparison: {op_type}")
                if not self.SAFE_OPS[op_type](left, right):
                    return False
                left = right
            return True
        
        # Boolean operations (and/or)
        if isinstance(node, ast.BoolOp):
            if isinstance(node.op, ast.And):
                return all(self._eval_node(v) for v in node.values)
            if isinstance(node.op, ast.Or):
                return any(self._eval_node(v) for v in node.values)
        
        # Function calls
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in self.SAFE_FUNCS:
                args = [self._eval_node(arg) for arg in node.args]
                return Decimal(str(self.SAFE_FUNCS[node.func.id](*args)))
            raise ValueError(f"Unsupported function: {node.func}")
        
        raise ValueError(f"Unsupported AST node type: {type(node)}")


# ============================================================================
# TOOL REGISTRY
# ============================================================================

class ToolRegistry:
    """Plugin system for external tools with schema validation."""
    
    def __init__(self):
        self._tools: Dict[str, Callable] = {}
        self._schemas: Dict[str, dict] = {}
    
    def register(self, name: str, fn: Callable, schema: Optional[dict] = None):
        """Register a tool with optional schema validation."""
        self._tools[name] = fn
        self._schemas[name] = schema or {}
    
    def call(self, name: str, **kwargs) -> Any:
        """Call a registered tool with validation."""
        if name not in self._tools:
            raise KeyError(f"Unknown tool: {name}")
        
        # Simple validation by required keys
        spec = self._schemas.get(name, {})
        required = set(k for k, v in spec.get("properties", {}).items() 
                      if v.get("required"))
        if not required.issubset(kwargs.keys()):
            missing = required - set(kwargs.keys())
            raise ValueError(f"Missing tool args: {missing}")
        
        return self._tools[name](**kwargs)
    
    def list_tools(self) -> List[str]:
        """List all registered tools."""
        return list(self._tools.keys())


def register_default_tools(registry: ToolRegistry):
    """Register built-in mathematical tools."""
    
    # Polynomial roots (quadratic)
    def poly_roots(a: float, b: float, c: float) -> Dict[str, Any]:
        """Solve ax² + bx + c = 0"""
        discriminant = b**2 - 4*a*c
        if discriminant < 0:
            return {"roots": [], "type": "no_real_roots", "discriminant": discriminant}
        elif discriminant == 0:
            root = -b / (2*a)
            return {"roots": [root], "type": "single_root", "discriminant": discriminant}
        else:
            import math
            sqrt_d = math.sqrt(discriminant)
            r1 = (-b + sqrt_d) / (2*a)
            r2 = (-b - sqrt_d) / (2*a)
            return {"roots": [r1, r2], "type": "two_roots", "discriminant": discriminant}
    
    registry.register("poly_roots", poly_roots, {
        "properties": {
            "a": {"type": "number", "required": True},
            "b": {"type": "number", "required": True},
            "c": {"type": "number", "required": True}
        }
    })
    
    # Unit conversion
    def convert_units(value: float, from_unit: str, to_unit: str) -> Dict[str, Any]:
        """Convert between common units."""
        conversions = {
            ("m", "cm"): 100,
            ("cm", "m"): 0.01,
            ("km", "m"): 1000,
            ("m", "km"): 0.001,
            ("kg", "g"): 1000,
            ("g", "kg"): 0.001,
            ("mph", "m/s"): 0.44704,
            ("m/s", "mph"): 2.23694,
            ("C", "F"): lambda x: x * 9/5 + 32,
            ("F", "C"): lambda x: (x - 32) * 5/9,
        }
        key = (from_unit, to_unit)
        if key not in conversions:
            return {"error": f"Unknown conversion: {from_unit} -> {to_unit}"}
        factor = conversions[key]
        if callable(factor):
            result = factor(value)
        else:
            result = value * factor
        return {"value": result, "unit": to_unit}
    
    registry.register("convert_units", convert_units)


# ============================================================================
# JSON UTILITIES
# ============================================================================

def extract_json_block(text: str) -> Optional[dict]:
    """
    Extract first valid JSON object from text.
    Handles code fences and balanced braces.
    """
    s = text.strip()
    
    # Remove code fences
    if s.startswith("```"):
        # Find first { after code fence
        fence_end = s.find("\n")
        if fence_end != -1:
            s = s[fence_end+1:]
        # Remove closing fence
        if "```" in s:
            s = s[:s.rfind("```")]
    
    # Find first balanced JSON object
    start = s.find("{")
    if start == -1:
        return None
    
    depth = 0
    for i, ch in enumerate(s[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = s[start:i+1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    break
    return None


# ============================================================================
# DAG VALIDATION
# ============================================================================

def topo_order(steps: List[ReasoningStep]) -> List[int]:
    """
    Topological sort with cycle detection for DAG validation.
    Returns execution order of step indices.
    Raises ValueError if cycle detected.
    """
    n = len(steps)
    g = {i: set(s.dependencies) for i, s in enumerate(steps)}
    order = []
    visited = set()
    temp = set()
    
    def dfs(u: int):
        if u in temp:
            raise ValueError(f"Cyclic dependency detected at step {u}")
        if u in visited:
            return
        temp.add(u)
        for v in g[u]:
            if v >= n:
                raise ValueError(f"Invalid dependency {v} in step {u}")
            dfs(v)
        temp.remove(u)
        visited.add(u)
        order.append(u)
    
    for i in range(n):
        dfs(i)
    
    return order


# ============================================================================
# EXPERIMENT TRACKER
# ============================================================================

class JsonlTracker:
    """Lightweight experiment tracker using JSONL format."""
    
    def __init__(self, path: str = "runs/mrf_logs.jsonl"):
        pathlib.Path(os.path.dirname(path) or ".").mkdir(parents=True, exist_ok=True)
        self.path = path
    
    def log(self, event: dict):
        """Log event with timestamp."""
        event["ts"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")


# ============================================================================
# PROMPT TEMPLATES
# ============================================================================

class PromptTemplates:
    """Centralized prompt management with versioning."""
    
    VERSION = "1.1.0"
    
    @staticmethod
    def specification(task: str) -> str:
        return f"""Generate a formal task specification as JSON.
Requirements:
- Decompose the problem into measurable components
- Identify all given information and unknowns
- Define clear success criteria
- Output ONLY valid JSON. No prose.

Task: {task}

JSON Schema:
{{
  "goal": "primary objective",
  "outputs": ["expected formats"],
  "knowns": [{{"name": "variable", "value": number, "unit": "optional"}}],
  "unknowns": ["variable names"],
  "constraints": ["mathematical or logical constraints"],
  "success_criteria": ["testable conditions"]
}}"""
    
    @staticmethod
    def planning(spec: TaskSpecification) -> str:
        return f"""Create a detailed reasoning plan based on specification.

Specification:
{spec.model_dump_json(indent=2)}

Requirements:
- Break down into atomic, verifiable steps
- Each step must have clear input/output
- Include verification tests
- Estimate computational complexity
- Output ONLY valid JSON. No prose.

JSON Schema:
{{
  "steps": [
    {{
      "step_type": "compute|derive|lookup|verify|tool",
      "expression": "mathematical expression if applicable",
      "description": "what this step does",
      "dependencies": [indices of prerequisite steps],
      "expected_output": "description of result",
      "confidence": 0.0-1.0,
      "tool": "tool name if step_type is tool",
      "args": {{}} 
    }}
  ],
  "verification_tests": [
    {{"name": "test name", "expression": "boolean condition"}}
  ],
  "estimated_complexity": "O(n)",
  "plan_confidence": 0.0-1.0
}}"""
    
    @staticmethod
    def reflection(spec: TaskSpecification, 
                  plan: ReasoningPlan, 
                  results: ExecutionResult) -> str:
        return f"""Analyze execution results and identify improvements.

Specification: {spec.model_dump_json()}
Plan confidence: {plan.plan_confidence}
Execution success rate: {results.success_rate}
Errors: {results.errors}

Identify:
1. Logic errors or incorrect assumptions
2. Missing verification steps
3. Precision or numerical issues
4. Alternative approaches

Output JSON:
{{
  "issues": ["list of identified problems"],
  "suggested_fixes": ["specific corrections"],
  "should_retry": true/false,
  "confidence_adjustment": -0.1 to 0.1
}}"""
    
    @staticmethod
    def synthesis(spec: TaskSpecification, 
                 results: ExecutionResult,
                 style_weights: Optional[Dict[str, float]] = None) -> str:
        weights = style_weights or {"precision": 0.4, "depth": 0.3, "conciseness": 0.3}
        return f"""Synthesize final answer from verified results.

Goal: {spec.goal}
Results: {json.dumps(results.step_results, default=str)}
Verification: {json.dumps(results.verification_outcomes)}

Style weights: {weights}

Requirements:
- Present answer clearly and concisely
- Include relevant units and precision
- Do NOT include reasoning process
- Output final answer only"""


# ============================================================================
# ORCHESTRATION CONFIG
# ============================================================================

@dataclass
class OrchestrationConfig:
    """Configuration for reasoning orchestrator."""
    consensus_samples: int = 3
    max_reflection_cycles: int = 2
    verification_threshold: float = 0.9
    temperature_spec: float = 0.1
    temperature_plan: float = 0.2
    temperature_synth: float = 0.3
    timeout_seconds: float = 30.0
    explain_mode: bool = False
    rga_ema_alpha: float = 0.3  # EMA smoothing for RGA weights


# ============================================================================
# REASONING ORCHESTRATOR
# ============================================================================

class ReasoningOrchestrator:
    """
    Main orchestrator for the reasoning pipeline.
    Implements: SPEC → PLAN → EXECUTE → VERIFY → REFLECT → SYNTHESIZE
    """
    
    def __init__(self, 
                 llm: LLMProvider,
                 config: Optional[OrchestrationConfig] = None,
                 tools: Optional[ToolRegistry] = None,
                 tracker: Optional[JsonlTracker] = None):
        self.llm = llm
        self.config = config or OrchestrationConfig()
        self.tools = tools or ToolRegistry()
        self.tracker = tracker
        self.evaluator = SafeMathEvaluator()
        self.metrics: Dict[str, Any] = {}
        self.style_weights = {"precision": 0.4, "depth": 0.3, "conciseness": 0.3}
    
    async def solve(self, task: str, rga_feedback: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Execute full reasoning pipeline for a task.
        
        Args:
            task: Problem description
            rga_feedback: Optional RGA feedback dict with 'vector' for style weights
            
        Returns:
            Dict with 'answer', 'results', 'metrics', and optionally 'explanation'
        """
        start_time = time.time()
        self.metrics = {"stages": {}}
        
        # Apply RGA feedback to style weights (EMA smoothing)
        if rga_feedback and "vector" in rga_feedback:
            for key, value in rga_feedback["vector"].items():
                if key in self.style_weights:
                    old = self.style_weights[key]
                    self.style_weights[key] = (
                        self.config.rga_ema_alpha * value + 
                        (1 - self.config.rga_ema_alpha) * old
                    )
        
        explanation_trace = [] if self.config.explain_mode else None
        
        # Phase 1: Specification
        spec = await self._phase_specification(task)
        if explanation_trace is not None:
            explanation_trace.append(f"SPEC: {spec.goal}")
        
        # Phase 2: Planning with consensus
        plan = await self._phase_planning(spec)
        if explanation_trace is not None:
            explanation_trace.append(f"PLAN: {len(plan.steps)} steps, confidence={plan.plan_confidence:.2f}")
        
        # Phase 3-4: Execute and Verify
        results = await self._phase_execution(spec, plan)
        if explanation_trace is not None:
            explanation_trace.append(f"EXEC: success_rate={results.success_rate:.2f}")
        
        # Phase 5: Reflection (if needed)
        reflection_cycles = 0
        while (results.success_rate < self.config.verification_threshold and 
               reflection_cycles < self.config.max_reflection_cycles):
            reflection = await self._phase_reflection(spec, plan, results)
            if not reflection.get("should_retry", False):
                break
            # Re-plan and re-execute
            plan = await self._phase_planning(spec)
            results = await self._phase_execution(spec, plan)
            reflection_cycles += 1
            if explanation_trace is not None:
                explanation_trace.append(f"REFLECT[{reflection_cycles}]: retry")
        
        # Phase 6: Synthesis
        answer = await self._phase_synthesis(spec, results)
        
        # Record metrics
        self.metrics["total_time_ms"] = (time.time() - start_time) * 1000
        self.metrics["reflection_cycles"] = reflection_cycles
        self.metrics["final_success_rate"] = results.success_rate
        
        # Log to tracker
        if self.tracker:
            self.tracker.log({
                "task_sha": hashlib.sha256(task.encode()).hexdigest()[:16],
                "metrics": self.metrics,
                "model": self.llm.get_model_info(),
            })
        
        result = {
            "answer": answer,
            "results": results.model_dump(),
            "metrics": self.metrics,
        }
        
        if explanation_trace is not None:
            result["explanation"] = "\n".join(explanation_trace)
        
        return result
    
    async def _phase_specification(self, task: str) -> TaskSpecification:
        """Generate formal task specification."""
        t0 = time.time()
        
        prompt = PromptTemplates.specification(task)
        messages = [{"role": "user", "content": prompt}]
        
        response = await self.llm.complete(
            messages, 
            temperature=self.config.temperature_spec
        )
        
        data = extract_json_block(response)
        if not data:
            raise ValueError(f"Failed to parse specification: {response[:200]}")
        
        spec = TaskSpecification(**data)
        
        self.metrics["stages"]["specification_ms"] = (time.time() - t0) * 1000
        return spec
    
    async def _phase_planning(self, spec: TaskSpecification) -> ReasoningPlan:
        """Generate reasoning plan with consensus voting."""
        t0 = time.time()
        
        prompt = PromptTemplates.planning(spec)
        messages = [{"role": "user", "content": prompt}]
        
        # Self-consistency: multiple samples
        plans = []
        for _ in range(self.config.consensus_samples):
            response = await self.llm.complete(
                messages,
                temperature=self.config.temperature_plan
            )
            data = extract_json_block(response)
            if data:
                try:
                    plans.append(ReasoningPlan(**data))
                except Exception:
                    pass
        
        if not plans:
            raise ValueError("Failed to generate any valid plan")
        
        # Select by highest confidence
        best_plan = max(plans, key=lambda p: p.plan_confidence)
        
        # Validate DAG
        try:
            topo_order(best_plan.steps)
        except ValueError as e:
            logger.warning(f"Plan has invalid dependencies: {e}")
        
        self.metrics["stages"]["planning_ms"] = (time.time() - t0) * 1000
        return best_plan
    
    async def _phase_execution(self, spec: TaskSpecification, plan: ReasoningPlan) -> ExecutionResult:
        """Execute plan steps and run verification tests."""
        t0 = time.time()
        
        # Setup variables from knowns
        variables = {}
        for known in spec.knowns:
            if isinstance(known, dict) and "name" in known and "value" in known:
                variables[known["name"]] = known["value"]
        self.evaluator.set_vars(variables)
        
        # Execute steps in topological order
        step_results = {}
        errors = []
        
        try:
            order = topo_order(plan.steps)
        except ValueError:
            order = list(range(len(plan.steps)))
        
        for idx in order:
            step = plan.steps[idx]
            try:
                if step.step_type == "compute" and step.expression:
                    result = self.evaluator.evaluate(step.expression)
                    step_results[idx] = float(result) if isinstance(result, Decimal) else result
                    # Add result to variables for subsequent steps
                    var_name = f"step_{idx}"
                    self.evaluator.vars[var_name] = Decimal(str(result))
                    
                elif step.step_type == "tool" and step.tool:
                    result = self.tools.call(step.tool, **step.args)
                    step_results[idx] = result
                    
                else:
                    step_results[idx] = step.description
                    
            except Exception as e:
                errors.append(f"Step {idx}: {e}")
                step_results[idx] = None
        
        # Run verification tests
        verification_outcomes = []
        passed = 0
        for test in plan.verification_tests:
            try:
                expr = test.get("expression", "True")
                result = self.evaluator.evaluate(expr)
                passed_test = bool(result)
                verification_outcomes.append({
                    "name": test.get("name", "unnamed"),
                    "expression": expr,
                    "passed": passed_test
                })
                if passed_test:
                    passed += 1
            except Exception as e:
                verification_outcomes.append({
                    "name": test.get("name", "unnamed"),
                    "expression": test.get("expression", ""),
                    "passed": False,
                    "error": str(e)
                })
        
        total_tests = len(plan.verification_tests)
        success_rate = passed / total_tests if total_tests > 0 else 1.0
        
        self.metrics["stages"]["execution_ms"] = (time.time() - t0) * 1000
        
        return ExecutionResult(
            step_results=step_results,
            verification_outcomes=verification_outcomes,
            errors=errors,
            success_rate=success_rate
        )
    
    async def _phase_reflection(self, 
                               spec: TaskSpecification, 
                               plan: ReasoningPlan, 
                               results: ExecutionResult) -> Dict[str, Any]:
        """Analyze results and suggest improvements."""
        t0 = time.time()
        
        prompt = PromptTemplates.reflection(spec, plan, results)
        messages = [{"role": "user", "content": prompt}]
        
        response = await self.llm.complete(messages, temperature=0.1)
        
        data = extract_json_block(response)
        if not data:
            data = {"should_retry": False, "issues": []}
        
        self.metrics["stages"]["reflection_ms"] = (time.time() - t0) * 1000
        return data
    
    async def _phase_synthesis(self, spec: TaskSpecification, results: ExecutionResult) -> str:
        """Generate final answer from verified results."""
        t0 = time.time()
        
        prompt = PromptTemplates.synthesis(spec, results, self.style_weights)
        messages = [{"role": "user", "content": prompt}]
        
        response = await self.llm.complete(
            messages, 
            temperature=self.config.temperature_synth
        )
        
        self.metrics["stages"]["synthesis_ms"] = (time.time() - t0) * 1000
        return response.strip()


# ============================================================================
# EXAMPLE LLM PROVIDERS
# ============================================================================

class GeminiFlashProvider(LLMProvider):
    """Google Gemini Flash provider."""
    
    def __init__(self, api_key: str, model: str = "gemini-2.0-flash-exp"):
        self.api_key = api_key
        self.model = model
    
    async def complete(self, 
                      messages: List[Dict[str, str]], 
                      temperature: float = 0.0,
                      max_tokens: Optional[int] = None) -> str:
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            model = genai.GenerativeModel(self.model)
            
            # Convert messages to Gemini format
            prompt = "\n\n".join([
                f"{msg['role'].upper()}: {msg['content']}" 
                for msg in messages
            ])
            
            response = model.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens or 2048
                )
            )
            return response.text
        except ImportError:
            raise ImportError("Install google-generativeai: pip install google-generativeai")
    
    def get_model_info(self) -> Dict[str, Any]:
        return {"provider": "google", "model": self.model}


class OpenAIProvider(LLMProvider):
    """OpenAI/GPT provider."""
    
    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        self.api_key = api_key
        self.model = model
    
    async def complete(self, 
                      messages: List[Dict[str, str]], 
                      temperature: float = 0.0,
                      max_tokens: Optional[int] = None) -> str:
        try:
            from openai import AsyncOpenAI
            client = AsyncOpenAI(api_key=self.api_key)
            
            response = await client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens or 2048
            )
            return response.choices[0].message.content
        except ImportError:
            raise ImportError("Install openai: pip install openai")
    
    def get_model_info(self) -> Dict[str, Any]:
        return {"provider": "openai", "model": self.model}


class AnthropicProvider(LLMProvider):
    """Anthropic Claude provider."""
    
    def __init__(self, api_key: str, model: str = "claude-3-haiku-20240307"):
        self.api_key = api_key
        self.model = model
    
    async def complete(self, 
                      messages: List[Dict[str, str]], 
                      temperature: float = 0.0,
                      max_tokens: Optional[int] = None) -> str:
        try:
            import anthropic
            client = anthropic.AsyncAnthropic(api_key=self.api_key)
            
            response = await client.messages.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens or 2048,
                temperature=temperature
            )
            return response.content[0].text
        except ImportError:
            raise ImportError("Install anthropic: pip install anthropic")
    
    def get_model_info(self) -> Dict[str, Any]:
        return {"provider": "anthropic", "model": self.model}


# ============================================================================
# CONVENIENCE WRAPPER
# ============================================================================

class Flash5Reasoner:
    """
    High-level wrapper for easy usage.
    
    Example:
        solver = Flash5Reasoner(provider=GeminiFlashProvider(api_key="..."))
        result = await solver.solve("Solve 3x² - 12x + 9 = 0")
        print(result["answer"])
    """
    
    def __init__(self, 
                 provider: LLMProvider,
                 consensus_samples: int = 3,
                 max_reflections: int = 2,
                 explain: bool = False,
                 log_path: Optional[str] = None):
        
        config = OrchestrationConfig(
            consensus_samples=consensus_samples,
            max_reflection_cycles=max_reflections,
            explain_mode=explain
        )
        
        tools = ToolRegistry()
        register_default_tools(tools)
        
        tracker = JsonlTracker(log_path) if log_path else None
        
        self.orchestrator = ReasoningOrchestrator(
            llm=provider,
            config=config,
            tools=tools,
            tracker=tracker
        )
    
    async def solve(self, task: str, rga: Optional[Dict] = None) -> Dict[str, Any]:
        """Solve a reasoning task."""
        return await self.orchestrator.solve(task, rga)
    
    def solve_sync(self, task: str, rga: Optional[Dict] = None) -> Dict[str, Any]:
        """Synchronous wrapper for solve()."""
        return asyncio.run(self.solve(task, rga))


# ============================================================================
# DEMO / TESTS
# ============================================================================

async def demo():
    """Demo with mock provider."""
    
    class MockProvider(LLMProvider):
        """Mock provider for testing."""
        
        async def complete(self, messages, temperature=0.0, max_tokens=None):
            # Return mock JSON responses
            content = messages[-1]["content"]
            
            if "specification" in content.lower():
                return json.dumps({
                    "goal": "Solve quadratic equation",
                    "outputs": ["roots"],
                    "knowns": [
                        {"name": "a", "value": 3},
                        {"name": "b", "value": -12},
                        {"name": "c", "value": 9}
                    ],
                    "unknowns": ["x"],
                    "constraints": ["3x² - 12x + 9 = 0"],
                    "success_criteria": ["roots satisfy equation"]
                })
            
            elif "planning" in content.lower():
                return json.dumps({
                    "steps": [
                        {
                            "step_type": "tool",
                            "description": "Use quadratic formula",
                            "dependencies": [],
                            "tool": "poly_roots",
                            "args": {"a": 3, "b": -12, "c": 9}
                        }
                    ],
                    "verification_tests": [
                        {"name": "check_roots", "expression": "True"}
                    ],
                    "estimated_complexity": "O(1)",
                    "plan_confidence": 0.95
                })
            
            elif "reflection" in content.lower():
                return json.dumps({
                    "should_retry": False,
                    "issues": []
                })
            
            else:
                return "x = 1 and x = 3"
        
        def get_model_info(self):
            return {"provider": "mock", "model": "test"}
    
    # Setup
    solver = Flash5Reasoner(
        provider=MockProvider(),
        consensus_samples=1,
        max_reflections=1,
        explain=True,
        log_path="runs/demo.jsonl"
    )
    
    # Solve
    task = "Solve the quadratic equation 3x² - 12x + 9 = 0"
    result = await solver.solve(task)
    
    print("=" * 60)
    print("TASK:", task)
    print("=" * 60)
    print("\nANSWER:", result["answer"])
    
    if "explanation" in result:
        print("\nEXPLANATION:")
        print(result["explanation"])
    
    print("\nMETRICS:")
    for key, value in result["metrics"].items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for k, v in value.items():
                if isinstance(v, float):
                    print(f"    {k}: {v:.2f} ms")
                else:
                    print(f"    {k}: {v}")
        else:
            print(f"  {key}: {value}")
    
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(demo())

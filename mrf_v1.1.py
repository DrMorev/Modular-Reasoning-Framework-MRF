#!/usr/bin/env python3

# -*- coding: utf-8 -*-

“””
Modular Reasoning Framework (MRF) v1.1

Authors: Sergey Morev, multiple LLM team (GPT), (Claude Opus), (Gemini)
License: MIT

Full features:

- Advanced SafeMath (arithmetic + logic + comparisons)
- ToolRegistry (external function calls)
- DAG Validation (cycle detection)
- JsonlTracker (execution logging)
- Robust JSON extraction
  “””

import asyncio
import json
import logging
import ast
import operator as op
import math
import os
import time
from datetime import datetime
from pathlib import Path
from enum import Enum
from typing import List, Dict, Any, Optional, Union, Callable, Set
from decimal import Decimal, getcontext, InvalidOperation

# Pydantic v2

from pydantic import BaseModel, Field, field_validator, ValidationInfo, model_validator

# — CONFIGURATION —

logging.basicConfig(
level=logging.INFO,
format=’%(asctime)s - %(name)s - %(levelname)s - %(message)s’
)
logger = logging.getLogger(“MRF_v1.1”)

# — UTILITIES —

def extract_json(text: str) -> str:
“””
Robust JSON extraction with bracket balancing.
Handles markdown code fences and truncated responses.
“””
text = text.strip()

```
# Handle markdown code fences
if "```json" in text:
    text = text.split("```json")[1].split("```")[0]
elif "```" in text:
    text = text.split("```")[1].split("```")[0]

text = text.strip()

# Find first brace if text doesn't start with one
if not (text.startswith("{") or text.startswith("[")):
    start = text.find("{")
    if start != -1:
        text = text[start:]
        # Find matching closing brace
        stack = 0
        for i, char in enumerate(text):
            if char == "{":
                stack += 1
            elif char == "}":
                stack -= 1
            if stack == 0:
                text = text[:i+1]
                break

return text
```

class JsonlTracker:
“”“Logs reasoning traces to a JSONL file for debugging and analysis.”””

```
def __init__(self, path: str = "runs/trace.jsonl"):
    self.path = Path(path)
    self.path.parent.mkdir(parents=True, exist_ok=True)

def log(self, event_type: str, data: Dict[str, Any]):
    """Log an event with timestamp."""
    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "event": event_type,
        "data": data
    }
    with open(self.path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
```

# — SAFE MATH & LOGIC EVALUATOR —

class SafeMathEvaluator:
“””
AST-based evaluator supporting arithmetic, comparison, and boolean logic.
No eval() - safe against code injection.
“””

```
ALLOWED_OPERATORS = {
    # Arithmetic
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Pow: op.pow,
    ast.Mod: op.mod,
    ast.FloorDiv: op.floordiv,
    ast.USub: op.neg,
    ast.UAdd: op.pos,
    # Comparison
    ast.Eq: op.eq,
    ast.NotEq: op.ne,
    ast.Lt: op.lt,
    ast.LtE: op.le,
    ast.Gt: op.gt,
    ast.GtE: op.ge,
}

ALLOWED_FUNCTIONS = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sqrt": math.sqrt,
    "ceil": math.ceil,
    "floor": math.floor,
    "log": math.log,
    "log10": math.log10,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
}

def __init__(self, precision: int = 10):
    getcontext().prec = precision
    self._context: Dict[str, Any] = {}

def evaluate(self, expression: str, context: Dict[str, Any] = None) -> Any:
    """
    Safely evaluate a mathematical/logical expression.
    
    Args:
        expression: String expression like "x + 5 > 3"
        context: Variable bindings like {"x": 10}
    
    Returns:
        Computed result (number or boolean)
    """
    if context is None:
        context = {}
    
    try:
        self._context = context
        
        # Normalize expression
        expression = expression.replace("true", "True").replace("false", "False")
        expression = expression.replace("^", "**")
        
        node = ast.parse(expression, mode='eval')
        return self._eval_node(node.body)
    except Exception as e:
        logger.error(f"Eval error '{expression}': {e}")
        raise ValueError(f"Evaluation failed: {e}")

def _eval_node(self, node) -> Any:
    """Recursively evaluate AST nodes."""
    
    # Literals
    if isinstance(node, ast.Num):  # Python < 3.8
        return node.n
    if isinstance(node, ast.Str):  # Python < 3.8
        return node.s
    if isinstance(node, ast.Constant):  # Python >= 3.8
        return node.value
    if isinstance(node, ast.NameConstant):  # True/False/None
        return node.value
    
    # Variables
    if isinstance(node, ast.Name):
        if node.id in self._context:
            return self._context[node.id]
        if node.id == 'True':
            return True
        if node.id == 'False':
            return False
        raise ValueError(f"Unknown variable: {node.id}")
    
    # Binary operations
    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in self.ALLOWED_OPERATORS:
            raise ValueError(f"Forbidden operator: {op_type.__name__}")
        left = self._eval_node(node.left)
        right = self._eval_node(node.right)
        return self.ALLOWED_OPERATORS[op_type](left, right)
    
    # Unary operations
    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type == ast.Not:
            return not self._eval_node(node.operand)
        if op_type not in self.ALLOWED_OPERATORS:
            raise ValueError(f"Forbidden unary operator: {op_type.__name__}")
        return self.ALLOWED_OPERATORS[op_type](self._eval_node(node.operand))
    
    # Comparisons
    if isinstance(node, ast.Compare):
        left = self._eval_node(node.left)
        for operation, comparator in zip(node.ops, node.comparators):
            op_type = type(operation)
            if op_type not in self.ALLOWED_OPERATORS:
                raise ValueError(f"Forbidden comparison: {op_type.__name__}")
            right = self._eval_node(comparator)
            if not self.ALLOWED_OPERATORS[op_type](left, right):
                return False
            left = right
        return True
    
    # Boolean operations (and/or)
    if isinstance(node, ast.BoolOp):
        values = [self._eval_node(v) for v in node.values]
        if isinstance(node.op, ast.And):
            return all(values)
        if isinstance(node.op, ast.Or):
            return any(values)
    
    # Function calls
    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
            if func_name in self.ALLOWED_FUNCTIONS:
                args = [self._eval_node(arg) for arg in node.args]
                return self.ALLOWED_FUNCTIONS[func_name](*args)
            raise ValueError(f"Forbidden function: {func_name}")
    
    # Subscript (e.g., step_0[0])
    if isinstance(node, ast.Subscript):
        value = self._eval_node(node.value)
        if isinstance(node.slice, ast.Index):  # Python < 3.9
            index = self._eval_node(node.slice.value)
        else:  # Python >= 3.9
            index = self._eval_node(node.slice)
        return value[index]
    
    raise ValueError(f"Forbidden AST node: {type(node).__name__}")
```

# — TOOL REGISTRY —

class ToolRegistry:
“”“Plugin system for external tool calls.”””

```
def __init__(self):
    self._tools: Dict[str, Callable] = {}
    self._schemas: Dict[str, dict] = {}

def register(self, name: str, func: Callable, schema: dict = None):
    """Register a tool function."""
    self._tools[name] = func
    self._schemas[name] = schema or {}
    logger.info(f"Tool registered: {name}")

def execute(self, tool_name: str, kwargs: Dict[str, Any]) -> Any:
    """Execute a registered tool."""
    if tool_name not in self._tools:
        raise ValueError(f"Tool '{tool_name}' not found. Available: {list(self._tools.keys())}")
    try:
        return self._tools[tool_name](**kwargs)
    except Exception as e:
        logger.error(f"Tool execution error: {e}")
        return {"error": str(e)}

def list_tools(self) -> List[str]:
    """List all registered tools."""
    return list(self._tools.keys())
```

# — SCHEMA WITH DAG VALIDATION —

class StepType(str, Enum):
COMPUTE = “compute”
LOGIC = “logic”
TOOL = “tool”
RETRIEVE = “retrieve”

class Step(BaseModel):
“”“Single step in a reasoning plan.”””
id: str
step_type: StepType
description: str
expression: Optional[str] = None
tool: Optional[str] = None
args: Optional[Dict[str, Any]] = None
dependencies: List[str] = Field(default_factory=list)
confidence: float = Field(default=1.0, ge=0.0, le=1.0)

class Plan(BaseModel):
“”“Reasoning plan with DAG validation.”””
steps: List[Step]
estimated_complexity: str = “O(1)”
plan_confidence: float = Field(default=1.0, ge=0.0, le=1.0)

```
@model_validator(mode='after')
def validate_dag(self) -> 'Plan':
    """Ensures the plan is a valid DAG (no cycles, no forward dependencies)."""
    step_ids = {s.id for s in self.steps}
    
    # 1. Check dependency existence and self-reference
    for step in self.steps:
        for dep in step.dependencies:
            if dep not in step_ids:
                raise ValueError(f"Step {step.id} depends on unknown step {dep}")
            if dep == step.id:
                raise ValueError(f"Step {step.id} depends on itself")
    
    # 2. Check topological order (dependencies must appear before the step)
    seen_ids: Set[str] = set()
    for step in self.steps:
        for dep in step.dependencies:
            if dep not in seen_ids:
                raise ValueError(
                    f"Step {step.id} depends on {dep} which hasn't been executed yet"
                )
        seen_ids.add(step.id)
    
    return self
```

# — ORCHESTRATOR —

class Flash5Reasoner:
“””
Main reasoning orchestrator.
Pipeline: Plan → Execute → Verify → Reflect → Synthesize
“””

```
def __init__(
    self,
    provider: Any,  # Expects async generate(prompt) method
    log_path: str = "runs/mrf_trace.jsonl",
    max_reflections: int = 2
):
    self.provider = provider
    self.tracker = JsonlTracker(log_path)
    self.math = SafeMathEvaluator()
    self.tools = ToolRegistry()
    self.max_reflections = max_reflections
    
    # Register default tools
    self._register_default_tools()

def _register_default_tools(self):
    """Register built-in tools."""
    
    def poly_roots(a: float, b: float, c: float) -> List[float]:
        """Solve ax^2 + bx + c = 0"""
        d = (b ** 2) - (4 * a * c)
        if d < 0:
            return []  # No real roots
        root1 = (-b - math.sqrt(d)) / (2 * a)
        root2 = (-b + math.sqrt(d)) / (2 * a)
        return sorted(list(set([root1, root2])))
    
    def convert_units(value: float, from_unit: str, to_unit: str) -> Dict[str, Any]:
        """Convert between common units."""
        conversions = {
            ("m", "cm"): 100,
            ("cm", "m"): 0.01,
            ("km", "m"): 1000,
            ("m", "km"): 0.001,
            ("kg", "g"): 1000,
            ("g", "kg"): 0.001,
        }
        key = (from_unit, to_unit)
        if key not in conversions:
            return {"error": f"Unknown conversion: {from_unit} -> {to_unit}"}
        return {"value": value * conversions[key], "unit": to_unit}
    
    self.tools.register("poly_roots", poly_roots)
    self.tools.register("convert_units", convert_units)

async def solve(self, task: str) -> Dict[str, Any]:
    """Execute full reasoning pipeline."""
    self.tracker.log("start_task", {"task": task})
    
    # 1. Generate plan
    plan = await self._generate_plan(task)
    self.tracker.log("plan_generated", plan.model_dump())
    
    # 2. Execute plan
    results = await self._execute_plan(plan)
    self.tracker.log("execution_done", results)
    
    # 3. Verify & Reflect loop
    verified = False
    attempts = 0
    
    while not verified and attempts < self.max_reflections:
        verification = await self._verify(task, results)
        self.tracker.log("verification", verification)
        
        if verification.get("is_valid"):
            verified = True
            break
        
        attempts += 1
        logger.warning(f"Verification failed (attempt {attempts}): {verification.get('issues')}")
        
        # Reflection
        critique = await self._reflect(task, results, verification.get("issues", []))
        self.tracker.log("reflection", {"critique": critique, "attempt": attempts})
    
    # 4. Synthesize final answer
    final_answer = await self._synthesize(task, results, verified)
    self.tracker.log("complete", {"answer": final_answer, "verified": verified})
    
    return {
        "task": task,
        "answer": final_answer,
        "trace": results,
        "verified": verified,
        "reflection_cycles": attempts
    }

async def _generate_plan(self, task: str) -> Plan:
    """Generate execution plan from LLM."""
    prompt = f"""TASK: {task}
```

Create a step-by-step execution plan in JSON format.

Available tools: {self.tools.list_tools()}
Step types: ‘compute’ (math expression), ‘tool’ (call function), ‘logic’ (LLM deduction)

Rules:

- IDs must be sequential: step_0, step_1, step_2…
- Dependencies must reference PREVIOUS step IDs only
- Each step should be atomic and verifiable

Output format:
{{
“steps”: [
{{“id”: “step_0”, “step_type”: “tool”, “tool”: “poly_roots”, “args”: {{“a”:1, “b”:2, “c”:1}}, “description”: “Find roots”, “dependencies”: []}}
],
“estimated_complexity”: “O(1)”,
“plan_confidence”: 0.95
}}

Return ONLY valid JSON.”””

```
    raw = await self.provider.generate(prompt)
    return Plan.model_validate_json(extract_json(raw))

async def _execute_plan(self, plan: Plan) -> Dict[str, Any]:
    """Execute plan steps sequentially."""
    results: Dict[str, Any] = {}
    
    for step in plan.steps:
        logger.info(f"Executing {step.id}: {step.description}")
        
        # Build context from previous results
        ctx = dict(results)
        
        try:
            if step.step_type == StepType.COMPUTE and step.expression:
                val = self.math.evaluate(step.expression, context=ctx)
                results[step.id] = val
            
            elif step.step_type == StepType.TOOL and step.tool:
                val = self.tools.execute(step.tool, step.args or {})
                results[step.id] = val
            
            else:
                # Logic/Retrieve via LLM
                prompt = f"Execute step: {step.description}\nContext: {json.dumps(ctx, default=str)}"
                val = await self.provider.generate(prompt)
                results[step.id] = val
                
        except Exception as e:
            logger.error(f"Step {step.id} failed: {e}")
            results[step.id] = {"error": str(e)}
    
    return results

async def _verify(self, task: str, results: Dict[str, Any]) -> Dict[str, Any]:
    """Verify results satisfy task requirements."""
    prompt = f"""Task: {task}
```

Execution Results: {json.dumps(results, default=str)}

Verify if the results correctly solve the task.
Return JSON: {{“is_valid”: true/false, “issues”: [“list of problems if any”]}}”””

```
    raw = await self.provider.generate(prompt)
    try:
        return json.loads(extract_json(raw))
    except json.JSONDecodeError:
        return {"is_valid": False, "issues": ["Failed to parse verification response"]}

async def _reflect(self, task: str, results: Dict[str, Any], issues: List[str]) -> str:
    """Reflect on failures and suggest fixes."""
    prompt = f"""Task: {task}
```

Results: {json.dumps(results, default=str)}
Issues: {issues}

Analyze what went wrong and suggest specific fixes.”””
return await self.provider.generate(prompt)

```
async def _synthesize(self, task: str, results: Dict[str, Any], verified: bool) -> str:
    """Generate final human-readable answer."""
    status = "VERIFIED" if verified else "UNVERIFIED"
    prompt = f"""Task: {task}
```

Results: {json.dumps(results, default=str)}
Status: {status}

Synthesize a clear, concise final answer. Include the key result and any relevant details.”””
return await self.provider.generate(prompt)

# — MOCK PROVIDER FOR TESTING —

class MockProvider:
“”“Mock LLM provider for testing without API calls.”””

```
async def generate(self, prompt: str) -> str:
    # Planning response
    if "execution plan" in prompt.lower():
        return json.dumps({
            "steps": [
                {
                    "id": "step_0",
                    "step_type": "tool",
                    "tool": "poly_roots",
                    "args": {"a": 3, "b": -12, "c": 9},
                    "description": "Solve quadratic equation 3x^2 - 12x + 9 = 0",
                    "dependencies": []
                },
                {
                    "id": "step_1",
                    "step_type": "compute",
                    "expression": "step_0[0] > 0",
                    "description": "Check if first root is positive",
                    "dependencies": ["step_0"]
                }
            ],
            "estimated_complexity": "O(1)",
            "plan_confidence": 0.95
        })
    
    # Verification response
    if "is_valid" in prompt.lower() or "verify" in prompt.lower():
        return '{"is_valid": true, "issues": []}'
    
    # Default synthesis
    return "The roots of the equation 3x² - 12x + 9 = 0 are x = 1.0 and x = 3.0"
```

# — ENTRY POINT —

async def main():
“”“Demo run with mock provider.”””
print(”=” * 60)
print(“MRF v1.1 Demo”)
print(”=” * 60)

```
agent = Flash5Reasoner(provider=MockProvider())
result = await agent.solve("Solve 3x^2 - 12x + 9 = 0")

print("\nResult:")
print(json.dumps(result, indent=2, default=str))
print("=" * 60)
```

if **name** == “**main**”:
asyncio.run(main())

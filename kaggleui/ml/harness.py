"""Deterministic evaluation harness: fixed cases, offline graders, no network."""

from __future__ import annotations

import json
import re
import time
from typing import Any, Literal

from pydantic import BaseModel, Field

Grader = Literal["exact", "contains_any", "regex", "json_valid", "tool_call_name"]

class EvalCase(BaseModel):
    id: str
    messages: list[dict[str, str]]
    grader: Grader
    expect: list[str] = Field(default_factory=list)
    required_keys: list[str] = Field(default_factory=list)
    gold: str | None = None

class HarnessProfile(BaseModel):
    name: str
    version: str
    description: str = ""
    cases: list[EvalCase]

class CaseResult(BaseModel):
    id: str
    passed: bool
    detail: str = ""
    answer: str = ""

class EvalReport(BaseModel):
    profile: str
    version: str
    score: float
    passed: int
    total: int
    cases: list[CaseResult]
    at: float = Field(default_factory=time.time)

def _content(response: Any) -> tuple[str, list[dict[str, Any]]]:
    """Pull assistant text and tool calls from an OpenAI-shaped completion."""
    try:
        choice = (response.get("choices") or [{}])[0]
    except AttributeError:
        return str(response), []
    message = choice.get("message") or {}
    return str(message.get("content") or ""), list(message.get("tool_calls") or [])

def grade(case: EvalCase, answer: str, tool_calls: list[dict[str, Any]] | None = None) -> tuple[bool, str]:
    """Return (passed, human-readable reason). Every grader is deterministic."""
    tool_calls = tool_calls or []
    if case.grader == "exact":
        expected = case.expect[0] if case.expect else (case.gold or "")
        ok = answer.strip() == expected.strip()
        return ok, "exact match" if ok else f"expected {expected!r}"
    if case.grader == "contains_any":
        ok = any(item.lower() in answer.lower() for item in case.expect)
        return ok, "matched" if ok else f"missing any of {case.expect}"
    if case.grader == "regex":
        pattern = case.expect[0] if case.expect else ""
        try:
            ok = bool(re.search(pattern, answer))
        except re.error as exc:
            return False, f"bad pattern: {exc}"
        return ok, "matched" if ok else f"no match for {pattern}"
    if case.grader == "json_valid":
        try:
            data = json.loads(answer.strip())
        except ValueError:
            return False, "not valid JSON"
        if not isinstance(data, dict):
            return False, "JSON is not an object"
        missing = [key for key in case.required_keys if key not in data]
        return (not missing), "valid" if not missing else f"missing keys {missing}"
    if case.grader == "tool_call_name":
        names = [call.get("function", {}).get("name") for call in tool_calls]
        ok = any(name in case.expect for name in names)
        return ok, "called" if ok else f"no call to {case.expect}; got {names}"
    return False, f"unknown grader {case.grader}"

async def evaluate(dep: Any, profile: HarnessProfile, *, max_tokens: int = 256) -> EvalReport:
    """Run every case through the deployment's backend directly (no auth, no HTTP)."""
    if dep.backend is None:
        raise RuntimeError(f"{dep.served_name} is not ready")
    results: list[CaseResult] = []
    for case in profile.cases:
        body: dict[str, Any] = {"model": dep.served_name, "messages": case.messages, "temperature": 0,
                                "seed": 0, "max_tokens": max_tokens}
        if case.grader == "tool_call_name":
            body["tools"] = [{"type": "function", "function": {"name": name, "description": name,
                                                               "parameters": {"type": "object", "properties": {}}}}
                             for name in case.expect]
        response = await dep.backend.chat(body)
        answer, calls = _content(response)
        passed, detail = grade(case, answer, calls)
        results.append(CaseResult(id=case.id, passed=passed, detail=detail, answer=answer[:500]))
    passed = sum(1 for r in results if r.passed)
    return EvalReport(profile=profile.name, version=profile.version, score=passed / len(results) if results else 0.0,
                      passed=passed, total=len(results), cases=results)

def profiles() -> list[HarnessProfile]:
    from .harness_data import PROFILES

    return list(PROFILES.values())

def find_profile(name: str) -> HarnessProfile:
    from .harness_data import PROFILES

    try:
        return PROFILES[name]
    except KeyError:
        raise ValueError(f"unknown harness profile {name!r}") from None

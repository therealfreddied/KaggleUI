"""Dataset ingestion for training: parse, normalize, and validate before a GPU job is queued.

Every error carries the 1-based sample index so the UI can point at the offending row.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

ROLES = ("system", "user", "assistant")
HARNESS_FORMAT = "jsonl"

class DatasetReport(BaseModel):
    ok: bool = False
    samples: int = 0
    detected_format: str = ""
    token_estimate: int = 0
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

def _role(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    role = value.strip().lower()
    if role in ("human", "prompter", "gpt"):
        return "user" if role == "human" else "assistant"
    if role in ("bot", "gpt_value"):
        return "assistant"
    return role if role in ROLES else None

def _text(value: Any) -> str | None:
    return value if isinstance(value, str) else None

def _from_messages(raw: Any) -> tuple[list[dict[str, str]] | None, str | None]:
    if not isinstance(raw, list):
        return None, "messages must be a list"
    turns: list[dict[str, str]] = []
    for turn in raw:
        if not isinstance(turn, dict):
            return None, "message must be an object"
        role = _role(turn.get("role"))
        if role is None:
            return None, f"unknown role {turn.get('role')!r}"
        content = _text(turn.get("content"))
        if content is None:
            return None, "message content must be a string"
        if not content.strip():
            return None, "message content is empty"
        turns.append({"role": role, "content": content})
    return turns, None

def _from_sharegpt(raw: Any) -> tuple[list[dict[str, str]] | None, str | None]:
    if not isinstance(raw, list):
        return None, "conversations must be a list"
    turns: list[dict[str, str]] = []
    for turn in raw:
        if not isinstance(turn, dict):
            return None, "conversation entry must be an object"
        role = _role(turn.get("from", turn.get("role")))
        content = _text(turn.get("value", turn.get("content")))
        if role is None or content is None:
            return None, "conversation entry needs from/value strings"
        turns.append({"role": role, "content": content})
    return turns, None

def _from_alpaca(row: dict[str, Any]) -> list[dict[str, str]]:
    prompt = _text(row.get("instruction")) or ""
    extra = _text(row.get("input")) or ""
    answer = _text(row.get("output")) or _text(row.get("response")) or ""
    turns = [{"role": "user", "content": (prompt + ("\n" + extra if extra else "")).strip()}]
    if answer.strip():
        turns.append({"role": "assistant", "content": answer})
    return turns

def _normalize(row: Any) -> tuple[list[dict[str, str]] | None, str, str | None]:
    """Return the normalized turns, the detected shape, and an error if the row is unusable."""
    if isinstance(row, dict) and "messages" in row:
        turns, err = _from_messages(row["messages"])
        return turns, "messages", err
    if isinstance(row, dict) and "conversations" in row:
        turns, err = _from_sharegpt(row["conversations"])
        return turns, "sharegpt", err
    if isinstance(row, dict) and ("instruction" in row or "output" in row):
        return _from_alpaca(row), "alpaca", None
    if isinstance(row, dict) and ("prompt" in row and ("response" in row or "completion" in row)):
        answer = _text(row.get("response")) or _text(row.get("completion")) or ""
        return [{"role": "user", "content": _text(row.get("prompt")) or ""},
                {"role": "assistant", "content": answer}], "prompt-response", None
    return None, "", f"unrecognized sample shape: {sorted(row) if isinstance(row, dict) else type(row).__name__}"

def _check_turns(turns: list[dict[str, str]]) -> str | None:
    if not turns:
        return "empty conversation"
    if turns[0]["role"] == "assistant":
        return "conversation starts with an assistant turn"
    if not any(t["role"] == "assistant" for t in turns):
        return "conversation has no assistant turn"
    return None

def _rows(path: Path, fmt: str | None) -> tuple[list[Any], str, str | None]:
    """Load raw rows from disk; returns (rows, detected format, fatal error)."""
    fmt = (fmt or path.suffix.lstrip(".")).lower()
    try:
        text = path.read_text()
    except OSError as exc:
        return [], fmt, f"cannot read {path.name}: {exc}"
    if fmt == "jsonl":
        rows = []
        for line in text.splitlines():
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except ValueError as exc:
                return [], fmt, f"invalid JSON line: {exc}"
        return rows, "jsonl", None
    if fmt == "json":
        try:
            data = json.loads(text)
        except ValueError as exc:
            return [], fmt, f"invalid JSON: {exc}"
        if isinstance(data, dict):
            for key in ("samples", "data", "rows", "conversations"):
                if isinstance(data.get(key), list):
                    return data[key], "json", None
            return [data], "json", None
        if isinstance(data, list):
            return data, "json", None
        return [], fmt, "JSON dataset must be a list or an object with a list field"
    if fmt == "csv":
        try:
            rows = list(csv.DictReader(text.splitlines()))
        except csv.Error as exc:
            return [], fmt, f"invalid CSV: {exc}"
        return rows, "csv", None
    return [], fmt, f"unsupported format {fmt!r} (use jsonl, json or csv)"

def _row_turns(row: Any, fmt: str) -> tuple[list[dict[str, str]] | None, str, str | None]:
    if fmt == "csv" and isinstance(row, dict):
        if row.get("messages"):
            try:
                parsed = json.loads(row["messages"])
            except ValueError:
                return None, "messages", "CSV messages column is not valid JSON"
            if isinstance(parsed, dict) and "messages" in parsed:
                parsed = parsed["messages"]
            turns, err = _from_messages(parsed)
            return turns, "messages", err
        return _normalize({k: v for k, v in row.items() if v not in (None, "")})
    return _normalize(row)

def validate(path: Path | str, fmt: str | None = None, *, limit: int | None = None) -> DatasetReport:
    """Parse and check a dataset without launching training."""
    path = Path(path)
    rows, detected, fatal = _rows(path, fmt)
    report = DatasetReport(detected_format=detected)
    if fatal:
        report.errors.append(fatal)
        return report
    if limit is not None:
        rows = rows[:limit]
    seen: set[str] = set()
    samples = 0
    tokens = 0
    duplicates = 0
    for index, row in enumerate(rows, start=1):
        turns, shape, error = _row_turns(row, detected)
        if error is None and turns is not None:
            error = _check_turns(turns)
        if error is not None:
            if len(report.errors) < 50:
                report.errors.append(f"sample {index}: {error}")
            continue
        assert turns is not None
        key = json.dumps(turns, ensure_ascii=False, sort_keys=True)
        if key in seen:
            duplicates += 1
            continue
        seen.add(key)
        samples += 1
        tokens += sum(len(t["content"]) // 4 for t in turns)
        if shape != "messages":
            report.detected_format = f"{detected}:{shape}" if detected != shape else shape
    report.samples = samples
    report.token_estimate = tokens
    if duplicates:
        report.warnings.append(f"{duplicates} duplicate sample{'s' if duplicates != 1 else ''} dropped")
    if samples == 0 and not report.errors:
        report.errors.append("dataset contains no samples")
    report.ok = samples > 0 and not report.errors
    return report

def build_harness_dataset(profile: Any, answers: dict[str, str] | None = None) -> list[dict[str, Any]]:
    """Turn harness cases plus gold answers into chat samples (only cases with answers are kept)."""
    answers = answers or {}
    out: list[dict[str, Any]] = []
    for case in profile.cases:
        answer = answers.get(case.id)
        if answer is None:
            answer = getattr(case, "gold", None)
        if not answer:
            continue
        out.append({"messages": [*case.messages, {"role": "assistant", "content": str(answer)}]})
    return out

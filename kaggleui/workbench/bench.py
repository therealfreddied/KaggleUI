"""Latency and throughput benchmarks against a running deployment.

Runs in-process against ``manager.resolve`` so no auth or HTTP hop is involved; the CLI is the only
entry point and the result is stored as an ``export`` artifact for later comparison.
"""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..schemas import ArtifactManifest
from .registry import RegistryError

if TYPE_CHECKING:
    from ..deploy.manager import Deployment
    from ..server.state import AppState

FILLER = "The quick brown fox jumps over the lazy dog and keeps running through the green field. "

def prompt_for(tokens: int) -> str:
    """Roughly ``tokens`` tokens of filler (four characters per token)."""
    return (FILLER * (tokens // 16 + 1))[: max(1, tokens * 4)]

async def _one_run(dep: Deployment, body: dict[str, Any], runs: int) -> dict[str, Any]:
    assert dep.backend is not None
    samples = []
    for _ in range(runs):
        started = time.perf_counter()
        first: float | None = None
        pieces = 0
        text: list[str] = []
        async for chunk in dep.backend.chat_stream(body):
            if first is None:
                first = time.perf_counter() - started
            for choice in chunk.get("choices") or []:
                delta = choice.get("delta") or {}
                content = delta.get("content")
                if content:
                    pieces += 1
                    text.append(content)
        total = time.perf_counter() - started
        output = len("".join(text).split())
        samples.append({"ttft_s": first if first is not None else total, "total_s": total,
                        "output_tokens": output,
                        "tokens_per_s": output / total if total > 0 else 0.0})
    return {
        "ttft_s": statistics.median(s["ttft_s"] for s in samples),
        "total_s": statistics.median(s["total_s"] for s in samples),
        "output_tokens": samples[-1]["output_tokens"],
        "tokens_per_s": statistics.median(s["tokens_per_s"] for s in samples),
        "runs": samples,
    }

async def bench(state: AppState, model: str, *, ctxs: tuple[int, ...] = (2048, 4096),
                prompt_tokens: tuple[int, ...] = (128, 512), runs: int = 3) -> dict[str, Any]:
    dep = state.manager.resolve(model, "text")
    if dep is None:
        for candidate in state.manager.deployments.values():
            if candidate.spec.modality == "text" and model in (candidate.served_name, candidate.id,
                                                               candidate.spec.model.id):
                dep = candidate
                break
    if dep is None:
        raise RuntimeError(f"No text deployment answers to '{model}'")
    if dep.status != "ready":
        await state.manager.ensure_ready(dep)
    results = []
    for ctx in ctxs:
        for size in prompt_tokens:
            body = {"model": dep.served_name, "messages": [{"role": "user", "content": prompt_for(size)}],
                    "max_tokens": 64, "temperature": 0, "stream": True,
                    "options": {"ctx": ctx}}
            row = await _one_run(dep, body, runs)
            results.append({"ctx": ctx, "prompt_tokens": size, **row})
    return {"schema_version": 1, "model": dep.served_name, "model_rev": dep.spec.model.id,
            "engine": dep.spec.model.engine, "runs_per_point": runs, "created_at": time.time(),
            "results": results}

def save(state: AppState, report: dict[str, Any], *, out: Path | str | None = None) -> dict[str, Path]:
    """Write the report to ``out`` and store it as an ``export`` artifact."""
    paths: dict[str, Path] = {}
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if out is not None:
        out_path = Path(out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(payload)
        paths["file"] = out_path
    artifact_id = f"bench-{report['model'].replace('/', '-')}"
    version = time.strftime("b%Y%m%d-%H%M%S", time.gmtime(report["created_at"]))
    scratch = state.paths.cache / "bench" / version
    scratch.mkdir(parents=True, exist_ok=True)
    report_path = scratch / "bench.json"
    report_path.write_text(payload)
    manifest = ArtifactManifest(
        artifact_id=artifact_id, version=version, kind="export", created_at=report["created_at"],
        recipe={"bench": report}, backend={"engine": report["engine"]},
        files=[], provenance={"tool": "kaggleui bench", "model_rev": report["model_rev"]},
    )
    try:
        stored = state.artifacts.add(manifest, [report_path], overwrite=True)
        paths["artifact_dir"] = state.artifacts.dir_for(stored.artifact_id, stored.version)
    except RegistryError as exc:  # a benchmark is not worth failing over storage
        paths["artifact_error"] = str(exc)  # type: ignore[assignment]
    return paths

def format_table(report: dict[str, Any]) -> str:
    head = f"{'ctx':>6} {'prompt':>7} {'ttft s':>7} {'total s':>8} {'tok/s':>7} {'out':>4}"
    lines = [head, "-" * len(head)]
    for row in report["results"]:
        lines.append(f"{row['ctx']:>6} {row['prompt_tokens']:>7} {row['ttft_s']:>7.3f} "
                     f"{row['total_s']:>8.3f} {row['tokens_per_s']:>7.1f} {row['output_tokens']:>4}")
    return "\n".join(lines)

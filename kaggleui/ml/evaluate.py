"""Run a harness profile against a deployment and keep the report as an artifact."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from ..schemas import ArtifactManifest
from .harness import EvalReport, HarnessProfile, evaluate, find_profile

if TYPE_CHECKING:
    from ..deploy.manager import Deployment
    from ..server.state import AppState

async def run(state: AppState, dep: Deployment, profile: str | HarnessProfile, *, job_id: str = "") -> dict:
    harness = find_profile(profile) if isinstance(profile, str) else profile
    report: EvalReport = await evaluate(dep, harness)
    out_dir = state.paths.outputs / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    name = job_id or f"{dep.id}-{harness.name}-{int(report.at)}"
    path = out_dir / f"{name}.json"
    payload = report.model_dump()
    path.write_text(json.dumps(payload, indent=2))
    version = f"e{int(report.at)}"
    manifest = ArtifactManifest(
        artifact_id=f"eval-{dep.served_name.replace('/', '-')}-{harness.name}", version=version,
        kind="export", created_at=report.at,
        eval_summary={"profile": report.profile, "version": report.version, "score": report.score,
                      "passed": report.passed, "total": report.total},
        files=[], provenance={"deployment": dep.id, "report": str(path)},
    )
    stored = state.artifacts.add(manifest, [path], overwrite=True)
    payload.update(artifact=f"{stored.artifact_id}@{stored.version}", report_path=str(path))
    return payload

def table(report: dict) -> str:
    head = f"{'case':<20} {'pass':<5} detail"
    lines = [head, "-" * len(head)]
    for case in report["cases"]:
        lines.append(f"{case['id']:<20} {'yes' if case['passed'] else 'no':<5} {case['detail']}")
    lines.append(f"score: {report['passed']}/{report['total']} ({report['score']:.1%})")
    return "\n".join(lines)

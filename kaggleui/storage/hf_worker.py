"""HF Hub worker: one short-lived process per transfer, so blocking hub calls never touch the server loop.

Usage: ``python -m kaggleui.storage.hf_worker <spec.json>`` where the spec is
``{"action": "upload", "repo": "owner/name", "repo_type": "model", "token": "...", "remote": "x",
   "local": "/path", "confirm": true}``. Progress is printed as ``KUI_EVENT {"fraction": 0.5}`` lines;
the final line is the JSON result, and a non-zero exit means failure.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

EVENT_TAG = "KUI_EVENT"

def _emit(payload: dict) -> None:
    print(f"{EVENT_TAG} {json.dumps(payload)}", flush=True)

def _api(spec: dict):
    from huggingface_hub import HfApi

    return HfApi(token=spec.get("token") or os.environ.get("HF_TOKEN") or None)

def _repo_type(spec: dict) -> str:
    value = spec.get("repo_type") or "model"
    return "dataset" if value == "dataset" else "model"

def run(spec: dict) -> dict:
    action = spec["action"]
    repo = spec["repo"]
    repo_type = _repo_type(spec)
    api = _api(spec)
    if action == "test":
        info = api.repo_info(repo, repo_type=repo_type)
        return {"ok": True, "repo": info.id, "private": bool(info.private), "repo_type": repo_type}
    if action == "list":
        prefix = (spec.get("prefix") or "").strip("/")
        rows = api.list_repo_tree(repo, path_in_repo=prefix or None, repo_type=repo_type, recursive=False)
        entries = []
        for row in rows:
            path = getattr(row, "path", "")
            size = getattr(row, "size", None)
            entries.append({"path": path, "size": size, "is_dir": getattr(row, "type", "file") == "directory"})
        return {"entries": entries}
    if action == "stat":
        remote = spec["remote"]
        try:
            info = api.get_paths_info([remote], repo_id=repo, repo_type=repo_type)[0]
        except Exception:
            return {"entry": None}
        return {"entry": {"path": remote, "size": getattr(info, "size", None), "is_dir": False}}
    if action == "upload":
        remote = spec["remote"]
        local = Path(spec["local"])
        _emit({"fraction": 0.05})
        api.upload_file(path_or_fileobj=str(local), path_in_repo=remote, repo_id=repo, repo_type=repo_type,
                        commit_message=f"kaggleui: add {remote}")
        _emit({"fraction": 1.0})
        return {"ok": True}
    if action == "download":
        remote = spec["remote"]
        local = Path(spec["local"])
        local.parent.mkdir(parents=True, exist_ok=True)
        from huggingface_hub import hf_hub_download

        got = hf_hub_download(repo_id=repo, filename=remote, repo_type=repo_type,
                              token=spec.get("token") or os.environ.get("HF_TOKEN") or None,
                              local_dir=str(local.parent))
        got_path = Path(got)
        if got_path.resolve() != local.resolve():
            local.write_bytes(got_path.read_bytes())
        _emit({"fraction": 1.0})
        return {"ok": True, "path": str(local)}
    if action == "delete":
        if not spec.get("confirm"):
            raise ValueError("delete requires confirm=True")
        api.delete_file(path_in_repo=spec["remote"], repo_id=repo, repo_type=repo_type,
                        commit_message=f"kaggleui: delete {spec['remote']}")
        return {"ok": True}
    raise ValueError(f"unknown action {action!r}")

def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        print("usage: python -m kaggleui.storage.hf_worker <spec.json>", file=sys.stderr)
        return 2
    spec = json.loads(Path(args[0]).read_text())
    try:
        result = run(spec)
    except Exception as exc:  # surfaced through the job log
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 1
    print(json.dumps(result), flush=True)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

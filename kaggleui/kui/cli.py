"""``kui``: local lifecycle and portability client.

Exit codes are stable: 2 usage, 3 profile/config, 4 unreachable, 5 rejected key, 6 remote error,
7 conflict, 8 Kaggle CLI. Human output is terse; ``--json`` emits the server value unchanged.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from .client import KuiError, ManagementClient
from .config import ProfileError, ProfileStore, config_dir
from .kaggle_adapter import KaggleError, KaggleNotebooks


def _read_stdin() -> str:
    value = sys.stdin.read().strip()
    if not value:
        raise ProfileError("stdin was empty")
    return value

def _store(args) -> ProfileStore:
    return ProfileStore(Path(args.config).expanduser() if getattr(args, "config", None) else None)

def _client(args, store: ProfileStore) -> ManagementClient:
    explicit_url = getattr(args, "url", None)
    explicit_token = getattr(args, "token", None)
    # A complete pair of global flags is self-contained; requiring a saved profile here made
    # `kui status --url … --token …` fail on a machine that deliberately stores no credentials.
    instance: dict = {}
    if not (explicit_url and explicit_token):
        try:
            resolved = store.resolve(getattr(args, "profile", None))
        except ProfileError:
            # An explicit or environment URL still identifies the target; the token may come from
            # the environment or the local state file. Only a bare `kui status` needs a profile.
            if not (explicit_url or os.environ.get("KAGGLEUI_URL")):
                raise
            resolved = {}
        instance = resolved.get("instance") or {}
    url = (explicit_url or instance.get("url") or os.environ.get("KAGGLEUI_URL") or "")
    token = (explicit_token or os.environ.get("KAGGLEUI_MGMT_TOKEN")
             or instance.get("token") or "")
    if not token:
        token = os.environ.get("KAGGLEUI_ADMIN_TOKEN") or ""
    if not token and url and _is_local(url):
        token = _local_admin_token()
    return ManagementClient(url, token)

def _is_local(url: str) -> bool:
    return any(host in url for host in ("127.0.0.1", "localhost", "::1"))

def _local_admin_token() -> str:
    home = Path(os.environ.get("KAGGLEUI_HOME") or "/tmp/kaggleui")
    for candidate in (home / "state.json", Path("/kaggle/working/kaggleui/state.json")):
        try:
            data = json.loads(candidate.read_text())
        except (OSError, ValueError):
            continue
        token = data.get("admin_token")
        if isinstance(token, str) and token:
            return token
    return ""

def _print(args, value: Any) -> None:
    if getattr(args, "json", False):
        print(json.dumps(value, indent=1, sort_keys=True, default=str))
        return
    if isinstance(value, str):
        print(value)
    elif isinstance(value, list):
        for row in value:
            print(row if isinstance(row, str) else json.dumps(row, default=str))
    else:
        print(json.dumps(value, indent=1, sort_keys=True, default=str))

def _table(rows: list[dict], columns: list[tuple[str, str]]) -> list[str]:
    if not rows:
        return ["(none)"]
    widths = [len(label) for _, label in columns]
    for row in rows:
        for i, (key, _) in enumerate(columns):
            widths[i] = max(widths[i], len(str(row.get(key, ""))))
    out = ["  ".join(label.ljust(widths[i]) for i, (_, label) in enumerate(columns))]
    for row in rows:
        out.append("  ".join(str(row.get(key, "")).ljust(widths[i])
                             for i, (key, _) in enumerate(columns)))
    return out

# -- profile commands --------------------------------------------------------------------------
def cmd_profile(args, store: ProfileStore) -> int:
    if args.action == "list":
        active = store.active_name()
        if not store.profile_names():
            print("(no profiles; run `kui profile add NAME --url URL --token-stdin`)")
        for name in store.profile_names():
            mark = "*" if name == active else " "
            profile = store.data["profiles"][name]
            linked = ", ".join(f"{k}={v}" for k, v in profile.items() if v) or "(empty)"
            print(f"{mark} {name:<16} {linked}")
        return 0
    if args.action == "use":
        store.set_active(args.name)
        print(f"active profile: {args.name}")
        return 0
    if args.action == "show":
        resolved = store.resolve(args.name)
        public = store.public()
        resolved["instances"] = public["instances"]
        resolved["credentials"] = public["credentials"]
        if resolved.get("instance", {}).get("token"):
            resolved["instance"]["token"] = "***"
        if resolved.get("credential", {}).get("key"):
            resolved["credential"]["key"] = "***"
        _print(args, resolved)
        return 0
    if args.action == "add":
        return _profile_add(args, store)
    raise ProfileError("unknown profile action")

def _profile_add(args, store: ProfileStore) -> int:
    name = args.name
    if args.url:
        values = {"url": args.url}
        token = args.token or (os.environ.get("KAGGLEUI_MGMT_TOKEN"))
        if args.token_stdin:
            token = _read_stdin()
        if token:
            values["token"] = token
        store.set_section("instances", name, values)
    if args.repo:
        store.set_section("workspaces", name, {"repo": args.repo, "branch": args.branch,
                                               "path": args.path, "token_secret": args.token_secret})
    if args.slug:
        store.set_section("notebooks", name, {"slug": args.slug, "source": args.source or "",
                                              "accelerator": args.accelerator,
                                              "timeout_s": args.timeout})
    if args.kaggle_username:
        key = _read_stdin() if args.key_stdin else None
        store.set_section("credentials", name, {"username": args.kaggle_username,
                                                "key": key, "key_file": args.key_file})
    profile = store.add_profile(name, instance=name if args.url else None,
                                workspace=name if args.repo else None,
                                notebook=name if args.slug else None,
                                credential=name if args.kaggle_username else None)
    if args.use:
        store.set_active(name)
    _print(args, {"profile": name, "links": profile,
                  "warning": "Kaggle allows one active account per person; profiles do not rotate accounts."})
    return 0

def cmd_instance(args, store: ProfileStore) -> int:
    if args.action == "list":
        for name, values in store.public()["instances"].items():
            print(f"{name:<16} {values.get('url', '')} token={'stored' if values.get('token') else 'none'}")
        return 0
    if args.action == "add":
        values = {"url": args.url}
        token = _read_stdin() if args.token_stdin else args.token
        if token:
            values["token"] = token
        store.set_section("instances", args.name, values)
        print(f"instance {args.name} saved; token is stored owner-only in {store.path}")
        return 0
    if args.action == "show":
        values = dict(store.data["instances"].get(args.name) or {})
        if not values:
            raise ProfileError(f"no instance named {args.name}")
        values["token"] = "***" if values.get("token") else ""
        _print(args, values)
        return 0
    raise ProfileError("unknown instance action")

def cmd_workspace_config(args, store: ProfileStore) -> int:
    values = {"repo": args.repo, "branch": args.branch, "path": args.path, "token_secret": args.token_secret}
    store.set_section("workspaces", args.name, values)
    print(f"workspace {args.name} saved in {config_dir()}")
    return 0

def cmd_notebook_config(args, store: ProfileStore) -> int:
    values = {"slug": args.slug, "source": args.source, "accelerator": args.accelerator,
              "timeout_s": args.timeout}
    store.set_section("notebooks", args.name, values)
    print(f"notebook {args.name} saved")
    return 0

# -- remote commands ---------------------------------------------------------------------------
def cmd_status(args, store: ProfileStore) -> int:
    info = _client(args, store).instance()
    if args.json:
        _print(args, info)
        return 0
    print(f"KaggleUI {info.get('version')}  {info.get('url')}")
    print(f"workspace {info.get('workspace_id')} ({info.get('workspace_name')})  "
          f"instance {info.get('instance_id')}  host {info.get('host')}")
    print(f"deployments {len(info.get('deployments') or [])}  uptime {int(info.get('uptime_s') or 0)}s")
    for line in _table(info.get("deployments") or [], [("id", "ID"), ("name", "NAME"),
                                                       ("modality", "MODALITY"), ("status", "STATUS")]):
        print(line)
    return 0

def cmd_deployments(args, store: ProfileStore) -> int:
    client = _client(args, store)
    if args.action == "list":
        rows = client.deployments()
        if args.json:
            _print(args, rows)
            return 0
        for line in _table(rows, [("id", "ID"), ("name", "NAME"), ("modality", "MODALITY"),
                                  ("status", "STATUS"), ("message", "MESSAGE")]):
            print(line)
        return 0
    if args.action in {"start", "stop"}:
        result = client.post(f"/_kui/api/workbench/deployments/{args.id}/{args.action}")
        _print(args, result)
        return 0
    raise ProfileError("unknown deployments action")

def cmd_jobs(args, store: ProfileStore) -> int:
    client = _client(args, store)
    if args.action == "list":
        rows = client.jobs(limit=args.limit)
        if args.json:
            _print(args, rows)
            return 0
        for line in _table(rows, [("id", "ID"), ("kind", "KIND"), ("state", "STATE"),
                                  ("message", "MESSAGE")]):
            print(line)
        return 0
    if args.action == "show":
        _print(args, client.get(f"/_kui/api/workbench/jobs/{args.id}"))
        return 0
    if args.action == "log":
        _print(args, client.get(f"/_kui/api/workbench/jobs/{args.id}/log", offset=args.offset,
                                limit=args.limit))
        return 0
    raise ProfileError("unknown jobs action")

def cmd_storage(args, store: ProfileStore) -> int:
    result = _client(args, store).get("/_kui/api/workbench/storage/cli")
    _print(args, result)
    return 0

def cmd_workspace(args, store: ProfileStore) -> int:
    client = _client(args, store)
    if args.action == "status":
        _print(args, client.get("/_kui/api/workspace/status"))
        return 0
    if args.action == "preview":
        _print(args, client.get("/_kui/api/workspace/preview"))
        return 0
    if args.action == "export":
        document = client.get("/_kui/api/workspace/export")
        if args.out:
            Path(args.out).expanduser().write_text(json.dumps(document, indent=1))
            print(f"wrote {args.out}")
        else:
            _print(args, document)
        return 0
    if args.action == "import":
        raw = json.loads(Path(args.file).expanduser().read_text())
        document = raw.get("document") if isinstance(raw, dict) and isinstance(raw.get("document"), dict) else raw
        if not args.confirm:
            raise ProfileError("--confirm is required to replace local settings")
        _print(args, client.post("/_kui/api/workspace/import",
                                 {"document": document, "confirm": True, "replace": not args.merge}))
        return 0
    if args.action == "backup":
        payload = {"confirm": args.confirm, "force": args.force, "fork": args.fork,
                   "message": args.message}
        _print(args, client.post("/_kui/api/workspace/backup", payload))
        return 0
    if args.action == "restore":
        if not args.confirm:
            raise ProfileError("--confirm is required; run `kui workspace preview` first")
        _print(args, client.post("/_kui/api/workspace/restore",
                                 {"confirm": True, "resolution": args.resolution, "sha": args.sha}))
        return 0
    if args.action == "repo":
        if bool(args.create) == bool(args.use):
            raise ProfileError("pass exactly one of --create NAME or --use OWNER/NAME")
        payload = {"mode": "create", "private": True} if args.create else {"mode": "existing"}
        if args.create:
            payload["name"] = args.create
        else:
            payload["repo"] = args.use
        _print(args, client.post("/_kui/api/workspace/repo", payload))
        return 0
    if args.action == "token":
        if not args.stdin:
            raise ProfileError("pass --stdin and pipe the token in; it is never echoed")
        token = sys.stdin.read().strip()
        if not token:
            raise ProfileError("no token on stdin")
        _print(args, client.put("/_kui/api/workspace/token", {"token": token}))
        return 0
    if args.action == "history":
        _print(args, client.get("/_kui/api/workspace/history", limit=args.limit))
        return 0
    if args.action == "rollback":
        if not args.confirm:
            raise ProfileError("--confirm is required")
        _print(args, client.post("/_kui/api/workspace/rollback", {"sha": args.sha, "confirm": True}))
        return 0
    raise ProfileError("unknown workspace action")

def cmd_models(args, store: ProfileStore) -> int:
    client = _client(args, store)
    if args.action == "status":
        _print(args, client.get("/_kui/api/workbench/model-backups", connection=args.connection))
        return 0
    if args.action == "remote":
        _print(args, client.get("/_kui/api/workbench/model-backups/remote",
                                connection=args.connection, target_path=args.target))
        return 0
    if args.action == "backup":
        if not args.confirm:
            raise ProfileError("--confirm is required; this uploads model files")
        job = client.post("/_kui/api/workbench/model-backups",
                          {"connection": args.connection, "artifact_ids": args.artifact,
                           "deployment_ids": args.deployment, "target_path": args.target,
                           "confirm": True})
        if args.wait:
            job = client.wait_job(job["id"], timeout=args.timeout)
        _print(args, job)
        return 0
    if args.action == "restore":
        if not args.confirm:
            raise ProfileError("--confirm is required; this downloads model files")
        artifact_id, _, embedded = args.artifact_id.partition("@")
        job = client.post("/_kui/api/workbench/model-backups/restore",
                          {"connection": args.connection, "artifact_id": artifact_id,
                           "version": args.version or embedded or None,
                           "target_path": args.target, "confirm": True})
        if args.wait:
            job = client.wait_job(job["id"], timeout=args.timeout)
        _print(args, job)
        return 0
    raise ProfileError("unknown models action")

def cmd_notebook(args, store: ProfileStore) -> int:
    resolved = store.resolve(args.profile)
    notebook = resolved.get("notebook") or {}
    credential = resolved.get("credential") or {}
    adapter = KaggleNotebooks(credential=credential)
    slug = args.slug or notebook.get("slug") or ""
    if not slug:
        raise ProfileError("no notebook slug; set one with `kui notebook set --slug owner/kernel`")
    if args.action == "deploy":
        source = args.source or notebook.get("source") or ""
        if not source:
            raise ProfileError("no notebook source path")
        inherited = None
        if args.metadata:
            try:
                payload = json.loads(Path(args.metadata).expanduser().read_text())
            except (OSError, ValueError) as exc:
                raise ProfileError(f"cannot read {args.metadata}: {exc}") from exc
            inherited = payload if isinstance(payload, dict) else None
        result = adapter.deploy(slug=slug, source=source, accelerator=args.accelerator or notebook.get("accelerator"),
                                timeout_s=args.timeout or notebook.get("timeout_s"),
                                run=not args.no_run, dataset_sources=args.dataset or [],
                                internet=not args.no_internet, extra_metadata=inherited)
        _print(args, result)
        return 0
    if args.action == "status":
        _print(args, adapter.status(slug))
        return 0
    if args.action == "open":
        print(adapter.open(slug))
        return 0
    if args.action == "stop":
        result = adapter.stop(slug)
        _print(args, result)
        print(result["note"], file=sys.stderr)
        return 8  # A missing capability is not success.
    if args.action == "output":
        _print(args, adapter.output(slug, args.out))
        return 0
    raise ProfileError("unknown notebook action")

def cmd_shutdown(args, store: ProfileStore) -> int:
    client = _client(args, store)
    if not args.confirm:
        raise ProfileError("--confirm is required; this stops the KaggleUI server")
    info = client.instance()
    _print(args, client.post("/_kui/api/shutdown", {"confirm": True, "stop_models": not args.keep_models}))
    print("KaggleUI is stopping. This does not end the Kaggle notebook/session; stop that manually.",
          file=sys.stderr)
    if info.get("host") == "kaggle":
        print("On Kaggle, use the notebook's Stop / End session control after the server exits.",
              file=sys.stderr)
    return 0

def cmd_tui(args, store: ProfileStore) -> int:
    from .tui import run
    return run(_client(args, store))

# -- parser ------------------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="kui", description="Manage KaggleUI and Kaggle notebooks locally.")
    parser.add_argument("--profile", help="profile name (default: active profile)")
    parser.add_argument("--url", help="instance URL override")
    parser.add_argument("--token", help="management key (prefer KAGGLEUI_MGMT_TOKEN)")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    parser.add_argument("--config", help="profile file path")
    sub = parser.add_subparsers(dest="command", required=True)

    profile = sub.add_parser("profile", help="profiles are references, not Kaggle accounts")
    psub = profile.add_subparsers(dest="action", required=True)
    psub.add_parser("list")
    show = psub.add_parser("show")
    show.add_argument("name", nargs="?")
    use = psub.add_parser("use")
    use.add_argument("name")
    add = psub.add_parser("add")
    add.add_argument("name")
    add.add_argument("--url", default=argparse.SUPPRESS)
    add.add_argument("--token", default=argparse.SUPPRESS)
    add.add_argument("--token-stdin", action="store_true")
    add.add_argument("--repo")
    add.add_argument("--branch", default="main")
    add.add_argument("--path", default="workspaces/default")
    add.add_argument("--token-secret", default="GITHUB_TOKEN")
    add.add_argument("--slug")
    add.add_argument("--source")
    add.add_argument("--accelerator")
    add.add_argument("--timeout", type=int)
    add.add_argument("--kaggle-username")
    add.add_argument("--key-file")
    add.add_argument("--key-stdin", action="store_true")
    add.add_argument("--use", action="store_true")

    instance = sub.add_parser("instance")
    isub = instance.add_subparsers(dest="action", required=True)
    isub.add_parser("list")
    iadd = isub.add_parser("add")
    iadd.add_argument("name")
    iadd.add_argument("--url", required=True)
    iadd.add_argument("--token")
    iadd.add_argument("--token-stdin", action="store_true")
    ishow = isub.add_parser("show")
    ishow.add_argument("name")

    ws_cfg = sub.add_parser("workspace-set", help="save local workspace profile")
    ws_cfg.add_argument("name")
    ws_cfg.add_argument("--repo", required=True)
    ws_cfg.add_argument("--branch", default="main")
    ws_cfg.add_argument("--path", default="workspaces/default")
    ws_cfg.add_argument("--token-secret", default="GITHUB_TOKEN")

    nb_cfg = sub.add_parser("notebook-set", help="save local notebook profile")
    nb_cfg.add_argument("name")
    nb_cfg.add_argument("--slug", required=True)
    nb_cfg.add_argument("--source", required=True)
    nb_cfg.add_argument("--accelerator")
    nb_cfg.add_argument("--timeout", type=int)

    sub.add_parser("status")

    deployments = sub.add_parser("deployments")
    dsub = deployments.add_subparsers(dest="action", required=True)
    dsub.add_parser("list")
    for action in ("start", "stop"):
        p = dsub.add_parser(action)
        p.add_argument("id")

    jobs = sub.add_parser("jobs")
    jsub = jobs.add_subparsers(dest="action", required=True)
    jlist = jsub.add_parser("list")
    jlist.add_argument("--limit", type=int, default=30)
    jshow = jsub.add_parser("show")
    jshow.add_argument("id")
    jlog = jsub.add_parser("log")
    jlog.add_argument("id")
    jlog.add_argument("--offset", type=int, default=0)
    jlog.add_argument("--limit", type=int, default=200)

    storage = sub.add_parser("storage")
    ssub = storage.add_subparsers(dest="action", required=True)
    ssub.add_parser("list")

    workspace = sub.add_parser("workspace")
    wsub = workspace.add_subparsers(dest="action", required=True)
    wsub.add_parser("status")
    wsub.add_parser("preview")
    wexport = wsub.add_parser("export")
    wexport.add_argument("--out")
    wimport = wsub.add_parser("import")
    wimport.add_argument("file")
    wimport.add_argument("--confirm", action="store_true")
    wimport.add_argument("--merge", action="store_true")
    wbackup = wsub.add_parser("backup")
    wbackup.add_argument("--confirm", action="store_true")
    wbackup.add_argument("--force", action="store_true")
    wbackup.add_argument("--fork", action="store_true")
    wbackup.add_argument("--message")
    wrestore = wsub.add_parser("restore")
    wrestore.add_argument("--confirm", action="store_true")
    wrestore.add_argument("--resolution", choices=("replace", "keep_local", "fork"), default="replace")
    wrestore.add_argument("--sha")
    whistory = wsub.add_parser("history")
    whistory.add_argument("--limit", type=int, default=20)
    wrollback = wsub.add_parser("rollback")
    wrollback.add_argument("sha")
    wrollback.add_argument("--confirm", action="store_true")
    wrepo = wsub.add_parser("repo", help="create a private repo or link an existing one")
    wrepo.add_argument("--create", metavar="NAME", help="create this private repo under the token's account")
    wrepo.add_argument("--use", metavar="OWNER/NAME", help="link an existing repository")
    wtoken = wsub.add_parser("token", help="store a GitHub token read from stdin")
    wtoken.add_argument("--stdin", action="store_true", help="read the token from stdin (recommended)")

    models = sub.add_parser("models")
    msub = models.add_subparsers(dest="action", required=True)
    mstatus = msub.add_parser("status")
    mstatus.add_argument("--connection")
    mremote = msub.add_parser("remote", help="list backups already stored on a destination")
    mremote.add_argument("--connection", required=True)
    mremote.add_argument("--target", default="kaggleui")
    mbackup = msub.add_parser("backup")
    mbackup.add_argument("--connection", required=True)
    mbackup.add_argument("--artifact", action="append", default=[])
    mbackup.add_argument("--deployment", action="append", default=[])
    mbackup.add_argument("--target", default="kaggleui")
    mbackup.add_argument("--confirm", action="store_true")
    mbackup.add_argument("--wait", action="store_true")
    mbackup.add_argument("--timeout", type=float, default=1800)
    mrestore = msub.add_parser("restore")
    mrestore.add_argument("--connection", required=True)
    mrestore.add_argument("--artifact-id", required=True)
    mrestore.add_argument("--version")
    mrestore.add_argument("--target", default="kaggleui")
    mrestore.add_argument("--confirm", action="store_true")
    mrestore.add_argument("--wait", action="store_true")
    mrestore.add_argument("--timeout", type=float, default=1800)

    notebook = sub.add_parser("notebook")
    nsub = notebook.add_subparsers(dest="action", required=True)
    ndeploy = nsub.add_parser("deploy")
    ndeploy.add_argument("--source")
    ndeploy.add_argument("--slug")
    ndeploy.add_argument("--accelerator")
    ndeploy.add_argument("--timeout", type=int)
    ndeploy.add_argument("--dataset", action="append")
    ndeploy.add_argument("--no-run", action="store_true")
    ndeploy.add_argument("--no-internet", action="store_true")
    ndeploy.add_argument("--metadata", help="kernel-metadata.json to inherit (BYOD image, machine shape)")
    nstatus = nsub.add_parser("status")
    nstatus.add_argument("slug", nargs="?")
    nopen = nsub.add_parser("open")
    nopen.add_argument("slug", nargs="?")
    nstop = nsub.add_parser("stop")
    nstop.add_argument("slug", nargs="?")
    noutput = nsub.add_parser("output")
    noutput.add_argument("slug", nargs="?")
    noutput.add_argument("--out", default="kaggle-output")
    for p in (nstatus, nopen, nstop, noutput):
        p.set_defaults(accelerator=None, timeout=None)

    shutdown = sub.add_parser("shutdown")
    shutdown.add_argument("--confirm", action="store_true")
    shutdown.add_argument("--keep-models", action="store_true")
    sub.add_parser("tui")
    return parser

def _normalize(argv: list[str]) -> list[str]:
    """Accept global flags before or after the subcommand; users type both."""
    moved: list[str] = []
    rest: list[str] = []
    index = 0
    while index < len(argv):
        item = argv[index]
        if item in {"--profile", "--url", "--token", "--config"}:
            moved.extend([item, argv[index + 1] if index + 1 < len(argv) else ""])
            index += 2
            continue
        if item == "--json":
            moved.append(item)
        else:
            rest.append(item)
        index += 1
    return moved + rest

def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(_normalize(list(sys.argv[1:] if argv is None else argv)))
    try:
        store = _store(args)
        if args.command == "profile":
            return cmd_profile(args, store)
        if args.command == "instance":
            return cmd_instance(args, store)
        if args.command == "workspace-set":
            return cmd_workspace_config(args, store)
        if args.command == "notebook-set":
            return cmd_notebook_config(args, store)
        if args.command == "status":
            return cmd_status(args, store)
        if args.command == "deployments":
            return cmd_deployments(args, store)
        if args.command == "jobs":
            return cmd_jobs(args, store)
        if args.command == "storage":
            return cmd_storage(args, store)
        if args.command == "workspace":
            return cmd_workspace(args, store)
        if args.command == "models":
            return cmd_models(args, store)
        if args.command == "notebook":
            return cmd_notebook(args, store)
        if args.command == "shutdown":
            return cmd_shutdown(args, store)
        if args.command == "tui":
            return cmd_tui(args, store)
        parser.error("unknown command")
    except ProfileError as exc:
        print(f"kui: {exc}", file=sys.stderr)
        return 3
    except KuiError as exc:
        print(f"kui: {exc}", file=sys.stderr)
        return exc.code
    except KaggleError as exc:
        print(f"kui: {exc}", file=sys.stderr)
        return 8
    except KeyboardInterrupt:
        return 130
    return 0

if __name__ == "__main__":
    raise SystemExit(main())



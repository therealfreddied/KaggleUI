"""Small terminal dashboard: instance health, deployments and lifecycle keys."""

from __future__ import annotations

import sys
from typing import Any

from .client import KuiError, ManagementClient


def snapshot(client: ManagementClient) -> dict[str, Any]:
    info = client.instance()
    jobs = client.jobs(limit=8)
    return {"instance": info, "jobs": jobs}

def render(info: dict[str, Any], selected: int) -> list[str]:
    inst = info["instance"]
    lines = [f"KaggleUI {inst.get('version')}  {inst.get('workspace_name') or ''}  "
             f"host={inst.get('host')}  up={int(inst.get('uptime_s') or 0)}s",
             f"URL {inst.get('url')}",
             f"workspace {inst.get('workspace_id')}  instance {inst.get('instance_id')}",
             "", "Deployments (s start, x stop):"]
    deployments = inst.get("deployments") or []
    if not deployments:
        lines.append("  (none)")
    for index, dep in enumerate(deployments):
        marker = ">" if index == selected else " "
        lines.append(f" {marker} {dep.get('id')}  {dep.get('served_name'):<24} "
                     f"{(dep.get('status') or ''):<10} {dep.get('modality') or ''}")
    lines += ["", "Recent jobs:"]
    for job in info["jobs"]:
        lines.append(f"   {job.get('id')}  {job.get('kind'):<20} {job.get('state'):<10} "
                     f"{(job.get('message') or '')[:40]}")
    lines += ["", "r refresh  s start  x stop  q quit"]
    return lines

def once(client: ManagementClient) -> int:
    try:
        info = snapshot(client)
    except KuiError as exc:
        print(f"[kui] {exc}", file=sys.stderr)
        return exc.code
    print("\n".join(render(info, 0)))
    return 0

def run(client: ManagementClient) -> int:
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        return once(client)
    import curses

    state: dict[str, Any] = {"info": snapshot(client), "selected": 0, "message": ""}

    def loop(screen):
        curses.curs_set(0)
        screen.nodelay(False)
        while True:
            screen.erase()
            lines = render(state["info"], state["selected"])
            if state["message"]:
                lines.append("")
                lines.append(state["message"])
            for row, line in enumerate(lines[: curses.LINES - 1]):
                try:
                    screen.addnstr(row, 0, line, curses.COLS - 1)
                except curses.error:
                    pass
            screen.refresh()
            key = screen.getch()
            deployments = state["info"]["instance"].get("deployments") or []
            if key in (ord("q"), 27):
                return 0
            if key == ord("r"):
                state["info"] = snapshot(client)
                state["selected"] = min(state["selected"], max(0, len(deployments) - 1))
                state["message"] = ""
            elif key in (curses.KEY_DOWN, ord("j")) and deployments:
                state["selected"] = min(len(deployments) - 1, state["selected"] + 1)
            elif key in (curses.KEY_UP, ord("k")):
                state["selected"] = max(0, state["selected"] - 1)
            elif key in (ord("s"), ord("x")) and deployments:
                dep = deployments[state["selected"]]
                action = "start" if key == ord("s") else "stop"
                try:
                    client.post(f"/_kui/api/workbench/deployments/{dep['id']}/{action}")
                    state["message"] = f"{action} requested for {dep['id']}"
                except KuiError as exc:
                    state["message"] = f"{action} failed: {exc}"
                state["info"] = snapshot(client)

    try:
        return curses.wrapper(loop)
    except KeyboardInterrupt:
        return 130


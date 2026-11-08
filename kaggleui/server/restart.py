"""Button-initiated restart.

A restart cannot rebind the port from inside the old process, so a detached helper waits for
the old server to exit and then re-executes ``python -m kaggleui serve`` with the same flags.
The old server keeps its public tunnel through the same handoff the updater uses, and the new
server adopts it, so the printed URL survives.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

from ..config import Paths
from ..schemas import Preset

SPEC = "restart.json"
PRESET = "restart-preset.json"

def prepare(state) -> Path:
    """Write the restart spec and spawn the detached helper. The caller then stops the server."""
    paths: Paths = state.paths
    preset = Preset(settings=state.settings,
                    deployments=[d.spec for d in state.manager.deployments.values()
                                 if d.status not in ("stopped", "stopping", "error")])
    preset_path = paths.home / PRESET
    preset_path.write_text(json.dumps(preset.model_dump()))
    preset_path.chmod(0o600)
    spec = paths.home / SPEC
    payload = {"home": str(paths.home), "port": state.port, "host": state.listen_host,
               "preset": str(preset_path), "mock": state.mock, "fake_hw": state.fake_hw,
               "pid": os.getpid()}
    spec.write_text(json.dumps(payload))
    spec.chmod(0o600)
    # The helper runs with cwd=home, which is not importable: pip installs the package, but a
    # source checkout (dev, termux) only exists on sys.path, so carry that root along.
    root = str(Path(__file__).resolve().parents[2])
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([root, env.get("PYTHONPATH", "")]).strip(os.pathsep)
    log = paths.logs / "server.log"
    with log.open("ab") as stream:
        subprocess.Popen([sys.executable, "-m", "kaggleui.server.restart", str(spec)], stdout=stream,
                         stderr=subprocess.STDOUT, start_new_session=True, cwd=str(paths.home), env=env)
    return spec

def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True

def _wait_for_exit(pid: int, timeout: float = 120) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not _alive(pid):
            return
        time.sleep(0.25)
    raise RuntimeError("The previous KaggleUI server did not exit; not restarting.")

def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    spec = Path(argv[0])
    config = json.loads(spec.read_text())
    _wait_for_exit(config["pid"])
    cmd = [sys.executable, "-m", "kaggleui", "serve", "--home", config["home"],
           "--port", str(config["port"]), "--host", config["host"], "--preset", config["preset"]]
    if config.get("mock"):
        cmd.append("--mock-engines")
    if config.get("fake_hw"):
        cmd += ["--fake-hw", config["fake_hw"]]
    log = Path(config["home"]) / "logs" / "server.log"
    with log.open("ab") as stream:
        stream.write(b"\n[kaggleui] Restarted from the panel.\n")
        stream.flush()
        subprocess.Popen(cmd, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    print("[kaggleui] Restarted.", flush=True)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

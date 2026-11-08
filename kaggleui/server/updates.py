"""Release discovery, compatibility policy, and a host-authorized wheel updater."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlsplit

import httpx
import psutil

from .. import __version__
from ..schemas import Preset

REPO = "therealfreddied/KaggleUI"
POLICY_URL = f"https://raw.githubusercontent.com/{REPO}/main/release-policy.json"
RELEASE_URL = f"https://api.github.com/repos/{REPO}/releases/latest"


def version(value: str) -> tuple[int, ...]:
    match = re.fullmatch(r"v?(\d+)\.(\d+)\.(\d+)", value)
    if not match:
        raise ValueError("Expected a stable release version")
    return tuple(int(part) for part in match.groups())


class Updates:
    def __init__(self, state):
        self.state = state
        self.latest: dict | None = None
        self.policy: dict = {}
        self.checked_at: float | None = None
        self.error: str | None = None
        self._lock = asyncio.Lock()
        self._process: subprocess.Popen | None = None
        self.folder = state.paths.home / "update"
        try:
            policy = json.loads((self.folder / "policy.json").read_text())
            version(policy["minimum_supported"])
            for revoked in policy.get("revoked_versions", []):
                version(revoked)
            self.policy = policy
        except (OSError, ValueError, KeyError, TypeError):
            pass

    @property
    def blocked(self) -> bool:
        minimum = self.policy.get("minimum_supported")
        return bool((minimum and version(__version__) < version(minimum)) or
                    __version__ in self.policy.get("revoked_versions", []))

    def status(self) -> dict:
        progress = None
        try:
            progress = json.loads((self.folder / "status.json").read_text())
        except (OSError, ValueError):
            pass
        if progress and progress.get("status") in ("installing", "restarting"):
            if self._process and self._process.poll() is not None:
                progress = {"status": "error", "message": "Update process exited. See System → update log or re-run the install cell."}
            elif progress.get("pid"):
                try:
                    process = psutil.Process(progress["pid"])
                    if abs(process.create_time() - progress["process_started_at"]) > .01 or process.status() == psutil.STATUS_ZOMBIE:
                        raise psutil.NoSuchProcess(process.pid)
                except (psutil.NoSuchProcess, KeyError):
                    progress = {"status": "error", "message": "The previous update was interrupted. Retry the update or re-run the install cell."}
        latest = self.latest
        return {"current": __version__, "latest": latest.get("version") if latest else None,
                "available": bool(latest and version(latest["version"]) > version(__version__)),
                "release_url": latest.get("url") if latest else f"https://github.com/{REPO}/releases",
                "minimum_supported": self.policy.get("minimum_supported"), "blocked": self.blocked,
                "message": self.policy.get("message", "This version needs an update before it can run models."),
                "checked_at": self.checked_at, "error": self.error, "progress": progress}

    async def check(self, force: bool = False) -> dict:
        async with self._lock:
            if self.checked_at and time.time() - self.checked_at < (30 if force else 3600):
                return self.status()
            was_blocked = self.blocked
            errors = []
            async with httpx.AsyncClient(timeout=10, follow_redirects=True,
                                         headers={"User-Agent": f"KaggleUI/{__version__}"}) as client:
                responses = await asyncio.gather(client.get(POLICY_URL), client.get(RELEASE_URL), return_exceptions=True)
            for kind, response in zip(("policy", "release"), responses):
                try:
                    if isinstance(response, BaseException):
                        raise response
                    response.raise_for_status()
                    data = response.json()
                    if kind == "policy":
                        version(data["minimum_supported"])
                        for revoked in data.get("revoked_versions", []):
                            version(revoked)
                        self.policy = data
                        self.folder.mkdir(parents=True, exist_ok=True)
                        (self.folder / "policy.json").write_text(json.dumps(data))
                    else:
                        release_version = data["tag_name"].removeprefix("v")
                        version(release_version)
                        name = f"kaggleui-{release_version}-py3-none-any.whl"
                        asset = next(a for a in data["assets"] if a["name"] == name)
                        url = asset["browser_download_url"]
                        if not url.startswith(f"https://github.com/{REPO}/releases/download/"):
                            raise ValueError("Unrecognized release asset URL")
                        self.latest = {"version": release_version, "url": data["html_url"], "wheel": url,
                                       "digest": asset.get("digest")}
                except Exception:
                    errors.append(f"Couldn't check the {kind}; using the last known result.")
            self.checked_at = time.time()
            self.error = " ".join(errors) or None
            if self.blocked and not was_blocked:
                await self.state.manager.shutdown()
            return self.status()

    async def run(self):
        while True:
            await self.check()
            await asyncio.sleep(600)

    async def install(self) -> dict:
        if self._process and self._process.poll() is None:
            return self.status()
        await self.check()
        if self._process and self._process.poll() is None:
            return self.status()
        if not self.latest or not self.status()["available"]:
            raise ValueError("No newer release with an installable wheel is available")
        self.folder.mkdir(parents=True, exist_ok=True)
        preset = Preset(settings=self.state.settings, deployments=[d.spec for d in self.state.manager.deployments.values()
                                                                  if d.status not in ("stopped", "stopping", "error")])
        config = {**self.latest, "home": str(self.state.paths.home), "port": self.state.port,
                  "host": self.state.listen_host, "local_url": self.state.local_url(),
                  "preset": preset.model_dump(), "mock": self.state.mock, "fake_hw": self.state.fake_hw}
        path = self.folder / "request.json"
        path.write_text(json.dumps(config))
        path.chmod(0o600)
        (self.folder / "status.json").write_text(json.dumps({"status": "installing", "message": "Downloading the release…"}))
        env = {**os.environ, "KAGGLEUI_ADMIN_TOKEN": self.state.auth.admin_token,
               "KAGGLEUI_PASSWORD": self.state.auth.password, "KAGGLEUI_API_KEY": self.state.auth.api_key}
        with (self.state.paths.logs / "update.log").open("ab") as log:
            self._process = subprocess.Popen([sys.executable, "-m", "kaggleui.server.updates", str(path)],
                                             env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        (self.folder / "status.json").write_text(json.dumps({"status": "installing", "message": "Downloading the release…",
                                                            "pid": self._process.pid,
                                                            "process_started_at": psutil.Process(self._process.pid).create_time()}))
        return self.status()


def _install(path: Path) -> None:
    """Separate process so replacing package files and restarting cannot terminate the installer."""
    config = json.loads(path.read_text())
    folder = path.parent

    def progress(status, message):
        temp = folder / "status.tmp"
        temp.write_text(json.dumps({"status": status, "message": message, "pid": os.getpid(),
                                    "process_started_at": psutil.Process().create_time()}))
        temp.replace(folder / "status.json")

    try:
        url = config["wheel"]
        if not url.startswith(f"https://github.com/{REPO}/releases/download/"):
            raise ValueError("Unrecognized wheel source")
        wheel = folder / Path(urlsplit(url).path).name
        digest = hashlib.sha256()
        with httpx.stream("GET", url, follow_redirects=True, timeout=120) as response, wheel.open("wb") as target:
            response.raise_for_status()
            for block in response.iter_bytes():
                target.write(block)
                digest.update(block)
        expected = config.get("digest")
        if expected and expected != "sha256:" + digest.hexdigest():
            raise ValueError("The wheel checksum does not match the GitHub release")
        progress("installing", "Installing KaggleUI " + config["version"] + "…")
        uv = shutil.which("uv")
        command = [uv, "pip", "install", "--python", sys.executable] if uv else [sys.executable, "-m", "pip", "install"]
        subprocess.run([*command, "--upgrade", str(wheel)], check=True)
        preset = folder / "restart.json"
        preset.write_text(json.dumps(config["preset"]))
        preset.chmod(0o600)
        progress("restarting", "Restarting the app. The existing public tunnel is kept when supported; this page reconnects automatically.")
        from ..launcher import stop

        stop(config["home"], _preserve_pid=os.getpid(), _keep_tunnel=True)
        cmd = [sys.executable, "-m", "kaggleui", "serve", "--home", config["home"], "--port", str(config["port"]),
               "--host", config["host"], "--preset", str(preset)]
        if config["mock"]:
            cmd.append("--mock-engines")
        if config.get("fake_hw"):
            cmd += ["--fake-hw", config["fake_hw"]]
        with (Path(config["home"]) / "logs" / "server.log").open("ab") as log:
            process = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        for _ in range(120):
            if process.poll() is not None:
                raise RuntimeError("The updated server exited. See server.log.")
            try:
                with httpx.Client(timeout=2) as client:
                    response = client.get(config["local_url"] + "/_kui/api/session")
                    if response.is_success and response.json().get("version") == config["version"]:
                        progress("complete", "Updated to " + config["version"])
                        return
            except httpx.HTTPError:
                pass
            time.sleep(1)
        raise RuntimeError("Restart timed out. Re-run the KaggleUI notebook cell.")
    except Exception as exc:
        progress("error", str(exc))
        raise
    finally:
        path.unlink(missing_ok=True)


if __name__ == "__main__":
    _install(Path(sys.argv[1]))

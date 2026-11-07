"""Hugging Face Hub and GitHub destinations.

Both speak HTTP through stdlib ``urllib`` only — no hub libraries are needed in the server process, and the
same code works in a training venv or a bare Colab image. Every blocking call is pushed to a thread so the
event loop keeps serving requests while a multi-gigabyte file uploads.

GitHub is deliberately text-only: the contents API is for catalog/manifest metadata, not weights.
"""

from __future__ import annotations

import asyncio
import base64
import importlib.util
import json
import os
import shutil
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from ..workbench.runner import EVENT_TAG
from .base import RemoteEntry, StorageError, check_remote, transfer_scratch

HF_ENDPOINT = "https://huggingface.co"
GITHUB_ENDPOINT = "https://api.github.com"
TIMEOUT = 60
GITHUB_MAX_BYTES = 1024 * 1024
GITHUB_BINARY_SUFFIXES = frozenset({
    ".bin", ".safetensors", ".gguf", ".pt", ".pth", ".ckpt", ".zip", ".tar", ".gz", ".tgz", ".7z",
    ".onnx", ".mp4", ".png", ".jpg", ".jpeg", ".webp", ".wav", ".mp3",
})

def _request(url: str, *, method: str = "GET", token: str | None = None, data: bytes | None = None,
             headers: dict[str, str] | None = None) -> tuple[int, bytes]:
    request = urllib.request.Request(url, data=data, method=method)
    request.add_header("User-Agent", "kaggleui")
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    for key, value in (headers or {}).items():
        request.add_header(key, value)
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()[:400]
    except urllib.error.URLError as exc:
        raise StorageError(f"network error talking to {url.split('/')[2]}: {exc.reason}") from exc

async def _arequest(url: str, **kwargs) -> tuple[int, bytes]:
    return await asyncio.to_thread(_request, url, **kwargs)

def _json(body: bytes, what: str):
    try:
        return json.loads(body or b"null")
    except ValueError as exc:
        raise StorageError(f"{what} returned non-JSON: {body[:120]!r}") from exc

def _stream(url: str, dest: Path, token: str | None, *, progress=None) -> None:
    request = urllib.request.Request(url)
    request.add_header("User-Agent", "kaggleui")
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response, dest.open("wb") as out:
            total = int(response.headers.get("Content-Length") or 0)
            done = 0
            while chunk := response.read(1024 * 1024):
                out.write(chunk)
                done += len(chunk)
                if progress and total:
                    progress(done / total)
    except urllib.error.HTTPError as exc:
        raise StorageError(f"download failed: HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:
        raise StorageError(f"network error downloading: {exc.reason}") from exc

def _put_file(upload_url: str, src: Path, token: str, *, progress=None) -> None:
    payload = src.read_bytes()
    request = urllib.request.Request(upload_url, data=payload, method="PUT")
    request.add_header("User-Agent", "kaggleui")
    request.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
            response.read()
            if progress:
                progress(1.0)
    except urllib.error.HTTPError as exc:
        raise StorageError(f"upload failed: HTTP {exc.code} {exc.read()[:200]!r}") from exc

def _put_file_with_total(upload_url: str, src: Path, token: str, progress=None) -> None:
    """``_put_file`` with indeterminate progress reported before and after the single PUT."""
    if progress:
        progress(0.1)
    _put_file(upload_url, src, token, progress=progress)

class HFProvider:
    """Hub uploads via ``python -m kaggleui.storage.hf_worker``; the token rides in the environment."""

    name = "hf"

    def __init__(self, repo: str, *, repo_type: str = "model", token: str | None = None,
                 endpoint: str = HF_ENDPOINT, cache: Path | None = None):
        if not repo or "/" not in repo:
            raise StorageError("HF destination must look like 'owner/name'")
        if repo_type not in ("model", "dataset"):
            raise StorageError("HF repo_type must be model or dataset")
        if importlib.util.find_spec("huggingface_hub") is None:
            raise StorageError("Hugging Face transfers need the huggingface_hub package; install it or use "
                               "an rclone remote")
        self.repo = repo
        self.repo_type = repo_type
        self.token = token
        self.endpoint = endpoint.rstrip("/")
        self.cache = Path(cache or "/tmp")

    async def _run(self, spec: dict, *, progress=None) -> dict:
        scratch = transfer_scratch(self.cache, "hf")
        spec_path = scratch / "spec.json"
        spec = {**spec, "repo": self.repo, "repo_type": self.repo_type}
        spec_path.write_text(json.dumps(spec))
        env = {**os.environ, "HF_HUB_DISABLE_PROGRESS_BARS": "1", "HF_ENDPOINT": self.endpoint}
        if self.token:
            env["HF_TOKEN"] = self.token
        try:
            proc = await asyncio.create_subprocess_exec(
                sys.executable, "-m", "kaggleui.storage.hf_worker", str(spec_path),
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, env=env)
            result: dict | None = None
            stderr = b""
            assert proc.stdout is not None
            async for raw in proc.stdout:
                line = raw.decode(errors="replace").rstrip()
                if line.startswith(EVENT_TAG):
                    try:
                        payload = json.loads(line[len(EVENT_TAG):].strip())
                    except ValueError:
                        continue
                    if progress and isinstance(payload.get("fraction"), (int, float)):
                        progress(float(payload["fraction"]))
                elif line:
                    try:
                        result = json.loads(line)
                    except ValueError:
                        continue
            if proc.stderr is not None:
                stderr = await proc.stderr.read()
            code = await proc.wait()
            if code != 0:
                detail = stderr.decode(errors="replace").strip().splitlines()
                raise StorageError(detail[-1] if detail else f"huggingface_hub worker exited with {code}")
            return result or {"ok": True}
        finally:
            shutil.rmtree(scratch, ignore_errors=True)

    async def test(self) -> dict:
        try:
            return await self._run({"action": "test"})
        except StorageError as exc:
            return {"ok": False, "error": str(exc)}

    async def list(self, prefix: str) -> list[RemoteEntry]:
        result = await self._run({"action": "list", "prefix": prefix})
        return [RemoteEntry(**row) for row in result.get("entries", [])]

    async def upload(self, local: Path, remote: str, *, progress=None) -> None:
        if not self.token:
            raise StorageError("HF uploads need the HF_TOKEN secret")
        await self._run({"action": "upload", "remote": check_remote(remote), "local": str(local)},
                        progress=progress)

    async def download(self, remote: str, local: Path, *, progress=None) -> None:
        await self._run({"action": "download", "remote": check_remote(remote), "local": str(local)},
                        progress=progress)

    async def stat(self, remote: str) -> RemoteEntry | None:
        result = await self._run({"action": "stat", "remote": check_remote(remote)})
        entry = result.get("entry")
        return RemoteEntry(**entry) if entry else None

    async def delete(self, remote: str) -> None:
        if not self.token:
            raise StorageError("HF deletes need the HF_TOKEN secret")
        await self._run({"action": "delete", "remote": check_remote(remote), "confirm": True})

class GitHubProvider:
    """Commits small text files (catalog entries, manifests) through the GitHub contents API."""

    name = "github"

    def __init__(self, repo: str, *, path: str = "", branch: str = "main", token: str | None = None,
                 endpoint: str = GITHUB_ENDPOINT, cache: Path | None = None):
        if repo.count("/") != 1:
            raise StorageError("GitHub destination must look like 'owner/repo'")
        self.repo = repo
        self.root = (path or "").strip("/")
        self.branch = branch or "main"
        self.token = token
        self.endpoint = endpoint.rstrip("/")
        self.cache = Path(cache or "/tmp")

    def _headers(self, accept: str = "application/vnd.github+json") -> dict[str, str]:
        headers = {"Accept": accept}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    def _url(self, remote: str = "") -> str:
        remote = check_remote(remote) if remote else ""
        full = "/".join(part for part in (self.root, remote) if part)
        quoted = urllib.parse.quote(full)
        return f"{self.endpoint}/repos/{self.repo}/contents/{quoted}"

    async def test(self) -> dict:
        code, body = await _arequest(f"{self.endpoint}/repos/{self.repo}", headers=self._headers())
        if code == 404:
            return {"ok": False, "error": f"{self.repo} not found (or not visible with this token)"}
        if code >= 400:
            return {"ok": False, "error": f"HTTP {code}: {body[:160]!r}"}
        info = _json(body, "GitHub repo")
        return {"ok": True, "repo": info.get("full_name"), "default_branch": info.get("default_branch"),
                "token": bool(self.token)}

    async def list(self, prefix: str) -> list[RemoteEntry]:
        code, body = await _arequest(self._url(prefix), headers=self._headers())
        if code == 404:
            return []
        if code >= 400:
            raise StorageError(f"GitHub listing failed: HTTP {code}")
        rows = _json(body, "GitHub contents") or []
        if isinstance(rows, dict):
            rows = [rows]
        base = "/".join(part for part in (self.root, prefix.strip("/")) if part)
        return [RemoteEntry(path=row["path"][len(base) + 1:] if base and row["path"].startswith(base + "/")
                            else row["path"].removeprefix(f"{self.root}/") if self.root else row["path"],
                            size=row.get("size"), is_dir=row.get("type") == "dir")
                for row in rows]

    def _check_file(self, local: Path) -> Path:
        src = Path(local)
        if src.suffix.lower() in GITHUB_BINARY_SUFFIXES:
            raise StorageError(f"GitHub is for text metadata only; {src.suffix} files don't belong there "
                               "(use an rclone remote or a Hugging Face repository)")
        if src.stat().st_size > GITHUB_MAX_BYTES:
            raise StorageError(f"{src.name} is larger than 1 MiB; GitHub is for catalog/manifest text only")
        return src

    async def _sha(self, remote: str) -> str | None:
        code, body = await _arequest(self._url(remote), headers=self._headers())
        if code == 404:
            return None
        if code >= 400:
            raise StorageError(f"GitHub lookup failed: HTTP {code}")
        info = _json(body, "GitHub contents")
        return (info or {}).get("sha")

    async def upload(self, local: Path, remote: str, *, progress=None) -> None:
        if not self.token:
            raise StorageError("GitHub uploads need the GITHUB_TOKEN secret")
        src = self._check_file(local)
        remote = check_remote(remote)
        sha = await self._sha(remote)
        payload: dict[str, object] = {
            "message": f"kaggleui: {'update' if sha else 'add'} {remote}",
            "content": base64.b64encode(src.read_bytes()).decode(),
            "branch": self.branch,
        }
        if sha:
            payload["sha"] = sha
        code, body = await _arequest(self._url(remote), method="PUT", headers=self._headers(),
                                     data=json.dumps(payload).encode())
        if code >= 400:
            raise StorageError(f"GitHub upload failed: HTTP {code} {body[:200]!r}")
        if progress:
            progress(1.0)

    async def download(self, remote: str, local: Path, *, progress=None) -> None:
        remote = check_remote(remote)
        dest = Path(local)
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(dest.name + ".part")
        await asyncio.to_thread(_stream, self._url(remote), tmp, self.token, progress=progress)
        tmp.replace(dest)

    async def stat(self, remote: str) -> RemoteEntry | None:
        code, body = await _arequest(self._url(remote), headers=self._headers())
        if code == 404:
            return None
        if code >= 400:
            raise StorageError(f"GitHub lookup failed: HTTP {code}")
        info = _json(body, "GitHub contents") or {}
        if info.get("type") != "file":
            return RemoteEntry(path=remote, is_dir=True)
        return RemoteEntry(path=remote, size=info.get("size"))

    async def delete(self, remote: str) -> None:
        if not self.token:
            raise StorageError("GitHub deletes need the GITHUB_TOKEN secret")
        remote = check_remote(remote)
        sha = await self._sha(remote)
        if sha is None:
            raise StorageError(f"{remote} is not in {self.repo}")
        code, body = await _arequest(self._url(remote), method="DELETE", headers=self._headers(),
                                     data=json.dumps({"message": f"kaggleui: delete {remote}",
                                                      "sha": sha, "branch": self.branch}).encode())
        if code >= 400:
            raise StorageError(f"GitHub delete failed: HTTP {code} {body[:200]!r}")

def clean_scratch(path: Path) -> None:
    """Remove a transfer scratch directory; tolerant of partial state."""
    shutil.rmtree(path, ignore_errors=True)

"""Durable artifact store: content-addressed file bundles with a manifest per version.

Layout under ``paths.models/artifacts``::

    <artifact_id>/<version>/manifest.json
    <artifact_id>/<version>/<relative file paths>

``add`` stages the files, hashes them, fsyncs and renames, so a crash leaves either the old version or
nothing — never a half-written one. Deletion needs ``confirm=True``.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import threading
import time
from pathlib import Path

from ..schemas import ArtifactFile, ArtifactManifest

CHUNK = 1 << 20

class RegistryError(Exception):
    """The artifact cannot be added, verified or removed."""

def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()

def _common_parent(files: list[Path]) -> Path:
    """Deepest directory that contains every file, so nested layouts keep their relative paths."""
    parents = [p.parent for p in files]
    base = parents[0]
    for parent in parents[1:]:
        while base not in parent.parents and parent != base:
            base = base.parent
    return base

def _safe_id(value: str) -> str:
    cleaned = "".join(c if c.isalnum() or c in "-_." else "-" for c in value).strip(".-")
    if not cleaned:
        raise RegistryError("artifact_id must contain letters or digits")
    return cleaned

class ArtifactRegistry:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.index = self.root / "index.json"
        # index.json is read-modify-written by every add/remove; the lock covers the whole cycle.
        self._lock = threading.Lock()

    # -- reading ---------------------------------------------------------------------------------
    def _entries(self) -> list[dict]:
        try:
            data = json.loads(self.index.read_text())
        except (OSError, ValueError):
            return []
        return data.get("artifacts", []) if isinstance(data, dict) else []

    def _write_index(self, entries: list[dict]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(dir=str(self.root), prefix=f"{self.index.name}.", suffix=".tmp")
        tmp = Path(name)
        try:
            with os.fdopen(fd, "w") as handle:
                json.dump({"schema_version": 1, "artifacts": entries}, handle, indent=1)
            os.chmod(tmp, 0o600)
            tmp.replace(self.index)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise

    def dir_for(self, artifact_id: str, version: str) -> Path:
        return self.root / _safe_id(artifact_id) / _safe_id(version)

    def get(self, artifact_id: str, version: str | None = None) -> ArtifactManifest | None:
        entries = [e for e in self._entries() if e.get("artifact_id") == artifact_id]
        if not entries:
            return None
        entry = max(entries, key=lambda e: e.get("created_at") or 0) if version is None else next(
            (e for e in entries if e.get("version") == version), None)
        if entry is None:
            return None
        manifest_path = Path(entry["dir"]) / "manifest.json"
        try:
            return ArtifactManifest.model_validate_json(manifest_path.read_text())
        except (OSError, ValueError):
            return None

    def list(self) -> list[ArtifactManifest]:
        out: list[ArtifactManifest] = []
        for entry in self._entries():
            try:
                out.append(ArtifactManifest.model_validate_json((Path(entry["dir"]) / "manifest.json").read_text()))
            except (OSError, ValueError):
                continue
        return sorted(out, key=lambda m: m.created_at, reverse=True)

    # -- writing ---------------------------------------------------------------------------------
    def add(self, manifest: ArtifactManifest, files: list[Path], *, overwrite: bool = False) -> ArtifactManifest:
        artifact_id = _safe_id(manifest.artifact_id)
        version = _safe_id(manifest.version)
        target = self.root / artifact_id / version
        if target.exists() and not overwrite:
            raise RegistryError(f"{artifact_id} {version} already exists; pass overwrite to replace it")
        if not files:
            raise RegistryError("an artifact needs at least one file")

        self.root.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=".staging-", dir=self.root))
        try:
            described: list[ArtifactFile] = []
            base = _common_parent([Path(f) for f in files])
            for src in (Path(f) for f in files):
                if not src.is_file():
                    raise RegistryError(f"{src} is not a file")
                rel = str(src.relative_to(base))
                dest = staging / rel
                if dest.exists():
                    raise RegistryError(f"duplicate file name {rel}")
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dest)
                with dest.open("rb+") as f:
                    os.fsync(f.fileno())
                described.append(ArtifactFile(path=rel, size=dest.stat().st_size, sha256=sha256_of(dest)))
            final = manifest.model_copy(update={"artifact_id": artifact_id, "version": version, "files": described,
                                                "created_at": manifest.created_at or time.time()})
            (staging / "manifest.json").write_text(json.dumps(final.model_dump(), indent=1))
            if target.exists():
                shutil.rmtree(target)
            target.parent.mkdir(parents=True, exist_ok=True)
            staging.replace(target)
        finally:
            if staging.exists():
                shutil.rmtree(staging, ignore_errors=True)

        self._lock_entries_write(artifact_id, version, {
            "artifact_id": artifact_id, "version": version, "kind": final.kind,
            "created_at": final.created_at, "dir": str(target)})
        return final

    def _lock_entries_write(self, artifact_id: str, version: str, entry: dict) -> None:
        with self._lock:
            entries = [e for e in self._entries() if not (e.get("artifact_id") == artifact_id
                                                           and e.get("version") == version)]
            entries.append(entry)
            self._write_index(entries)

    # -- maintenance -----------------------------------------------------------------------------
    def verify(self, artifact_id: str, version: str | None = None) -> dict:
        manifest = self.get(artifact_id, version)
        if manifest is None:
            return {"ok": False, "error": "no such artifact", "problems": []}
        base = self.dir_for(manifest.artifact_id, manifest.version)
        problems: list[str] = []
        for entry in manifest.files:
            path = base / entry.path
            if not path.exists():
                problems.append(f"missing: {entry.path}")
            elif path.stat().st_size != entry.size:
                problems.append(f"size changed: {entry.path}")
            elif sha256_of(path) != entry.sha256:
                problems.append(f"hash mismatch: {entry.path}")
        return {"ok": not problems, "problems": problems, "artifact": manifest.model_dump()}

    def remove(self, artifact_id: str, version: str, *, confirm: bool) -> None:
        if not confirm:
            raise RegistryError("deleting an artifact requires confirm=True")
        target = self.dir_for(artifact_id, version)
        if not target.exists():
            raise RegistryError(f"{artifact_id} {version} does not exist")
        shutil.rmtree(target)
        with self._lock:
            self._write_index([e for e in self._entries()
                               if not (e.get("artifact_id") == artifact_id and e.get("version") == version)])

    def public(self) -> list[dict]:
        return [m.model_dump() for m in self.list()]

"""Scoped management keys for local CLI/TUI clients.

The admin token opens the whole panel. These keys are narrower, stored hashed, and can be revoked per
machine. They are deliberately not valid for the HTML panel, deployment creation, model downloads or
the gateway; a stolen CLI key cannot replace an admin session.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
import time
from pathlib import Path

from fastapi import HTTPException
from starlette.requests import HTTPConnection as Request

from ..workbench.store import _read, _write

SCOPES = ("status", "deployments", "jobs", "workspace", "storage", "shutdown")
DEFAULT_SCOPES = ("status", "deployments", "jobs", "workspace", "storage")

def _hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()

class ManagementTokens:
    def __init__(self, path: Path):
        self.path = path
        self.data = _read(path, {"schema_version": 1, "tokens": []})
        if not isinstance(self.data.get("tokens"), list):
            self.data["tokens"] = []

    def _save(self) -> None:
        _write(self.path, self.data)

    def _public(self, record: dict) -> dict:
        return {k: record.get(k) for k in ("id", "label", "prefix", "scopes", "created_at",
                                           "last_used_at", "revoked_at")}

    def list(self) -> list[dict]:
        return [self._public(r) for r in self.data["tokens"]]

    def create(self, label: str, scopes: list[str] | None = None) -> tuple[str, dict]:
        clean = [s for s in (scopes or list(DEFAULT_SCOPES)) if s in SCOPES]
        if not clean:
            raise ValueError("at least one management scope is required")
        if "shutdown" in clean and "shutdown" not in (scopes or []):
            clean.remove("shutdown")
        token = f"kui_mgmt_{secrets.token_urlsafe(32)}"
        record = {"id": secrets.token_hex(6), "label": (label or "cli").strip()[:60] or "cli",
                  "prefix": token[:16], "hash": _hash(token), "scopes": clean,
                  "created_at": time.time(), "last_used_at": None, "revoked_at": None}
        self.data["tokens"].append(record)
        self._save()
        return token, self._public(record)

    def revoke(self, token_id: str) -> bool:
        for record in self.data["tokens"]:
            if record.get("id") == token_id and not record.get("revoked_at"):
                record["revoked_at"] = time.time()
                self._save()
                return True
        return False

    def verify(self, token: str | None) -> dict | None:
        if not token or not token.startswith("kui_mgmt_"):
            return None
        digest = _hash(token)
        for record in self.data["tokens"]:
            if record.get("revoked_at") or not hmac.compare_digest(str(record.get("hash")), digest):
                continue
            record["last_used_at"] = time.time()
            self._save()
            return self._public(record)
        return None

def bearer(request: Request) -> str | None:
    header = request.headers.get("authorization") or ""
    if header.lower().startswith("bearer "):
        return header[7:].strip()
    return None

def require_scope(request: Request, scope: str) -> dict:
    """Admin token passes everything; a management key needs the named capability."""
    st = request.app.state.kui  # type: ignore[attr-defined]
    if st.auth.is_admin(request):
        return {"label": "admin", "scopes": list(SCOPES)}
    record = st.management.verify(bearer(request))
    if record is None:
        raise HTTPException(401, "Admin token or management key required")
    if scope not in record["scopes"]:
        raise HTTPException(403, f"Management key lacks the {scope} capability")
    return record


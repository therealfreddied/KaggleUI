"""Typed-enough HTTP client shared by the kui CLI and TUI.

Only scoped management routes are called; the admin token is accepted by the same routes for local
use. Tokens are never included in exception text or debug output.
"""

from __future__ import annotations

import time
from typing import Any

import httpx


class KuiError(Exception):
    def __init__(self, message: str, *, code: int = 6, status: int | None = None, body: Any = None):
        super().__init__(message)
        self.code = code
        self.status = status
        self.body = body

class ManagementClient:
    def __init__(self, url: str, token: str, *, timeout: float = 20.0, transport: httpx.BaseTransport | None = None):
        if not url:
            raise KuiError("no instance URL configured", code=3)
        if not token:
            raise KuiError("no management key configured for this instance", code=3)
        self.base = url.rstrip("/")
        self._token = token
        self.timeout = timeout
        self.transport = transport

    def _request(self, method: str, path: str, *, json: dict | None = None,
                 params: dict | None = None) -> Any:
        headers = {"Authorization": f"Bearer {self._token}", "Accept": "application/json"}
        try:
            with httpx.Client(timeout=self.timeout, follow_redirects=False,
                              transport=self.transport) as client:
                response = client.request(method, f"{self.base}{path}", headers=headers, json=json, params=params)
        except httpx.HTTPError as exc:
            raise KuiError(f"cannot reach {self.base}: {exc.__class__.__name__}", code=4) from exc
        if response.status_code == 401:
            raise KuiError("the management key was rejected", code=5, status=401)
        if response.status_code == 403:
            raise KuiError("the management key lacks this capability", code=5, status=403)
        try:
            body = response.json()
        except ValueError:
            body = {"detail": response.text[:500]}
        if response.status_code >= 400:
            detail = body.get("detail") if isinstance(body, dict) else body
            if isinstance(detail, dict):
                detail = detail.get("message") or str(detail)
            message = str(detail or f"HTTP {response.status_code}")
            code = 7 if response.status_code == 409 else 6
            raise KuiError(message, code=code, status=response.status_code, body=body)
        return body

    def get(self, path: str, **params: Any) -> Any:
        return self._request("GET", path, params={k: v for k, v in params.items() if v is not None})

    def post(self, path: str, payload: dict | None = None) -> Any:
        return self._request("POST", path, json=payload or {})

    def put(self, path: str, payload: dict) -> Any:
        return self._request("PUT", path, json=payload)

    def delete(self, path: str) -> Any:
        return self._request("DELETE", path)

    # -- higher-level calls -------------------------------------------------------------------
    def instance(self) -> dict:
        return self.get("/_kui/api/instance")

    def deployments(self) -> list[dict]:
        return self.get("/_kui/api/workbench/deployments").get("deployments", [])

    def jobs(self, limit: int = 30) -> list[dict]:
        return self.get("/_kui/api/workbench/jobs", limit=limit).get("jobs", [])

    def wait_job(self, job_id: str, *, timeout: float = 900.0) -> dict:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            job = self.get(f"/_kui/api/workbench/jobs/{job_id}")
            if job.get("state") in {"succeeded", "failed", "cancelled", "interrupted"}:
                return job
            time.sleep(1.0)
        raise KuiError(f"job {job_id} did not finish within {int(timeout)}s", code=6)


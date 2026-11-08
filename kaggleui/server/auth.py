"""Credentials and request checks.

* **Admin token** — always required for the control panel. Printed in the notebook as ``/?token=…``.
* **Access password** — optional lock for Studio / Gradio / third-party UIs.
* **API key** — for the OpenAI / Anthropic / Ollama-compatible endpoints.
* **Internal key** — lets our own subprocesses (e.g. the Gradio app) call the gateway.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import secrets
import time
from collections import deque
from pathlib import Path

from starlette.requests import HTTPConnection as Request
from starlette.responses import Response

from ..config import Paths
from ..schemas import Settings
from ..tunnels.persistent import read_record, recorded_process
from ..util import generate_api_key, generate_password

ADMIN_COOKIE = "kui_admin"
ACCESS_COOKIE = "kui_access"
COOKIE_MAX_AGE = 7 * 24 * 3600
MAX_FAILURES = 10
FAILURE_WINDOW = 300


class Auth:
    def __init__(self, secret_path: Path | None = None) -> None:
        previous = {}
        if secret_path is not None:
            record = read_record(Paths(secret_path.parent))
            if recorded_process(record) is not None and isinstance(record.get("credentials"), dict):
                previous = {key: value for key, value in record["credentials"].items()
                            if isinstance(value, str) and value}
        self.admin_token = os.environ.get("KAGGLEUI_ADMIN_TOKEN") or previous.get("admin_token") or secrets.token_urlsafe(18)
        self.password = os.environ.get("KAGGLEUI_PASSWORD") or previous.get("password") or generate_password()
        self.api_key = os.environ.get("KAGGLEUI_API_KEY") or previous.get("api_key") or generate_api_key()
        self.internal_key = secrets.token_urlsafe(32)
        self._secret = secrets.token_bytes(32)
        if secret_path is not None:
            secret_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                fd = os.open(secret_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
                with os.fdopen(fd, "wb") as stream:
                    stream.write(self._secret)
            except FileExistsError:
                value = secret_path.read_bytes()
                if len(value) == 32:
                    self._secret = value
            secret_path.chmod(0o600)
        self._failures: dict[str, deque[float]] = {}

    # -- cookie values ------------------------------------------------------------------------
    def _sig(self, purpose: str, credential: str) -> str:
        return hmac.new(self._secret, f"{purpose}:{credential}".encode(), hashlib.sha256).hexdigest()

    def admin_cookie_value(self) -> str:
        return self._sig("admin", self.admin_token)

    def access_cookie_value(self) -> str:
        return self._sig("access", self.password)

    # -- checks -------------------------------------------------------------------------------
    @staticmethod
    def _eq(a: str | None, b: str) -> bool:
        return bool(a) and hmac.compare_digest(str(a).encode(), b.encode())

    def check_admin_token(self, token: str | None) -> bool:
        return self._eq(token, self.admin_token)

    def check_password(self, password: str | None) -> bool:
        return self._eq(password, self.password)

    @staticmethod
    def _bearer(request: Request) -> str | None:
        auth = request.headers.get("authorization", "")
        if auth.lower().startswith("bearer "):
            return auth[7:].strip()
        return None

    def is_internal(self, request: Request) -> bool:
        """Our own subprocesses: header, or as a Bearer/x-api-key (Open WebUI only knows how to send keys)."""
        if self._eq(request.headers.get("x-kui-internal"), self.internal_key):
            return True
        key = self._bearer(request) or request.headers.get("x-api-key")
        return self._eq(key, self.internal_key)

    def is_admin(self, request: Request) -> bool:
        if self._eq(request.cookies.get(ADMIN_COOKIE), self.admin_cookie_value()):
            return True
        if self._eq(self._bearer(request), self.admin_token):
            return True
        return self.is_internal(request)

    def has_access(self, request: Request, settings: Settings) -> bool:
        if not settings.access.password_enabled:
            return True
        if self.is_admin(request):
            return True
        return self._eq(request.cookies.get(ACCESS_COOKIE), self.access_cookie_value())

    def check_api_key(self, request: Request) -> bool:
        key = self._bearer(request) or request.headers.get("x-api-key")
        return self._eq(key, self.api_key)

    # -- cookies ------------------------------------------------------------------------------
    @staticmethod
    def _secure(request: Request) -> bool:
        return request.headers.get("x-forwarded-proto", request.url.scheme) == "https"

    def set_admin_cookie(self, request: Request, response: Response) -> None:
        response.set_cookie(ADMIN_COOKIE, self.admin_cookie_value(), max_age=COOKIE_MAX_AGE,
                            httponly=True, samesite="lax", secure=self._secure(request), path="/")

    def set_access_cookie(self, request: Request, response: Response) -> None:
        response.set_cookie(ACCESS_COOKIE, self.access_cookie_value(), max_age=COOKIE_MAX_AGE,
                            httponly=True, samesite="lax", secure=self._secure(request), path="/")

    # -- rate limiting ------------------------------------------------------------------------
    @staticmethod
    def client_ip(request: Request) -> str:
        # Never read X-Forwarded-For here: any caller could rotate it and pick a fresh rate-limit
        # bucket. uvicorn (run.py: proxy_headers=True, forwarded_allow_ips="127.0.0.1") already
        # rewrites request.client.host from that header when — and only when — the peer is the
        # local proxy.
        return request.client.host if request.client else "unknown"

    def is_rate_limited(self, ip: str) -> bool:
        q = self._failures.get(ip)
        if not q:
            return False
        cutoff = time.time() - FAILURE_WINDOW
        while q and q[0] < cutoff:
            q.popleft()
        return len(q) >= MAX_FAILURES

    def record_failure(self, ip: str) -> None:
        self._failures.setdefault(ip, deque()).append(time.time())

    # -- rotation -----------------------------------------------------------------------------
    def regenerate(self, which: str) -> None:
        if which == "password":
            self.password = generate_password()
        elif which == "api_key":
            self.api_key = generate_api_key()
        else:
            raise ValueError(which)

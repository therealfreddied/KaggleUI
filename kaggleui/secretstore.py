"""Optional credentials: from Kaggle Secrets (Add-ons → Secrets), the environment, or entered in KaggleUI.

Values live in os.environ (so engines and subprocesses inherit them). Ones entered in KaggleUI are also
kept in ``$KAGGLEUI_HOME/secrets.json`` (mode 600, on the ephemeral disk), so they survive a server
restart within the session. They are never written to /kaggle/working, which Kaggle saves as output.
Values are never returned to the browser or logged; only whether each is set and where it came from.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

SECRET_NAMES = (
    "HF_TOKEN",
    "GITHUB_TOKEN",
    "NGROK_AUTHTOKEN",
    "NGROK_DOMAIN",
    "CF_TUNNEL_TOKEN",
    "CF_TUNNEL_HOSTNAME",
    "KAGGLEUI_PASSWORD",
    "KAGGLEUI_ADMIN_TOKEN",
    "KAGGLEUI_API_KEY",
    "KAGGLEUI_NOTIFY_URL",
)

# Secrets that can be set from the panel. Credentials for KaggleUI itself are managed on the Access page.
EDITABLE = ("HF_TOKEN", "GITHUB_TOKEN", "NGROK_AUTHTOKEN", "NGROK_DOMAIN", "CF_TUNNEL_TOKEN", "CF_TUNNEL_HOSTNAME",
            "KAGGLEUI_NOTIFY_URL")

# Kaggle secret labels are case-sensitive and people name them differently.
ALIASES: dict[str, tuple[str, ...]] = {
    "HF_TOKEN": ("HF_TOKEN", "hf_token", "HUGGINGFACE_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACEHUB_API_TOKEN",
                 "HF_API_TOKEN", "HUGGINGFACE_API_KEY", "huggingface_token", "huggingface", "HF"),
    "GITHUB_TOKEN": ("GITHUB_TOKEN", "GH_TOKEN", "GITHUB_PAT", "github_token", "github"),
    "NGROK_AUTHTOKEN": ("NGROK_AUTHTOKEN", "NGROK_AUTH_TOKEN", "NGROK_TOKEN", "ngrok_token", "ngrok"),
    "KAGGLEUI_NOTIFY_URL": ("KAGGLEUI_NOTIFY_URL", "NTFY_URL", "DISCORD_WEBHOOK", "DISCORD_WEBHOOK_URL"),
}

_sources: dict[str, str] = {}  # name -> "kaggle" | "colab" | "attached" | "env" | "kaggleui"
_attach_status: dict[str, str | None] = {"state": "unknown", "error": None}
_local_path: Path | None = None


def _apply(name: str, value: str, source: str) -> None:
    os.environ[name] = value
    _sources[name] = source
    if name == "HF_TOKEN":
        os.environ["HUGGING_FACE_HUB_TOKEN"] = value


def _attached_reader() -> tuple[str, object] | None:
    """Return ``(source_label, reader)`` for the host's secret store, or ``None`` when unavailable."""
    try:
        from kaggle_secrets import UserSecretsClient  # type: ignore

        return "kaggle", UserSecretsClient()
    except Exception:
        pass
    try:
        from google.colab import userdata  # type: ignore

        return "colab", userdata
    except Exception:
        return None

def load_attached_secrets(refresh: bool = False) -> dict[str, bool]:
    """Copy host-attached secrets into os.environ. ``refresh`` re-reads ones already sourced from it."""
    from_kernel = set(filter(None, os.environ.get("KAGGLEUI_ATTACHED_SECRETS", "").split(",")))
    for name in SECRET_NAMES:
        if os.environ.get(name) and name not in _sources:
            _sources[name] = "attached" if name in from_kernel else "env"
    host = _attached_reader()
    if host is None:
        _attach_status.update(state="unavailable", error="Not running in a hosted notebook")
        return secret_status()
    source, client = host
    _attach_status.update(state="available", source=source, error=None)
    for name in SECRET_NAMES:
        if os.environ.get(name) and not (refresh and _sources.get(name) == source):
            continue
        for label in ALIASES.get(name, (name,)):
            try:
                value = client.get_secret(label) if source == "kaggle" else client.get(label)  # type: ignore[union-attr]
            except Exception:
                continue
            if value and value.strip():
                _apply(name, value.strip(), source)
                break
    return secret_status()


def init_local(home: Path) -> None:
    """Load secrets saved from the panel earlier in this session."""
    global _local_path
    _local_path = home / "secrets.json"
    try:
        saved = json.loads(_local_path.read_text())
    except Exception:
        return
    for name, value in saved.items():
        if name in EDITABLE and value and not os.environ.get(name):
            _apply(name, value, "kaggleui")


def _persist() -> None:
    if not _local_path:
        return
    data = {n: os.environ[n] for n in EDITABLE if _sources.get(n) == "kaggleui" and os.environ.get(n)}
    tmp = _local_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data))
    os.chmod(tmp, 0o600)
    tmp.replace(_local_path)


def set_secret(name: str, value: str, source: str = "kaggleui") -> None:
    if name not in EDITABLE:
        raise KeyError(name)
    _apply(name, value.strip(), source)
    _persist()


def clear_secret(name: str) -> None:
    if name not in EDITABLE:
        raise KeyError(name)
    os.environ.pop(name, None)
    if name == "HF_TOKEN":
        os.environ.pop("HUGGING_FACE_HUB_TOKEN", None)
    _sources.pop(name, None)
    _persist()


def attached_sourced() -> list[str]:
    return [n for n, src in _sources.items() if src in ("kaggle", "colab") and os.environ.get(n)]


def secret_status() -> dict[str, bool]:
    return {name: bool(os.environ.get(name)) for name in SECRET_NAMES}


def secret_details() -> dict:
    return {
        "items": [{"name": n, "set": bool(os.environ.get(n)), "source": _sources.get(n) if os.environ.get(n) else None,
                   "editable": n in EDITABLE} for n in SECRET_NAMES],
        "kaggle": dict(_attach_status),
    }

"""KaggleUI — one-cell control panel for hosting AI models on Kaggle's free GPUs."""

from __future__ import annotations

__version__ = "0.4.3"


def launch(preset=None, *, block: bool = True, tunnel: str | None = None, port: int | None = None,
           mock: bool = False, fake_hw: str | None = None, home: str | None = None,
           workspace: dict | None = None):
    """Start KaggleUI from a notebook cell.

    Starts the server in a background process, prints a link card with the admin token,
    and (when ``block`` is true) keeps the cell alive so batch ("Save & Run All") sessions
    don't end while models are being served.

    ``workspace`` seeds the GitHub-sync settings before the server starts, so a fresh session can
    restore its workspace: ``{"enabled": True, "restore_on_launch": True, "auto_backup": True}``.
    The repository defaults to the ``KAGGLEUI_WORKSPACE_REPO`` secret; the token stays a
    ``secretstore`` reference (``GITHUB_TOKEN`` by default) and is never written to the document.
    """
    from .launcher import launch as _launch

    return _launch(preset=preset, block=block, tunnel=tunnel, port=port, mock=mock, fake_hw=fake_hw,
                   home=home, workspace=workspace)


def stop(home: str | None = None) -> bool:
    """Stop a KaggleUI server started with :func:`launch` (and the models it runs)."""
    from .launcher import stop as _stop

    return _stop(home)


def restart_tunnel(home: str | None = None) -> dict:
    """Reconnect the public URL without restarting running models."""
    from .launcher import restart_tunnel as _restart

    return _restart(home)

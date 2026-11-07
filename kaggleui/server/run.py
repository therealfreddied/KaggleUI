"""Run the KaggleUI server in the foreground."""

from __future__ import annotations

import asyncio
import signal

import uvicorn

from ..config import Paths, default_tunnel
from ..hosts import detect
from ..schemas import Preset, Settings
from ..secretstore import init_local
from .app import create_app
from .state import AppState


def build_state(*, port: int, tunnel: str | None = None, preset: Preset | None = None, mock: bool = False,
                fake_hw: str | None = None, home: str | None = None) -> AppState:
    secrets = detect().load_secrets()
    paths = Paths(home).ensure()
    init_local(paths.home)
    paths.apply_env()
    settings = preset.settings.model_copy(deep=True) if preset else Settings()
    if tunnel:
        settings.exposure.provider = tunnel  # type: ignore[assignment]
    elif not preset:
        settings.exposure.provider = default_tunnel()  # type: ignore[assignment]
    return AppState(paths=paths, settings=settings, port=port, mock=mock, fake_hw=fake_hw, preset=preset,
                    secrets=secrets)


def make_server(state: AppState, host: str = "127.0.0.1") -> uvicorn.Server:
    state.listen_host = host
    app = create_app(state)
    config = uvicorn.Config(app, host=host, port=state.port, log_level="warning", proxy_headers=True,
                            forwarded_allow_ips="127.0.0.1", ws_max_size=64 * 1024 * 1024,
                            timeout_graceful_shutdown=5)
    return uvicorn.Server(config)


async def serve_async(state: AppState, host: str = "127.0.0.1") -> None:
    server = make_server(state, host)
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, lambda: setattr(server, "should_exit", True))
        except (NotImplementedError, RuntimeError):
            pass
    await server.serve()


def serve(state: AppState, host: str = "127.0.0.1") -> None:
    asyncio.run(serve_async(state, host))

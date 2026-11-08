"""Admin terminal over WebSocket.

A pty is the only honest way to hand the panel a shell: pipes would break colours, job control
and anything that asks for a terminal. The pty belongs to this connection and dies with it, so
no shell outlives the tab.
"""

from __future__ import annotations

import asyncio
import os
import signal
import struct
import subprocess
import threading

from fastapi import WebSocket, WebSocketDisconnect

from .state import AppState

SHELL = ("bash", "-l")


def _shell_argv() -> list[str]:
    return ["bash", "-l"]


def _spawn(state: AppState) -> tuple[int, subprocess.Popen] | None:
    import fcntl
    import pty
    import termios

    master, slave = pty.openpty()
    fcntl.ioctl(master, termios.TIOCSWINSZ, struct.pack("HHHH", 24, 100, 0, 0))
    env = dict(os.environ)
    env.update({"TERM": "xterm-256color", "KAGGLEUI_HOME": str(state.paths.home), "COLORTERM": "truecolor"})
    try:
        proc = subprocess.Popen(_shell_argv(), stdin=slave, stdout=slave, stderr=slave,
                                cwd=str(state.paths.home), env=env, start_new_session=True)
    except OSError:
        os.close(master)
        os.close(slave)
        return None
    os.close(slave)
    return master, proc


async def console_ws(ws: WebSocket, state: AppState) -> None:
    if not state.auth.is_admin(ws):  # type: ignore[arg-type]
        await ws.close(code=4401)
        return
    try:
        spawned = _spawn(state)
    except ImportError:
        spawned = None
    if not spawned:
        await ws.accept()
        await ws.close(code=4403, reason="A terminal needs a POSIX pty; this host has none.")
        return
    master, proc = spawned
    await ws.accept()

    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[bytes | None] = asyncio.Queue()

    def pump() -> None:
        try:
            while True:
                data = os.read(master, 65536)
                if not data:
                    break
                loop.call_soon_threadsafe(queue.put_nowait, data)
        except OSError:
            pass
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, None)

    threading.Thread(target=pump, name="kui-console", daemon=True).start()
    stop = asyncio.Event()

    async def send_terminal() -> None:
        while True:
            data = await queue.get()
            if data is None:
                await ws.send_json({"op": "exit", "code": proc.poll()})
                return
            await ws.send_bytes(data)

    loop = asyncio.get_running_loop()

    async def read_client() -> None:
        while True:
            msg = await ws.receive_json()
            op = msg.get("op")
            if op == "input":
                text = msg.get("data") or ""
                if text:
                    # A shell that stops reading would otherwise block the whole server on a big paste.
                    await loop.run_in_executor(None, os.write, master, text.encode())
            elif op == "resize":
                cols, rows = int(msg.get("cols") or 80), int(msg.get("rows") or 24)
                import fcntl
                import termios

                fcntl.ioctl(master, termios.TIOCSWINSZ, struct.pack("HHHH", rows, cols, 0, 0))

    sender = asyncio.create_task(send_terminal())
    reader = asyncio.create_task(read_client())
    try:
        await asyncio.wait({sender, reader}, return_when=asyncio.FIRST_COMPLETED)
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        stop.set()
        for task in (sender, reader):
            task.cancel()
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except OSError:
            pass
        try:
            await asyncio.to_thread(proc.wait, 2)
        except subprocess.TimeoutExpired:
            proc.kill()
        try:
            os.close(master)
        except OSError:
            pass

"""Tell the host where KaggleUI is: a notebook link card and an optional push notification.

Push notifications matter for "Save & Run All" (batch) sessions, where you can't watch the cell output.
Set a KAGGLEUI_NOTIFY_URL secret to an ntfy.sh topic URL or a Discord webhook URL.
"""

from __future__ import annotations

import html
import os

import httpx


def panel_link(url: str, admin_token: str) -> str:
    return f"{url.rstrip('/')}/?token={admin_token}"


async def push(url: str, admin_token: str) -> str | None:
    target = os.environ.get("KAGGLEUI_NOTIFY_URL")
    if not target:
        return None
    link = panel_link(url, admin_token)
    text = f"KaggleUI is up: {link}"
    try:
        async with httpx.AsyncClient(timeout=10) as c:
            if "discord.com/api/webhooks" in target or "discordapp.com/api/webhooks" in target:
                r = await c.post(target, json={"content": text})
            elif "ntfy" in target:
                r = await c.post(target, content=text.encode(), headers={"Title": "KaggleUI is up", "Click": link,
                                                                          "Tags": "rocket"})
            else:
                r = await c.post(target, json={"text": text, "url": link})
        return f"sent ({r.status_code})"
    except httpx.HTTPError as e:
        return f"failed: {type(e).__name__}"


def card_html(url: str, admin_token: str, password: str | None, api_key: str | None, note: str = "",
              guidance: list[str] | None = None) -> str:
    link = panel_link(url, admin_token)
    rows = [("Control panel", f'<a href="{html.escape(link)}" target="_blank" style="color:#1f6feb;font-weight:600">'
             f'{html.escape(url)}</a>')]
    rows.append(("Admin token", f"<code>{html.escape(admin_token)}</code>"))
    if password:
        rows.append(("Access password", f"<code>{html.escape(password)}</code>"))
    if api_key:
        rows.append(("API key", f"<code>{html.escape(api_key)}</code>"))
    body = "".join(f'<tr><td style="padding:4px 14px 4px 0;color:#667;white-space:nowrap">{k}</td><td>{v}</td></tr>'
                   for k, v in rows)
    hints = "".join(f'<div style="margin-top:2px;color:#667;font-size:12px">{html.escape(line)}</div>'
                    for line in guidance or [])
    extra = (f'<div style="margin-top:8px;color:#667;font-size:12px">{html.escape(note)}</div>' if note else "") + hints
    return ('<div style="font-family:system-ui,sans-serif;border:1px solid #d0d7de;border-radius:10px;padding:14px 16px;'
            'max-width:760px;font-size:14px"><div style="font-weight:700;margin-bottom:8px">KaggleUI is running</div>'
            f'<table style="border-collapse:collapse">{body}</table>{extra}</div>')


def card_text(url: str, admin_token: str, password: str | None, api_key: str | None,
              guidance: list[str] | None = None) -> str:
    lines = ["", "=" * 64, " KaggleUI is running", "=" * 64,
             f" Open:            {panel_link(url, admin_token)}",
             f" Admin token:     {admin_token}"]
    if password:
        lines.append(f" Access password: {password}")
    if api_key:
        lines.append(f" API key:         {api_key}")
    lines += [f" {line}" for line in guidance or []]
    lines += ["=" * 64, ""]
    return "\n".join(lines)

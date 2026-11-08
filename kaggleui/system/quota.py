"""Measured accelerator time, budgeted by the user.

Kaggle exposes no weekly GPU-hours API (``ListKernelSessions`` only carries ``cpuInteractive``), so the
ledger records what actually runs here: one interval per server lifetime, tagged with the accelerator the
probe reported. Idle periods between servers are not counted, which keeps the number a floor, never an
overstatement.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

KEEP_S = 8 * 7 * 86400

def week_start(ts: float | None = None) -> float:
    """Monday 00:00 local time of the week containing ``ts``."""
    day = datetime.fromtimestamp(ts if ts is not None else time.time())
    monday = day - timedelta(days=day.weekday())
    return monday.replace(hour=0, minute=0, second=0, microsecond=0).timestamp()

@dataclass
class QuotaLedger:
    path: Path
    _open_start: float | None = field(default=None, repr=False)

    def _read(self) -> list[dict]:
        try:
            rows = json.loads(self.path.read_text())
        except (OSError, ValueError):
            return []
        if not isinstance(rows, list):
            return []
        return [row for row in rows if isinstance(row, dict) and row.get("accelerator")
                and isinstance(row.get("start"), (int, float))]

    def _write(self, rows: list[dict]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(rows))
        tmp.replace(self.path)

    def begin(self, accelerator: str, *, at: float | None = None) -> None:
        """Open an interval for this server lifetime; a crash leaves it open and still counts."""
        now = time.time() if at is None else at
        rows = self._read()
        for row in rows:
            if row.get("end") is None:
                row["end"] = now
        cutoff = now - KEEP_S
        rows = [row for row in rows if row["start"] >= cutoff]
        rows.append({"start": now, "end": None, "accelerator": accelerator})
        self._open_start = now
        self._write(rows)

    def end(self, *, at: float | None = None) -> None:
        now = time.time() if at is None else at
        rows = self._read()
        for row in rows:
            if row.get("end") is None:
                row["end"] = max(row["start"], now)
        self._open_start = None
        self._write(rows)

    def used(self, accelerator: str, since: float, *, until: float | None = None) -> float:
        """Seconds of ``accelerator`` time inside the half-open window ``[since, until]``."""
        now = time.time() if until is None else until
        total = 0.0
        for row in self._read():
            if row["accelerator"] != accelerator:
                continue
            start = max(float(row["start"]), since)
            end = min(float(row["end"]) if row.get("end") is not None else now, now)
            if end > start:
                total += end - start
        return total

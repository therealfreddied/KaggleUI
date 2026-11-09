"""Rate-limit bookkeeping."""

import time
from collections import deque

from kaggleui.server.auth import FAILURE_WINDOW, Auth


def test_record_failure_prunes_entries_older_than_the_window():
    auth = Auth()
    stale = time.time() - FAILURE_WINDOW - 10
    auth._failures["1.2.3.4"] = deque(stale for _ in range(50))
    auth.record_failure("1.2.3.4")
    assert len(auth._failures["1.2.3.4"]) == 1

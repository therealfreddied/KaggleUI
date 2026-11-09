"""ManagedProcess line pump edge cases."""

import sys

from kaggleui.proc import run_cmd


async def test_oversized_line_does_not_eat_later_output():
    lines: list[str] = []
    code = ("import sys\n"
            "sys.stdout.write('x' * 200000 + '\\n')\n"
            "sys.stdout.flush()\n"
            "print('MARKER')\n"
            "sys.stdout.flush()\n")
    rc = await run_cmd([sys.executable, "-c", code], on_line=lines.append)
    assert rc == 0
    assert len(lines[0]) == 200000
    assert lines[-1] == "MARKER"

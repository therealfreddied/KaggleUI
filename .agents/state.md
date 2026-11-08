# State

Updated: 2026-11-08

## Goal

Fix the five concurrency / input-validation bugs (F1–F5) documented by the staged strict-`xfail`
tests. Complete: all five fixed, markers removed, suite green except the three environmental
`test_shutdown.py` failures.

## Progress

- F1 in-process job cancel: `JobStore._locals` + `attach_local`/`detach_local`; `cancel()` cancels and
  awaits the task. `runner.start_local` registers it.
- F2 `JobStore.update`: read + write under one lock (SELECT inside `self._lock`).
- F3 gateway bodies: `gateway/common.require_messages` → 400 from `/v1/chat/completions` and
  `/v1/messages` before translation.
- F4 lockout evasion: `Auth.client_ip` returns `request.client.host`; forwarded headers are only
  honoured through uvicorn's trusted-proxy handling.
- F5 registry index: `ArtifactRegistry._lock` around the index cycle; unique `mkstemp` temp file
  (also applied to `remove`).
- Removed a dead `LLAMACPP_RELEASE` import in `ml/pipeline._engine_name` (was failing `ruff check`).
- Tests: F2/F5 harnesses rewritten (sleep in the read path instead of a barrier that would deadlock a
  correct lock); F1/F3/F4 kept as written, `xfail` markers dropped.
- Verification: `ruff check kaggleui tests` → `All checks passed!`; affected modules →
  `33 passed in 57.21s`; full suite → `3 failed, 355 passed, 1 skipped in 138.40s` (only
  `psutil.AccessDenied` in `test_shutdown.py`). `docs/implementation/progress.md` appended.

## Blockers

- None for code work. Host limits unchanged (no `.venv`, python3 3.14.8, `/tmp` unwritable, no
  android-arm64 `tsc` binary).

## Next actions

1. Commit the work (six code files + four test modules + `ml/pipeline.py` + progress/bank notes);
   nothing is committed yet.
2. Consider a `0.4.2` patch bump (`kaggleui/__init__.py`, `pyproject.toml`, `web/package.json`, then
   `pnpm build`) if these fixes ship as a release.
3. Not run this session: `selftest --mock-engines --fake-hw t4x2` and `vite build`.

# State

Updated: 2026-11-08

## Goal

No active feature request. `0.4.2` released from `main`: the five concurrency / input-validation
fixes (F1–F5) plus the version bump and rebuilt wheel.

## Progress

- `10af3f4` — fixes: in-process job cancel (`JobStore._locals` + `attach_local`/`detach_local`),
  `JobStore.update` reads and writes under one lock, `gateway/common.require_messages` → 400 for
  malformed bodies on `/v1/chat/completions` and `/v1/messages`, `Auth.client_ip` returns
  `request.client.host` (forwarded headers trusted only through uvicorn's local proxy), artifact
  `index.json` cycle locked with a unique `mkstemp` temp. Dead `LLAMACPP_RELEASE` import removed.
- Version `0.4.2` in `kaggleui/__init__.py`, `pyproject.toml`, `web/package.json`; bundle rebuilt;
  `dist/kaggleui-0.4.2-py3-none-any.whl` built (51 static assets, bundle stamped `0.4.2`).
- Verification: `ruff check kaggleui tests` → `All checks passed!`; affected modules →
  `33 passed in 57.21s`; full suite → `3 failed, 355 passed, 1 skipped in 138.40s` (only
  `psutil.AccessDenied` in `test_shutdown.py`);
  `python3 -m pytest -q tests/test_release_versions.py` → `3 passed`.

## Blockers

- None for code work. Host limits unchanged: no `.venv`, `python3` 3.14.8, `/tmp` unwritable,
  `pnpm build` fails on the missing android-arm64 `tsc` binary — build with
  `cd web && pnpm exec vite build`, typecheck separately with the `dlx tsc@5.9.3` workaround.
  Under that workaround `web/src/pages/{Access,Storage,System}.tsx` report pre-existing type errors
  (unused `claudeShell`/`SectionTitle`, `System.tsx:35` missing argument); the files are untouched
  by this release and CI's bundled `tsc` 7.0.2 has not been run here.

## Next actions

1. Nothing pending. Next session: run `ruff check kaggleui` and the narrowest relevant `pytest`
   module before editing to confirm the host baseline.
2. Keep `memory.md` append-only; rewrite this file at session end (`/sync`).
3. Append milestone-grade results to `docs/implementation/progress.md`, not here.

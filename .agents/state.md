# State

Updated: 2026-11-08

## Goal

No active feature request. `0.4.2` is released from local `main` (`0c6ccf0`, tag `v0.4.2`) and pushed
to the `self` remote. The five concurrency / input-validation defects (F1–F5) documented by the staged
strict-`xfail` tests are fixed, committed and tagged.

## Progress

- `10af3f4` — fixes: `JobStore.cancel` now cancels the task of in-process jobs (`_locals` +
  `attach_local`/`detach_local`, registered by `runner.start_local`); `JobStore.update` reads and
  writes under one lock; `gateway/common.require_messages` returns 400 for malformed bodies on
  `/v1/chat/completions` and `/v1/messages`; `Auth.client_ip` returns `request.client.host` (forwarded
  headers trusted only via uvicorn's local-proxy handling); artifact `index.json` cycle locked with a
  unique `mkstemp` temp (also in `remove`). Dead `LLAMACPP_RELEASE` import removed from `ml/pipeline.py`.
- `0c6ccf0` — `0.4.2` in `kaggleui/__init__.py`, `pyproject.toml`, `web/package.json`; SPA bundle
  rebuilt; `dist/kaggleui-0.4.2-py3-none-any.whl` built (51 static assets, bundle stamped `0.4.2`);
  tag `v0.4.2`.
- Pushed: `git push self main v0.4.2` → `328aa68..0c6ccf0`, new tag `v0.4.2` on
  `therealfreddied/KaggleUI`. `origin` (UltraFEmotes) refused: 403, credential is `therealfreddied`.
- Verification: `ruff check kaggleui tests` → `All checks passed!`; affected modules →
  `33 passed in 57.21s`; full suite → `3 failed, 355 passed, 1 skipped in 138.40s` (only
  `psutil.AccessDenied` in `test_shutdown.py`); `tests/test_release_versions.py` → `3 passed`.

## Blockers

- `origin` push needs write access for `therealfreddied` (or a token/SSH key for UltraFEmotes).
- Host limits: no `.venv`, `python3` 3.14.8, `/tmp` unwritable, no android-arm64 `tsc` (use
  `cd web && pnpm exec vite build` + `pnpm --package=typescript@5.9.3 dlx tsc -p . --noEmit`).

## Next actions

1. Resolve `origin` access, then `git push origin main` (and tags `v0.3.*`, `v0.4.0`, `v0.4.1`,
   `v0.4.2` if they should be published).
2. Run the unrun release checks: `selftest --mock-engines --fake-hw t4x2 --port 8899 --home
   "$HOME/.cache/kui"`, and confirm CI's `tsc` 7.0.2 build on a host that supports it
   (`Access/Storage/System.tsx` report errors under the 5.9.3 workaround).
3. Upload `dist/kaggleui-0.4.2-py3-none-any.whl` if a wheel artifact is wanted on GitHub/PyPI.
4. Keep `memory.md` append-only; rewrite this file at session end (`/sync`). Milestone history goes to
   `docs/implementation/progress.md`.

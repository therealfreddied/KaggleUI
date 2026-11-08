# State

Updated: 2026-11-08

## Goal

No active feature request. `0.4.2` is released from local `main` and published to the only repo that
matters — the personal fork `therealfreddied/KaggleUI` (`self`). Upstream `UltraFEmotes/KaggleUI`
(`origin`) is not owned by this account; nothing is expected to land there.

## Progress

- `10af3f4` — fixes for F1–F5: `JobStore.cancel` cancels the task of in-process jobs (`_locals` +
  `attach_local`/`detach_local`, registered by `runner.start_local`); `JobStore.update` reads and
  writes under one lock; `gateway/common.require_messages` → 400 for malformed bodies on
  `/v1/chat/completions` and `/v1/messages`; `Auth.client_ip` returns `request.client.host` (forwarded
  headers trusted only through uvicorn's local proxy); artifact `index.json` cycle locked with a unique
  `mkstemp` temp (also in `remove`); dead `LLAMACPP_RELEASE` import dropped from `ml/pipeline.py`.
- `0c6ccf0` — `0.4.2` in `kaggleui/__init__.py`, `pyproject.toml`, `web/package.json`; SPA bundle
  rebuilt; `dist/kaggleui-0.4.2-py3-none-any.whl` built (51 static assets, bundle stamped `0.4.2`);
  tag `v0.4.2`.
- Published to the fork: `main` through `6b9f581` (bank updates included) and tag `v0.4.2`.
  `git config branch.main.pushRemote self`, so plain `git push --follow-tags` targets the fork;
  `--dry-run` → `Everything up-to-date`.
- Verification: `ruff check kaggleui tests` → `All checks passed!`; affected modules →
  `33 passed in 57.21s`; full suite → `3 failed, 355 passed, 1 skipped in 138.40s` (only
  `psutil.AccessDenied` in `test_shutdown.py`); `tests/test_release_versions.py` → `3 passed`.

## Blockers

- None. Host limits only: no `.venv`, `python3` 3.14.8, `/tmp` unwritable, no android-arm64 `tsc`
  (build with `cd web && pnpm exec vite build`, typecheck with
  `pnpm --package=typescript@5.9.3 dlx tsc -p . --noEmit`).

## Next actions

1. Run the release checks that were skipped: `selftest --mock-engines --fake-hw t4x2 --port 8899
   --home "$HOME/.cache/kui"`; confirm the CI `tsc` 7.0.2 build on a supported host
   (`Access/Storage/System.tsx` error under the 5.9.3 workaround).
2. Upload `dist/kaggleui-0.4.2-py3-none-any.whl` to a GitHub release on the fork if a wheel artifact
   is wanted; `dist/` is gitignored so it is local-only today.
3. Keep `memory.md` append-only; rewrite this file at session end (`/sync`). Milestone history goes to
   `docs/implementation/progress.md`.

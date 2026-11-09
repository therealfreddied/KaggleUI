# State

Updated: 2026-11-09

## Goal

No active feature request. `0.4.3` (bug-hunt fixes F1–F8) is released from local `main` and
published to the personal fork `therealfreddied/KaggleUI` (`self`), tag `v0.4.3`, wheel attached to
the GitHub release. Upstream `UltraFEmotes/KaggleUI` (`origin`) is read-only for this account.

## Progress

- Bug hunt (4 confirmed): F1 `ManagedProcess._pump` killed by any child line >64 KiB
  (`LimitOverrunError`, silent output loss + child deadlock); F2 `GitHubProvider.download` saved the
  contents-API JSON envelope instead of file bytes; F3 non-numeric `thinking.budget_tokens` → 500;
  F4 corrupt registry `index.json` + next `add()` orphaned all artifacts. Plus F5–F8 (bounded
  `read_log`/`tail_log`, `start_local` cancel race, auth failure-deque pruning, backup restore
  status labels).
- `a2923cd` — all fixes + version `0.4.3` in `kaggleui/__init__.py`, `pyproject.toml`,
  `web/package.json`; SPA rebuilt via `pnpm exec vite build` (bundle stamped 0.4.3); 4 xfail
  markers flipped to active tests; new `tests/test_proc.py`, `tests/test_auth.py`, `tail_log` test.
- Verification: `ruff check kaggleui tests` clean; affected suites `58 passed`; full suite
  `354 passed, 1 skipped` (`test_shutdown.py` excluded, Android `psutil` artifact); mock selftest
  `PASSED`; `dist/kaggleui-0.4.3-py3-none-any.whl` (76 static assets).
- Published: `git push --follow-tags` → `b49e9ca..a2923cd main`, tag `v0.4.3`;
  `gh release create v0.4.3` on the fork with the wheel attached (verified via `gh release view`).
- `install.sh` in the repo root was replaced by an unrelated claude-code-android installer
  (foreign, uncommitted). Excluded from the 0.4.3 commit; left in the working tree untouched.

## Blockers

- None. Host limits only: no `.venv`, `python3` 3.14.8, `/tmp` unwritable, no android-arm64 `tsc`
  (build with `cd web && pnpm exec vite build`, typecheck with
  `pnpm --package=typescript@5.9.3 dlx tsc -p . --noEmit`).

## Next actions

1. Decide what to do about the foreign `install.sh` in the working tree (it clobbers the real
   installer referenced by the README if ever committed).
2. Confirm the CI `tsc` 7.0.2 build on a supported host (`Access/Storage/System.tsx` error under
   the 5.9.3 workaround).
3. Keep `memory.md` append-only; rewrite this file at session end (`/sync`). Milestone history goes
   to `docs/implementation/progress.md`.

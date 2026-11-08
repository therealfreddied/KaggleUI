# KaggleUI — durable memory

Repository rules (conventions, recipes, sharp edges) live in the root `AGENTS.md`; read it first.
This file records what that document does not: host environment, decisions with reasons, and
failure modes already paid for.

## Tech stack (verified 2026-11-08)

- Python package `kaggleui/` (≥3.10, CI 3.11), FastAPI server + subprocess engines. Wheels shipped
  in `dist/` up to `0.4.1`; `__version__ = "0.4.1"`.
- React 19 SPA in `web/`, Vite, TypeScript, Tailwind v4, built into `kaggleui/static/` (gitignored).
- 47 test modules in `tests/`; pytest with `asyncio_mode = "auto"`.
- CI (`.github/workflows/ci.yml`): `pnpm install --frozen-lockfile && pnpm build` → `uv venv --python
  3.11 && uv pip install -e ".[dev]"` → `ruff check kaggleui` → `pytest -q` → `selftest
  --mock-engines --fake-hw t4x2 --port 8899` → `uv build --wheel`.

## Architecture invariants

- `AppState` (`server/state.py`) is the only process-wide container; routes read
  `request.app.state.kui`.
- Engines run as supervised subprocesses tracked on `ctx.processes` (`ManagedProcess`). Never
  `pkill` by name; only recorded handles are stopped.
- Workers (`engines/workers/*.py`) import only fastapi/uvicorn/pydantic and run inside isolated `uv`
  venvs; never import heavy stacks into the server process.
- Missing capability → `Unsupported`; deployment failure → `DeployError`. Gateway maps
  `Unsupported` to HTTP 400 with a user-facing message. No tracebacks in API responses.
- `web/src/lib/types.ts` mirrors `kaggleui/schemas.py` by hand — both sides change together.
- Version bump touches `kaggleui/__init__.py`, `pyproject.toml`, `web/package.json`, then
  `pnpm build`; `tests/test_release_versions.py` enforces agreement with the built bundle.
- Mock parity is part of the contract: `--mock-engines` + `--fake-hw` are what CI and local dev use.

## Host environment (this Termux/Android arm64 workstation)

- No `.venv` here. `python3` is **3.14.8**; `ruff` is `/data/data/com.termux/files/usr/bin/ruff`
  (0.17.3). `node` v24.22.2, `pnpm` 12.16.1.
- `/tmp` is not writable → pass `--home $HOME/.cache/kui` (any dir under `$HOME`) to the server and
  selftest, or set `KAGGLEUI_HOME`.
- `_write_state` / `test_shutdown.py` fail with `psutil.AccessDenied` reading foreign `/proc`
  entries under Android; `test_shutdown.py` timeouts are host artifacts, not regressions.
- `tsc` on this host: `pnpm --package=typescript@5.9.3 dlx tsc -p web --noEmit` works around the
  missing android-arm64 binary; `vite build` produces the bundle regardless.
- Files written by tooling can lack a trailing newline → `ruff check --fix` (W292) after writing.

## Gotchas paid for

- `write`-created files without trailing newline trip W292; run `ruff check --fix kaggleui`.
- `util.wait_for(lambda: await …)` silently passes a coroutine and never succeeds — wrap async
  predicates in a nested `async def`.
- `kaggleui/static/` is generated; hand edits are lost on the next `pnpm build`.
- Weights/venvs/logs must stay under `KAGGLEUI_HOME`; `/kaggle/working` is capped at 20 GB of saved
  output.

## 2026-11-08 — concurrency + gateway input hardening (session 2)

- `JobStore.update` is read-modify-write over the whole row: it MUST hold `self._lock` across the
  SELECT and the UPDATE. Reading through `get()` first silently restored stale `cancel_requested`
  or `progress` from a concurrent writer.
- In-process jobs (`runner.start_local`) have no child process; `JobStore.cancel` only interrupted
  subprocess jobs until the task registry (`attach_local`/`detach_local`, `asyncio.Task.cancel()` +
  `gather`) was added. Any new in-process job path must go through `start_local`, not a bare
  `create_task`.
- `Auth.client_ip` MUST NOT read `X-Forwarded-For`/`cf-connecting-ip`/`X-Real-IP`: the login
  lockout keys on it, so a caller rotating the header picks a fresh bucket. Trust comes from uvicorn
  (`run.py`: `proxy_headers=True`, `forwarded_allow_ips="127.0.0.1"`), which rewrites
  `request.client.host` only when the peer is the local proxy. Do not "fix" this by re-reading headers.
- `ArtifactRegistry` serialises its `index.json` read-modify-write under `self._lock` and writes via
  `tempfile.mkstemp` in the registry root (a fixed `<name>.tmp` raced with itself).
- Gateway input validation lives in `gateway/common.require_messages` (400 `GatewayError` before
  translation); `openai.chat_completions` and `anthropic._body` both call it. Malformed bodies used to
  reach the translation layer and surface as 500.
- Test-harness note: a `threading.Barrier` that fires *inside* the read path cannot be used to test a
  correct lock — the second thread blocks on the lock and the barrier times out. Use a sleep in the
  read path instead (it widens the unlocked window and still passes once serialised).
- Host baseline for the full suite now: `3 failed, 355 passed, 1 skipped` — the three are
  `tests/test_shutdown.py::…` with `psutil.AccessDenied` (Android `/proc` restriction), not regressions.

## 2026-11-08 — 0.4.2 release mechanics (host)

- `cd web && pnpm build` is broken on this host (TypeScript 7.0.2 has no android-arm64 binary).
  Build the bundle with `cd web && pnpm exec vite build` — vite itself is fine and stamps
  `__APP_VERSION__` from `web/package.json`; typecheck separately with
  `pnpm --package=typescript@5.9.3 dlx tsc -p . --noEmit`.
- Under that 5.9.3 typecheck, `web/src/pages/{Access,Storage,System}.tsx` report pre-existing errors
  (unused `claudeShell`, unused `SectionTitle`, `System.tsx:35` missing argument / `.note` access).
  Untouched by 0.4.2; CI uses the bundled tsc 7.0.2, which has not been run on this host.
- `uv build --wheel` needs a writable TMPDIR here: `TMPDIR="$HOME/.cache/kui-tmp" uv build --wheel`.
  Wheel output goes to `dist/` (gitignored); bundles live in the gitignored `kaggleui/static/`.
  Verify a wheel carries the bundle with
  `python3 -c "import zipfile; z=zipfile.ZipFile('dist/kaggleui-0.4.2-py3-none-any.whl'); …"`.

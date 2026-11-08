# KaggleUI agent entrypoint

Load this bank before touching code:

@memory.md
@state.md

The **root `AGENTS.md` is authoritative** for architecture, conventions, change recipes, and sharp
edges. This file only adds the host-verified commands.

## Commands (this Termux/Android host — no `.venv`)

```bash
ruff check kaggleui                       # /data/data/com.termux/files/usr/bin/ruff 0.17.3
python3 -m pytest -q tests/test_fit.py    # python3 is 3.14.8; expect psutil.AccessDenied in
                                          # test_shutdown.py and _write_state paths
pnpm --package=typescript@5.9.3 dlx tsc -p web --noEmit
cd web && pnpm build                      # emits into kaggleui/static (gitignored)
python3 -m kaggleui selftest --mock-engines --fake-hw t4x2 --port 8899 --home "$HOME/.cache/kui"
```

`/tmp` is unwritable: always pass `--home` or set `KAGGLEUI_HOME`.

## Rules of engagement

- Bump versions in `kaggleui/__init__.py`, `pyproject.toml`, `web/package.json`, then `pnpm build`.
- Never hand-edit `kaggleui/static/`; never import torch at module import time; never install into
  the system/Kaggle environment — heavy stacks go into per-engine `uv` venvs.
- Run `ruff check --fix` after writing files that may lack a trailing newline.
- Rewrite `state.md` at session end; append durable facts to `memory.md`.

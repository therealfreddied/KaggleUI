# Memory bank

Persistent project memory for KaggleUI. Entry point: `.agents/AGENTS.md`.
Repository rules live in the root `AGENTS.md` — that file is authoritative for conventions; this
bank records decisions, environment facts, and current state.

## Files

- `memory.md` — durable knowledge: tech stack, architecture invariants, environment, gotchas.
  Append `## <YYYY-MM-DD> — <topic>` sections. Never silently delete prior entries; supersede by
  adding a newer entry that names the old one.
- `state.md` — always-current operational state: Goal, Progress, Blockers, Next actions. Rewrite
  (not append) at session start and end.

## Maintenance rules

- `/boot` reads `README.md` → `memory.md` → `state.md` when resuming.
- `/sync` before ending a session: rewrite `state.md`, append anything durable to `memory.md`.
- Record only facts that cost time to rediscover: a green/red check with the exact command, a
  host quirk, a design decision and the reason, a sharp edge a test cannot express.
- Verification lines must be copied verbatim from an actual run. Unrun checks are marked `-`
  rather than guessed.
- Milestone-by-milestone history stays in `docs/implementation/progress.md` (append-only); this
  bank holds what a reader needs before touching code.

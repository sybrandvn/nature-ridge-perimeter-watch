# Repository guidance

## Start here

- [README.md](README.md): repository overview and development commands.
- [scripts/README.md](scripts/README.md): current tool groups and invocation syntax.
- [tests/README.md](tests/README.md): test placement, fixtures and validation.
- [data/reports/README.md](data/reports/README.md): authoritative report layout and examples.
- [data/README.md](data/README.md): other local data and stable runtime paths.
- [docs/next_agent_handoff.md](docs/next_agent_handoff.md): current detector and deployment context.

`docs/handoff.md`, `docs/plan.md` and dated reviews include historical instructions and paths.
Use the current guides above for repository conventions and commands.

## Code and verification

- Work from the repository root. Run tools as `uv run python -m scripts.<group>.<tool>`.
- Keep reusable logic in `src/`, operator tools in their existing `scripts/` groups,
  application tests in `tests/unit/`, and tool tests in matching `tests/scripts/` groups.
- `src/` must never import `scripts/`. Shared versioned fixture records live in `tests/fixtures/`.
- Use `make check` for Python changes; `make test-unit` and `make test-scripts` support focused runs.
  For documentation-only changes, check links and `git diff --check`; a full test run is unnecessary.
- Keep feature work, structural moves and unrelated existing work in separate logical commits.

## Reports and local state

- Follow the implemented layout in the reports guide and current CLI defaults. A layout
  recommendation discussed in chat does not mean the filesystem has already been migrated.
- Keep one authoritative layout definition in the reports guide. Other documents should link
  to it; examples and output defaults must agree with it.
- When changing the layout, move the actual ignored report folders and update CLI defaults,
  affected tests and maintained path references together. Verify moved files with checksums
  and retain a local move manifest under the report archive.
- Keep each investigation's evidence together. Preserve historical artifact contents and
  provenance; use move manifests to resolve old paths.
- Local artifacts, databases, sessions and secrets stay outside Git. Only the existing data
  guides and empty-directory placeholders are versioned under `data/`.
- Preserve database, session, history, live-media and production-reference paths used by
  configuration and persisted rows. A repository cleanup does not require starting or
  deploying the watcher.

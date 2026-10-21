# Testing

Run from the repository root after `uv sync --locked` or `make setup`.

| Location | Coverage |
| --- | --- |
| `unit/` | Reusable `src/` modules, detector rules, persistence, bot handlers and the import boundary |
| `scripts/analysis/` | Backtests, candidate ranking and offline analysis orchestration |
| `scripts/ingest/` | Downloads and media recovery |
| `scripts/review/` | Labelling, debug rendering and review visuals |
| `scripts/ops/` | Watcher entry point and container readiness, including an H.264 round trip |
| `scripts/migrations/` | Database schema upgrades |
| `fixtures/` | Small versioned regression records shared by tests and offline checks |

```bash
make check                          # Ruff and all automated tests
make test-unit                      # application/domain tests
make test-scripts                   # operator-tool tests
make test PYTEST_ARGS="-k warmup -x" # focus on a behavior
uv run pytest -q tests/scripts/review/test_render_debug.py
```

`make test` is equivalent to `uv run pytest -q`. Pytest discovers both groups recursively
through `pyproject.toml`. Tests use temporary databases, synthetic media and fake transports;
they do not require `.env`, a Telegram session or the local history corpus.

Keep automated tests here and sizeable/generated media in `data/`. Share fixture records in
`tests/fixtures/` rather than copying them into each group. The vegetation prototype's unit
tests include small offline-proposal helpers alongside the reusable evidence calculations.

Real-footage checks are separate operator commands because their inputs are local and ignored:

```bash
uv run python -m scripts.analysis.check_incident_regression
uv run python -m scripts.analysis.check_environment_hypothesis --help
```

See the [operator guide](../docs/operator-guide.md) and
[current handoff](../docs/next_agent_handoff.md) for corpus preservation requirements and
known baseline issues. These checks are not part of `make check`.

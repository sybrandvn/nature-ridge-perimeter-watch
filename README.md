# Nature Ridge Perimeter Watch

Motion-based perimeter monitoring from a Telegram security-camera feed. The system ingests
camera clips, detects and classifies motion, resolves related events, and delivers alerts and
role-restricted bot queries.

## Getting started

Requires Python 3.12+, [uv](https://docs.astral.sh/uv/) and Make. Run commands from the repository root:

```bash
make setup
cp .env.example .env   # fill in credentials and environment-specific settings
make check
```

Without Make, use `uv sync --locked`, `uv run ruff check .` and `uv run pytest -q`.
Camera geometry and detector thresholds live in `config/`; environment settings live in `.env`.
Local source footage, database state, sessions and reports live in `data/` and are ignored by Git.

## Structure

```text
config/                 Camera roster, geometry and thresholds
src/                    Reusable application and detector code
scripts/
  ingest/               Telegram setup, backfill, downloads and recovery
  analysis/             Backtests, calibration, diagnostics and offline experiments
  review/               Labelling, debug renders and visual review
  ops/                  Watcher, health checks and operational tools
  migrations/           Explicit upgrades for existing databases
tests/
  unit/                 Application and domain tests
  scripts/              Tool tests, grouped like scripts/
  fixtures/             Small versioned regression records
data/                   Local state, media, reports, backups and logs
docs/                   Operator guide, deployment, current handoff and historical findings
Makefile                Setup, checks and test commands
pyproject.toml          Dependencies, pytest and Ruff configuration
uv.lock                 Locked dependency versions
Dockerfile              Runtime image
compose.yaml            Watcher, toolbox and session bootstrap services
```

Add reusable logic to `src/`, commands to the appropriate script group, and tests to the matching
test location. `src/` must not import `scripts/`; an architecture test enforces this boundary.
The [script index](scripts/README.md), [testing guide](tests/README.md) and
[data guide](data/README.md) describe these conventions in detail.

## Development

```bash
make check                           # lint and all automated tests
make test-unit                       # application/domain tests
make test-scripts                    # operator-tool tests
make test PYTEST_ARGS="-k warmup -x"  # a focused run
make help                            # all development commands
```

Automated tests use temporary databases, synthetic media and fake transports. They run without
Telegram credentials or downloaded footage. Real-corpus validation is documented separately
in the testing and operator guides.

## Running tools and services

All command examples use grouped Python modules:

```bash
uv run python -m scripts.analysis.backtest --help
uv run python -m scripts.review.render_debug --help
docker compose --profile tools build toolbox
docker compose --profile tools run --rm toolbox
```

For deployment, follow the [Oracle Cloud runbook](docs/oracle_cloud_deployment.md), including
session bootstrap, backups and schema checks before starting the watcher. Compose bind-mounts
`./data` at `/app/data`. Rebuild the image when upgrading to the grouped script paths.

The [operator guide](docs/operator-guide.md) covers ingestion, camera calibration, labelling,
backtests, alerts, bot commands and the offline vegetation prototype. The
[current handoff](docs/next_agent_handoff.md) records detector behavior, validation and next work;
[the capability map](docs/detection_capability_map.md) explains the detector's limits.

Shape and trajectory evidence may only escalate an outside alert. Blob shape must never
downgrade or suppress one; the classifier and event resolver preserve this policy.

# Oracle Cloud deployment runbook

This is the repository-specific upgrade procedure for the existing Oracle Cloud watcher. It
assumes Docker Engine and Docker Compose are already installed and the server already has its
working `.env`, `data/perimeter_watch.db`, and `data/session.session`. It deliberately does not
cover generic VM, SSH, firewall, or DNS setup.

The release expects database schema 10. The currently deployed schema-8 database must be migrated
to v9 and then v10, in that order. New watcher code intentionally refuses to start against an
older schema.

## Before the maintenance window

Make the release commit available to the server and confirm the server checkout has no local
changes that would be overwritten. Do not replace the server's `.env` or `data/` directory from
Git; both are deployment state and are ignored by the repository.

Confirm the bind-mount owner matches `PUID`/`PGID` in `.env`, and record the native architecture:

```bash
id -u
id -g
uname -m
docker compose ps
git status --short --branch
```

The established server was `x86_64`/Docker `amd64`. If `uname -m` instead reports `aarch64`, build
on that host and treat the post-build H.264 smoke test below as mandatory; a cross-build alone is
not ARM64 runtime validation.

Check these `.env` inputs without printing their secret values:

- `TELEGRAM_API_ID`, `TELEGRAM_API_HASH`, `SOURCE_CHANNEL`, and
  `TELEGRAM_SESSION_PATH` for source ingestion.
- `TELEGRAM_BOT_TOKEN` and `ALERT_CHANNEL_ID` for Telegram alerts and the query bot.
- `NTFY_BASE_URL`, `NTFY_TOPIC`, and optional `NTFY_TOKEN` for ntfy. Container `localhost` refers
  to the watcher container, not the Oracle host; use a reachable service name or host URL.
- `PUID` and `PGID` matching the owner of `data/`.
- Leave `EVENT_WAIT_SECONDS=300` unless newer measurements justify changing it.
- Choose `MEDIA_RETENTION_ENABLED` deliberately. It is disabled by default.

Do not run `session-bootstrap` during a normal upgrade. The existing Telethon session is persistent
and should be reused.

## Stop, preserve, build, and migrate

Run these commands from the repository root. The backup directory is gitignored and contains
secrets, so keep its permissions restricted and copy it off-host after the deployment.

```bash
docker compose stop watcher

backup_stamp="$(date -u +%Y%m%dT%H%M%SZ)"
backup_dir="data/backups/pre-schema-v10-${backup_stamp}"
install -d -m 700 "$backup_dir"
cp -a data/perimeter_watch.db "$backup_dir/"
if [ -f data/perimeter_watch.db-wal ]; then cp -a data/perimeter_watch.db-wal "$backup_dir/"; fi
if [ -f data/perimeter_watch.db-shm ]; then cp -a data/perimeter_watch.db-shm "$backup_dir/"; fi
cp -a data/session.session "$backup_dir/"
if [ -f data/session.session-journal ]; then cp -a data/session.session-journal "$backup_dir/"; fi
cp -a .env "$backup_dir/env"
chmod 600 "$backup_dir/env"

docker image tag nature-ridge-perimeter-watch:local \
  "nature-ridge-perimeter-watch:pre-v10-${backup_stamp}"
docker compose --profile tools build --pull toolbox
```

Check the database version with the new toolbox image. This command opens SQLite directly and
therefore does not invoke the application's schema-10 guard:

```bash
docker compose --profile tools run --rm --no-deps toolbox \
  python -c "import os,sqlite3; c=sqlite3.connect(os.environ['DB_PATH']); print(c.execute('SELECT version FROM schema_version').fetchone()[0])"
```

For the expected v8 deployment, run exactly:

```bash
docker compose --profile tools run --rm --no-deps toolbox \
  python -m scripts.migrate_schema_v9
docker compose --profile tools run --rm --no-deps toolbox \
  python -m scripts.migrate_schema_v10
```

Each migration is transactional and version-gated. Stop if either command reports a version other
than the one it expects. Do not skip directly from v8 to v10. If the version check already reports
9, run only v10; if it reports 10, do not rerun either migration.

Validate schema, configuration, writable persistent storage, and the native media stack before
starting the long-running service:

```bash
docker compose --profile tools run --rm --no-deps toolbox \
  python -m scripts.container_healthcheck --readiness --media-smoke
```

The JSON result must contain `"ready": true`, `"schema_version": 10`, and
`"media_smoke": true`.

## Start and verify

The toolbox and watcher share the same built image tag, so no second build is needed:

```bash
docker compose up -d --no-build watcher
docker compose ps
docker compose logs --tail=200 watcher
docker compose exec watcher python -m scripts.container_healthcheck \
  --readiness \
  --heartbeat /app/data/live/watcher.heartbeat \
  --max-heartbeat-age 600
```

Require the watcher to become `healthy`. The logs should contain `watcher_started`; when a Bot API
token is configured they should also contain `query_bot_started`. Investigate any
`ambiguous_deliveries_require_review`, `live_delivery_failed`, or `system_delivery_failed` entry
instead of deleting its outbox row.

An optional, intentional external smoke test sends one clearly marked text notification through
each configured transport:

```bash
docker compose --profile tools run --rm --no-deps toolbox \
  python scripts/send_test_alert.py
```

Then verify one harmless bot query such as `/health`. Do not manufacture a camera event just to
test priority; automated tests cover the routing policy:

- incident: Telegram plus urgent ntfy;
- animal, resident, and neighbour: Telegram plus default-priority ntfy;
- recognized maintenance/system transitions: Telegram plus default-priority ntfy.

The v9 migration preserves existing camera delivery state. At startup, interrupted `sending` rows
become `ambiguous` and are not automatically duplicated. Known failed/pending work resumes. If a
transport was configured only after old animal/incident events finalized, recovery can enqueue
their missing delivery; old resident/neighbour events are not retrospectively sent to ntfy. Schema
v10 does not back-notify historical system events; it creates the durable outbox for new recognized
transitions.

## Rollback boundary

Do not start old watcher code against the migrated schema-10 database. If rollback is necessary,
stop the watcher, preserve the failed deployment state for diagnosis, restore the complete
pre-upgrade database set (main DB and any matching WAL/SHM files) plus the Telethon session from
the same backup directory, retag the saved `pre-v10` image as
`nature-ridge-perimeter-watch:local`, and recreate the watcher. Never mix a database main file
with WAL/SHM files from a different backup.


# Operator tools

Run commands from the repository root using `uv run python -m scripts.<group>.<tool>`.
Each group is a Python package. Shared application and detector logic belongs in `src/`;
command parsing, interactive review and offline orchestration belong here.

| Group | Tools | Purpose |
| --- | --- | --- |
| `ingest` | `list_channels`, `meta_backfill`, `download_clips`, `restore_media`, `sync_system_events` | Telegram setup, source history and media recovery |
| `analysis` | `backtest`, `spike`, `infer_camera_order`, `rank_candidates`, `find_storm_events`, `build_reference_bg` | Detector evaluation, calibration and review shortlists |
| `analysis` | `audit_detection_assumptions`, `audit_system_messages`, `check_incident_regression`, `check_environment_hypothesis`, `probe_vegetation` | Diagnostics, preservation checks and offline experiments |
| `review` | `label`, `render_debug`, `extract_frame`, `motion_heatmap`, `visualize_zone` | Human review, labels and visual evidence |
| `ops` | `watch`, `container_healthcheck`, `send_test_alert`, `reclassify_system_events` | Live operation, readiness, transport tests and system-event repair |
| `migrations` | `migrate_schema_v5` through `migrate_schema_v10`, `migrate_add_resident_label`, `migrate_add_neighbour_label` | Explicit upgrades for existing databases |

Common commands:

```bash
uv run python -m scripts.analysis.backtest --labelled-only --no-record
uv run python -m scripts.review.label --camera cam07
uv run python -m scripts.review.render_debug --exact-message-id 22289
uv run python -m scripts.analysis.probe_vegetation --help
```

Compose uses `scripts.ops.watch`, `scripts.ops.container_healthcheck` and
`scripts.ingest.list_channels`. Operator tools run in the same image:

```bash
docker compose --profile tools run --rm toolbox \
  python -m scripts.analysis.backtest --labelled-only --no-record
```

The previous flat paths, such as `scripts/backtest.py` and `scripts.watch`, moved into
these groups. Update saved commands and rebuild the Docker image together with the Compose
configuration. The [operator guide](../docs/operator-guide.md) contains the full workflows;
the [deployment runbook](../docs/oracle_cloud_deployment.md) covers schema upgrades.

Keep generated output in the locations described by [the data guide](../data/README.md).
Add tool tests to the matching group under `tests/scripts/`.

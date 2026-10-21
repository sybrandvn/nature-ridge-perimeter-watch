# Local data

Data lives in this working directory and is ignored by Git. Only layout guides and empty-directory
placeholders are versioned. A fresh clone has no database, session, source footage or reports.
Run `make data-dirs` to create the standard folders without changing existing files.

| Path | Contents |
| --- | --- |
| `perimeter_watch.db` and its `-wal` / `-shm` companions | Application database and SQLite state |
| `session.session` | Telethon authorization session |
| `history/<camera>/<message-id>.mp4` | Backfilled source clips referenced by database rows |
| `live/clips/` | Source media retained for live operation |
| `live/debug/<camera>/` | Regenerable bot debug cache |
| `live/watcher.heartbeat` | Watcher liveness marker |
| `reference_bg/` | Production reference manifest and derived camera images |
| `reports/investigations/YYYY-MM-DD/<topic>/` | Self-contained investigations, baselines and evidence |
| `reports/reviews/YYYY-MM-DD/<topic>/` | Review CSVs and paired `.message_ids` shortlists |
| `reports/debug/YYYY-MM-DD/<topic>/<camera>/` | Deliberate visual reviews; dated `unfiled/` is the CLI fallback |
| `reports/scratch/YYYY-MM-DD/<topic>/` | Intermediate output and disposable experiments |
| `reports/archive/YYYY-MM-DD/<topic>/` | Preserved historical output and cleanup manifests |
| `backups/database/` | Standalone SQLite snapshots |
| `backups/labels/` | Exported label snapshots |
| `backups/<snapshot-name>/` | Complete upgrade/rollback bundles, kept together |
| `logs/` | Retained operational logs |

Use the date an investigation or review begins, then a short topic name. Keep its inputs,
figures, validation results and reference provenance together. For example:

```text
reports/investigations/2026-10-21/vegetation-prototype/
reports/reviews/2026-09-20/night-outside-candidates/night_outside_candidates_2026-09-20.csv
reports/reviews/2026-09-20/night-outside-candidates/night_outside_candidates_2026-09-20.message_ids
reports/debug/2026-09-20/low-blob-environment/cam07/
```

Use the CLI's `--out` / `--out-dir` options to select these destinations. Temporary experiments
go under a dated topic in `reports/scratch/`; keep the top of `reports/` free of loose output
files. The [reports guide](reports/README.md) defines the common category/date/topic layout.

The 2026-10-21 cleanup filed the local data as follows:

| Previous location under `data/` | Current location |
| --- | --- |
| `reports/code_review_2026-09-19/` | `reports/investigations/2026-09-19/code-review/` |
| `reports/false-alarm-review-2026-10-19/` | `reports/investigations/2026-10-19/false-alarm-review/` |
| `reports/environment-investigation-2026-10-20/` | `reports/investigations/2026-10-20/environment/` |
| `reports/vegetation-prototype-2026-10-21/` | `reports/investigations/2026-10-21/vegetation-prototype/` |
| Loose dated review CSVs and `.message_ids` | `reports/reviews/<date>/<topic>/` |
| Former flat scratch experiment folders | `reports/scratch/<date>/<topic>/` |
| Former archived debug sets and backtests | `reports/archive/<date>/<topic>/` |
| Loose `backups/*.db` | `backups/database/` |
| Loose `backups/*.jsonl` | `backups/labels/` |

All moved files were verified with SHA-256 before and after the move. The local
`reports/archive/2026-10-21/repository-cleanup/data-moves.json` records the initial data moves,
and `reports/archive/2026-10-21/report-layout/moves.json` records the report standardization.
Both contain original paths, current paths, sizes and checksums. Historical artifact contents
retain their original provenance strings; follow the manifests in order for old output paths.

Database, session, history, live-media and production-reference paths stay in place because
configuration and persisted rows use them. Compose mounts this entire directory at `/app/data`.
Stop writers and use SQLite's backup API or the deployment runbook before copying a live
database. Keep complete rollback bundles together, including their session and configuration.

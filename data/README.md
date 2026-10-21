# Local data

Data lives in this working directory and is ignored by Git. Only this guide and empty-directory
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
| `reports/reviews/YYYY-MM-DD/` | Review CSVs and paired `.message_ids` shortlists |
| `reports/debug/YYYY-MM-DD/<topic>/<camera>/` | Deliberate visual reviews; `unfiled/` is the CLI fallback |
| `reports/scratch/<experiment>/` | Intermediate output and disposable experiments |
| `reports/archive/` and `reports/debug/archive/` | Preserved historical output |
| `backups/database/` | Standalone SQLite snapshots |
| `backups/labels/` | Exported label snapshots |
| `backups/<snapshot-name>/` | Complete upgrade/rollback bundles, kept together |
| `logs/` | Retained operational logs |

Use the date an investigation or review begins, then a short topic name. Keep its inputs,
figures, validation results and reference provenance together. For example:

```text
reports/investigations/2026-10-21/vegetation-prototype/
reports/reviews/2026-09-20/night_outside_candidates_2026-09-20.csv
reports/reviews/2026-09-20/night_outside_candidates_2026-09-20.message_ids
reports/debug/2026-09-20/low-blob-environment/cam07/
```

Use the CLI's `--out` / `--out-dir` options to select these destinations. Temporary experiments
go under `reports/scratch/`; keep the top of `reports/` free of loose output files.

The 2026-10-21 cleanup filed the local data as follows:

| Previous location under `data/` | Current location |
| --- | --- |
| `reports/code_review_2026-09-19/` | `reports/investigations/2026-09-19/code-review/` |
| `reports/false-alarm-review-2026-10-19/` | `reports/investigations/2026-10-19/false-alarm-review/` |
| `reports/environment-investigation-2026-10-20/` | `reports/investigations/2026-10-20/environment/` |
| `reports/vegetation-prototype-2026-10-21/` | `reports/investigations/2026-10-21/vegetation-prototype/` |
| Loose dated review CSVs and `.message_ids` | `reports/reviews/<date>/` |
| Loose `backups/*.db` | `backups/database/` |
| Loose `backups/*.jsonl` | `backups/labels/` |

All moved files were verified with SHA-256 before and after the move. The local
`reports/archive/repository-cleanup-2026-10-21.json` records every original path, current path,
size and checksum. Historical artifact contents retain their original provenance strings;
use this mapping when following an old report's output paths.

Database, session, history, live-media and production-reference paths stay in place because
configuration and persisted rows use them. Compose mounts this entire directory at `/app/data`.
Stop writers and use SQLite's backup API or the deployment runbook before copying a live
database. Keep complete rollback bundles together, including their session and configuration.

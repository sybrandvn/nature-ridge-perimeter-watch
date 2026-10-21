# Reports

Use one layout for all generated reports:

```text
<category>/<YYYY-MM-DD>/<topic>/
```

This is the implemented layout used by the existing folders and debug CLI defaults.
Topic-first ordering (`<category>/<topic>/<YYYY-MM-DD>/`) was recommended in discussion
but has not been implemented. Use the layout above for current work. A future migration
must update the actual folders, output defaults, affected tests and path references together.

This guide is the source of truth for report placement. Link to it from other guidance
rather than defining a different convention elsewhere.

Use a lowercase topic with words separated by hyphens. The date is the report or experiment's
start date in Africa/Johannesburg, rather than the date of the source footage. Keep an existing
bundle's internal files together; filenames inside it can preserve their original provenance.

| Category | Contents | Example |
| --- | --- | --- |
| `investigations` | Baselines, findings, figures and validation for an investigation | `investigations/2026-10-21/vegetation-prototype/` |
| `reviews` | Human-review tables and matching `.message_ids` queues | `reviews/2026-09-20/night-outside-candidates/` |
| `debug` | Deliberate detector renders, optionally grouped by camera inside the topic | `debug/2026-09-20/low-blob-environment/cam07/` |
| `scratch` | Intermediate output and experiments | `scratch/2026-09-09/refactor-baseline/` |
| `archive` | Preserved completed/legacy output and cleanup manifests | `archive/2026-09-19/debug-review-false-alerts-v2/` |

For new output, pass a destination below a date and topic. For example:

```bash
uv run python -m scripts.analysis.backtest --labelled-only --no-record \
  --out data/reports/scratch/2026-10-21/backtest/results.csv
uv run python -m scripts.analysis.rank_candidates \
  --out data/reports/reviews/2026-10-21/candidates/candidates.csv
uv run python -m scripts.review.render_debug --exact-message-id 22289 \
  --out data/reports/debug/2026-10-21/camera-review/cam07/22289.mp4
```

Without `--out`, debug rendering uses `debug/<today-in-SAST>/unfiled/<camera>/<message-id>.mp4`.
Name the topic when saving a deliberate review. Production bot renders continue to use the
separate operational cache at `data/live/debug/`.

Keep the report root free of artifacts and aliases. All categories use the same date/topic
structure, including scratch work and archived renders. Pair a review CSV with its queue in
the same topic directory. Original file contents and names are retained during filing.

The local manifests record original paths, current paths, sizes and SHA-256 checksums:

- `archive/2026-10-21/repository-cleanup/data-moves.json`: initial data cleanup.
- `archive/2026-10-21/report-layout/moves.json`: the subsequent report standardization.

Follow the manifests in sequence when an older report contains a historical output path.
For legacy folders without a date in their name, the report cleanup used the earliest artifact
modification date in Africa/Johannesburg and recorded that inference in its manifest. Archived
debug output is consolidated under `archive/`; the former `debug_render` alias was removed.

This guide is versioned. Report artifacts and local manifests are ignored by Git and are
available only in this workspace. See [the data guide](../README.md) for all other local data.

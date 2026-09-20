# Next-agent handoff: operate and improve the production watcher

Updated 2026-09-20 on `main` after the detector, Docker, Telegram, retention, and restoration work
was completed.

## Completed production path

- `src/clip_analysis.py` is the shared production/backtest analysis entry point.
- `src/event_keys.py` owns embedded timestamp grouping and lifecycle parsing.
- schema v8 durably stores processed messages, pending/final events, analyzed siblings, and
  per-transport delivery attempts; migrate v7 with `scripts/migrate_schema_v8.py`.
- `src/live_watcher.py` and `scripts/watch.py` provide live Telethon ingestion, atomic media
  downloads, measured sibling buffering, conservative urgent-event resolution, Telegram video
  delivery, ntfy delivery, bounded retry, restart recovery, and heartbeat maintenance.
- Compose now runs `watcher` by default with `restart: unless-stopped`, persistent `./data`, and a
  readiness-plus-heartbeat healthcheck. The operator and session-bootstrap profiles remain.
- `src/bot.py` supplies the role-restricted Phase 5 query bot. Both roles receive `/about`,
  `/tonight`, `/health`, `/power`, `/batteries`, `/panel`, `/faults`, `/animal`, `/animals`,
  `/incidents`, `/residents`, `/neighbours`, paged `/history`, direct `/event <id>`, the
  approximate `/map`, and
  `/last <camera_id>`; missing media is restored from the source channel. `/patrols` is enforced as
  trustee only in its handler. Bot registration and allowlist values remain deployment inputs.
  `/menu` and `/start` provide a five-section inline interface with camera buttons, paged event
  buttons, and back navigation. The four visible slash commands (`/menu`, `/tonight`, `/health`,
  `/about`) all return useful information with one tap. Specialized status commands such as
  `/batteries` and parameterized commands remain available through inline navigation and direct
  handlers but are omitted from Telegram's command list.
  Events has paged Animal, Incident, Resident, and Neighbour histories. All four categories are
  delivered to the configured Telegram channel; ntfy remains animal/incident-only because its
  configured priority is urgent.
  The Info section contains About and Security contacts. Contact values are optional environment
  inputs (`SECURITY_COMPANY_NAME`, `SECURITY_COMPANY_PHONE`, `CONTROL_ROOM_PHONE`, and
  `ARMED_RESPONSE_PHONE`) and deliberately render as `Not configured` until supplied.
- Every proactive Telegram alert and video returned by the query bot carries a `Debug view`
  callback. It restores missing source media, renders the exact production detector overlay in a
  worker thread, and caches it below
  `data/debug` with the extraction fingerprint and renderer version. The secondary
  `/debug <video-id>` command accepts any known clip, including a latest-camera video outside the
  animal/incident history. `scripts.watch.render_bot_debug_video` supplies the timestamp-specific
  zone, reference background, and full `MotionThresholds` record to `render_clip`.
- `scripts/send_test_alert.py --examples` sends clearly marked copies of the standing cam08/7360
  animal and cam06/21520 incident clips to the configured Telegram alert destination.
- `src/retention.py` keeps urgent evidence, pending events, and one newest local video per camera.
  When explicitly enabled, other managed videos are removed hourly while metadata remains
  available for live retrieval. The same pass applies that policy to renderer-owned `data/debug`
  videos and removes obsolete fingerprint/version variants. It is opt-in via
  `MEDIA_RETENTION_ENABLED=true` and defaults off on development machines.
- `scripts/restore_media.py` resumably restores every missing database clip in Telegram batches.
- The measured sibling wait is 300 seconds: only 6 of 8,274 observed sibling groups exceeded it;
  99.9% completed within 286 seconds. Completion captions finalize immediately, and a late urgent
  sibling reopens an already-finalized suppressed event.
- The resolver preserves any `animal_candidate` or `incident_candidate` sibling. The proposed
  completed-guard and longest-only overrides remain rejected because measured protected events
  lost alerts under those policies.

Independent code review, later the same day:
[code_effectiveness_review_2026-09-19.md](code_effectiveness_review_2026-09-19.md).
Read it before changing the resolver. It reproduces validation, renderer,
warmup, geometry and tracking defects; measures all labelled sibling events; and
shows why longest-only and blanket guard-overrides lose protected or explicitly
approved alerts. Its implementation follow-up records the completed detector fixes,
the rejected coexistence override, and the new cold-corpus measurements.

## Scope

Operate the completed watcher, inspect real live outcomes, and improve it only from measured
false-positive/false-negative evidence.

Keep the reusable production logic in `src/`. Thin process/CLI entry points may live in `scripts/`,
but do not promote the existing backtest or debug scripts into the service.

## Current state

- Detector review and production work are committed logically on `main`.
- `EXTRACTOR_VERSION` is `motion-features-v8` (2026-09-20: added
  `warmup_dynamic_inside_bottom_left_fraction`).
- Ruff is clean and the full suite is **851 passing tests** as of 2026-09-20; rerun both
  before changing behavior because the count grows with each layer.
- Fresh labelled extraction plus final classifier replay: **817 clips**, TP 23 / FP 29 /
  FN 25 / TN 740, precision 0.442, recall 0.479, F1 0.460 (unknowns counted negative).
  This is the current, expanded ground-truth corpus and is not directly comparable to the
  older 708-clip snapshot.
- Protected-event regression remains green for all 5 incident and 20 animal events.
- cam12/9162 remains `animal_candidate`: it is labelled `unknown`, but the user's
  review says it may contain a small animal as well as a camera artifact.
- Telegram and Docker production paths are implemented as summarized above. BotFather setup is
  complete, the token and alert destination are configured, one trustee is authorized, and the
  watcher container was healthy after the latest rebuild. No security-role user is configured.
- The representative Telegram smoke test successfully sent cam08/7360 (animal) and cam06/21520
  (incident), including their `Debug view` buttons. Telegram has normal or silent delivery but no
  Bot API priority level; ntfy is configured as `urgent` and receives only animal/incident alerts.
- The new security contact directory is deployed but its four optional values are still blank.
  Populate them in `.env` when the user obtains the security company, control-room, and armed
  response details, then recreate the watcher so Compose reloads the environment.
- Media retention is currently disabled on this analysis PC as intended. Enable it only in the
  server environment; source and debug-cache cleanup then follow the documented policy.
- Local bulk restoration recovered 16,897 of 16,899 clip paths and verified every recorded path
  exists. An expanded eight-attempt retry on 2026-09-20 still received Telegram `GetFileRequest`
  timeouts for `cam02/15475` and `cam01b/17386`; both remain null-path rows and are safe to retry
  later without revisiting successful downloads.

Run the normal checks with:

```bash
uv run ruff check .
uv run pytest -q
uv run python scripts/backtest.py \
  --out /tmp/perimeter-labelled.csv --labelled-only --no-record
```

Use `--no-record` for experiments. Bump `src.motion.EXTRACTOR_VERSION` whenever cached feature
semantics change; classifier-only changes do not require an extractor bump.

## Operating checklist

Keep `MEDIA_RETENTION_ENABLED=false` on an analysis workstation. Set it to `true` only in the
server's `.env` when aggressive cleanup is wanted. After changing the value, recreate the watcher
so Compose loads the updated environment.

To recover server-cleaned clips onto a machine that shares the same database and Telethon session,
stop the watcher and run:

```bash
docker compose stop watcher
docker compose --profile tools run --rm toolbox \
  python -m scripts.restore_media --concurrency 8
docker compose up -d watcher
```

The restore records each successful download immediately and safely resumes by selecting only rows
whose `file_path` is still null. Keep the watcher stopped during restoration because both processes
use the same Telethon session.

For deployment, rebuild the image, run the schema migration if upgrading from v7, start the default
watcher service, and verify its readiness-plus-heartbeat healthcheck. Keep credentials in `.env`;
the logging configuration redacts Telegram bot tokens and suppresses URL-bearing HTTP client INFO
logs, but secrets must still never be pasted into reports or chat.

The Bot API token currently remains a deployment input. If Telegram rejects it, source ingestion
through Telethon can still run, while query-bot startup and Bot API alert delivery remain
unavailable until a valid token is installed.

## Detection behavior that is already settled

### Warmup motion and flashlight

`src/scoring.py` now measures adjacent, photometrically corrected warmup frames through
`warmup_dynamic_frame_fraction` and `warmup_dynamic_outside_fraction`. The flashlight reading is
also independent of the settled-track summary. Five labelled guards that previously returned
`no_motion` are recovered:

- 5464 and 17501 through `warmup_dynamic_inside`;
- 18105, 21491, and 21506 through `warmup_flashlight`.

The no-settled-motion condition is deliberate. Onside warmup activity can coexist with a genuine
outside subject, so warmup evidence must not globally override a later scored track.

`scripts/render_debug.py` now uses the scorer's exact exclusion mask, warmup daylight gate,
raw-frame flashlight ratios, and per-track peak flashlight scores. Every qualifying green-light
component is outlined in warmup and settled frames. Daylight-gated warmup candidates remain visible
in a distinct colour and are labelled as gated; their returned/scored ratio remains zero. A
persistent flashlight track stays labelled across its full lifetime after any frame crosses the
same 0.02 peak threshold used by `multi_object_flashlight` classification.

### Camera artifacts

Black/white rectangular artifacts are exposed as evidence, not a blanket veto.

- `rectangular_black_white_balance` scans rectangular contours even when they did not win the
  main track.
- A terminal reverse seed plus clipped pixels only routes to guard when there is independent,
  persistent onside warmup movement (`warmup_inside_terminal_artifact`, currently cam03/5465).
- cam12/9162 has artifact evidence but no such warmup corroboration, so it remains alert-worthy.
- cam07/4487 is unclear but explicitly approved to alert.

Do not reintroduce a rule that suppresses a clip merely because an artifact is present.

### Camera/fence movement

`global_camera_shift_score` is reporting-only. It rises on the reviewed wind/camera examples
(cam15/16208 = 2.323, cam15/16564 = 1.319, cam05/18679 = 1.494), but it also rises when a large
real subject dominates the image (guard cam03/20522 = 6.826). It is useful in the debug HUD but is
not safe as an environment gate.

### Cam12 geometry

There are two evidence-backed Cam12 eras, not three:

1. the user's improved trace applies to the entire pre-2026-remount period;
2. the remounted geometry starts at `2026-03-02T16:13:54Z`.

Do not restore the unsupported `2024-11-23` middle era. Clips 8982 and 9161/9162 show the same
pose, and earlier registration put 3457 and 9162 within roughly one pixel.

### Alert-worthy ambiguous footage

Do not tune these away without new user review:

- cam05/9694 and 9695 — possible very small daylight animal;
- cam07/19288 — unclear, but approved to alert;
- cam12/9162 — possible animal/unknown plus artifact;
- cam13/7778 — looks artifact-like, but approved as an incident alert;
- cam07/4487 — unclear, but approved to alert.

cam07/19145 is a bag covering much of the camera. Its incident category and maintenance flag are
intentionally orthogonal.

## Review data

- Refreshed debug set: `data/reports/review_false_alerts_2026-09-19-v2/`
- Label backup before the review:
  `data/backups/labels_pre_false_alert_review_20260919.jsonl`
- Label backup after the review:
  `data/backups/labels_after_false_alert_review_20260919.jsonl`
- Live labels/database: `data/perimeter_watch.db`
- Reference backgrounds: `data/reference_bg/`

These paths are gitignored operational data. Do not delete or reorganize them merely because they
do not appear in `git status`.

## Architecture map

- `src/motion.py` — detector and cached extraction identity.
- `src/scoring.py` — feature calculation from a `ClipDetection`.
- `src/classify.py` — deterministic per-clip classification and reason codes.
- `src/clip_analysis.py` — shared reference/cache/extraction/classification path.
- `src/event_keys.py` — production sibling grouping and lifecycle parsing.
- `src/live_watcher.py`, `src/live_state.py` — watcher orchestration and durable state transitions.
- `src/media_download.py` — video-type validation and atomic downloads.
- `src/backfill.py`, `src/message_parsing.py` — reusable Telegram message ingestion/parsing pieces.
- `src/telegram_alert.py`, `src/ntfy_alert.py` — injectable Telegram video/text and ntfy transports.
- `src/db.py` — schema v8 persistence, including live event and delivery state.
- `src/config.py`, `config/thresholds.yaml` — typed classification thresholds.
- `config/cameras.yaml` — dated geometry and camera configuration.
- `scripts/backtest.py` — labelled/full-corpus evaluation harness.
- `scripts/render_debug.py` — visual detector inspection; its HUD shows classification, warmup
  evidence, rectangular artifact evidence, and global shift evidence.
- `scripts/label.py` — label workflow using the shared production event key.
- `scripts/watch.py` — Telethon process entry point and heartbeat loop.
- `scripts/migrate_schema_v8.py` — additive v7-to-v8 migration.
- `scripts/send_test_alert.py` — manual text-transport smoke test; `--examples` sends the standing
  labelled animal and incident videos with debug buttons.
- `.env.example` — existing Telegram, Bot API, ntfy, DB, and operating-window settings.
- `docs/handoff.md` — full historical investigation log. Start with session #22; older sections
  are retained for provenance and sometimes explicitly superseded.
- `docs/detection_capability_map.md` — feature inventory and scope.

`tests/test_architecture.py` enforces that `src` does not import `scripts`.

## Optional work, not a blocker

The persistent cache currently stores final feature mappings rather than fully zone-independent raw
tracks. Extracting a compact raw-track payload would allow fence geometry to be rescored without
rerunning detection. The user accepts reprocessing after detector changes, so this refactor is not
required before the event-policy work and should not displace it.

Likewise, do not create global suppression thresholds for storms, bushes, spider webs, rectangular
artifacts, or camera movement without protected-class margin. Their evidence currently overlaps
real animal/incident footage.

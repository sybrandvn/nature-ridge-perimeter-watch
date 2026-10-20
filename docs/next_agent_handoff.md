# Next-agent handoff: operate and improve the production watcher

Updated 2026-10-19 for the cam15 mixed-motion environment rule. The deployment notes below that
date to September remain historical; no local watcher container was running during this update.

The approved update is now live on the Oracle host reached by SSH alias `bastion`, at
`/opt/nature-ridge-perimeter-watch`. Activation on 2026-10-19 passed readiness and heartbeat checks;
the single watcher is healthy on ARM64, Python 3.12.14, schema 10, extractor v9 and renderer v3.
The existing runtime was retained. The rollback image is
`nature-ridge-perimeter-watch:pre-cam15-20261019T184538Z`; consistent database/session backups,
previous source and validation reports are under the server's
`data/backups/pre-cam15-20261019T184538Z/`. Ground-truth labels remain identical (1,014 entries).
On the server, the source update is applied as working-tree changes on top of `b1c8721`;
preserve and reconcile those changes before a future Git update.

### Cam15 classifier update, 2026-10-19

- User approved the measured cam15-only rule after reviewing its offline preservation results.
  It acts only before animal/incident returns, retains flashlight priority, and requires mixed
  scene activity, exactly one direct subject detection, mostly inferred boxes and a flare ramp.
  The six numeric limits are in `config/thresholds.yaml`; camera identity is an explicit keyword
  passed by shared clip analysis, debug rendering, and real-clip audit tools.
- `motion-features-v9` adds `genuine_detection_count` and `scored_frame_count` without changing
  detector tracking. Old feature caches are invalidated. Debug rendering is version 3 and receives
  the same camera identity as production classification.
- The frozen baseline at `data/reports/false-alarm-review-2026-10-19/validation/baseline.jsonl`
  covers 1,014 tagged clips (845 class labels and 169 startup-only tags) plus eight live review
  clips. The production implementation matches the measured hypothesis: only cam15/24395,
  24398 and 24680 change to `environment_candidate/cam15_mixed_environment`. All historical
  tagged predictions, 39 alerting positive clips and all 24 animal + five incident events remain
  preserved. October 19's 24680 event is diagnostic and still lacks human confirmation; the two
  October 10 false alarms are user-confirmed environment.
- Verification completed: **941 tests pass**, Ruff is clean, and all 1,022 records were freshly
  extracted from video and replayed through the production analysis path with no cache. The
  deployed ARM64 image also passed all 24 frozen feature examples, H.264 encode/decode, and an
  old/new comparison using 13 review videos already present on the server, including both cam15
  animal-event carrier clips and all three nuisance alert clips. The other 11 review videos were
  absent from server storage; no footage was uploaded. Existing shared features had no drift.
- Replay using `python -m scripts.check_environment_hypothesis --production --baseline ...
  --out ...`. The check deliberately keeps cam13/7778's pre-existing approved-alert discrepancy
  visible: distance 0.099375 is below the 0.10 threshold, so both old and new paths are unclassified.
  `regression_passed` can be true while the strict overall `passed` remains false. Do not remove
  this approved clip from coverage or change its label to obtain a green result.
- Labels and live delivery records are not changed by the rule or regression tools. Do not start
  this stale local database/session as another live watcher merely to activate the update; use
  the actual running deployment and preserve its state.

## Completed production path

- `src/clip_analysis.py` is the shared production/backtest analysis entry point.
- `src/event_keys.py` owns embedded timestamp grouping and lifecycle parsing.
- schema v10 durably stores processed messages, pending/final camera events, analyzed siblings,
  event resolution state, staged camera deliveries, and a separate system-event notification
  outbox. Migrate v7 with v8, v9, then v10; v8 needs v9 then v10, and v9 needs only v10.
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
  delivered to the configured Telegram channel and ntfy. Urgent ntfy priority applies only to
  incidents; animal, resident, and neighbour notifications use default priority.
  The Info section contains About and Security contacts. Contact values are optional environment
  inputs (`SECURITY_COMPANY_NAME`, `SECURITY_COMPANY_PHONE`, `CONTROL_ROOM_PHONE`, and
  `ARMED_RESPONSE_PHONE`) and deliberately render as `Not configured` until supplied.
- Every proactive Telegram alert and video returned by the query bot carries a `Debug view`
  callback. It restores missing source media, renders the exact production detector overlay in a
  worker thread, and caches it below
  `data/live/debug` with the extraction fingerprint and renderer version. The secondary
  `/debug <video-id>` command accepts any known clip, including a latest-camera video outside the
  animal/incident history. `scripts.watch.render_bot_debug_video` supplies the timestamp-specific
  zone, reference background, and full `MotionThresholds` record to `render_clip`.
- `scripts/send_test_alert.py --examples` sends clearly marked copies of the standing cam08/7360
  animal and cam06/21520 incident clips to the configured Telegram alert destination.
- `src/retention.py` keeps urgent evidence, pending events, and one newest local video per camera.
  When explicitly enabled, other managed videos are removed hourly while metadata remains
  available for live retrieval. The same pass applies that policy to renderer-owned `data/live/debug`
  videos and removes obsolete fingerprint/version variants. It is opt-in via
  `MEDIA_RETENTION_ENABLED=true` and defaults off on development machines.
- `scripts/restore_media.py` resumably restores every missing database clip in Telegram batches.
- The measured sibling wait is 300 seconds: only 6 of 8,274 observed sibling groups exceeded it;
  99.9% completed within 286 seconds. An Initial incident now sends a clearly marked preliminary
  warning immediately. Completion captions finalize immediately and send a confirmed,
  conflicting, or likely-resolved update; timeout sends an unconfirmed update. Late completions
  reopen unconfirmed/conflicting events, and late urgent evidence also reopens likely-resolved
  events.
- Recognized system events are proactively delivered as well as retained for bot queries. Telegram
  receives battery, power, tamper, supervision, communication, and panel transitions. ntfy sends
  every system transition at default priority; only camera incidents use urgent priority.
  Detector-only `blinding_foreground` findings remain an offline maintenance queue.
- The resolver preserves any `animal_candidate` or `incident_candidate` sibling. The proposed
  completed-guard and longest-only overrides remain rejected because measured protected events
  lost alerts under those policies.
- Warmup debug rendering now consumes the same `WarmupMotionAnalysis` instance as feature scoring.
  It shows the scorer-corrected warmup frame, marks the selected object, valid-but-unselected
  objects, ignore-region objects and size-gated objects, and reports their counts. Use
  `scripts/render_debug.py --exact-message-id` for clip-level audits; ordinary `--message-id`
  intentionally resolves to the clearer sibling.
- A 2026-09-20 audit found five environment-labelled Telegram candidates with `blob_count <= 2`:
  cam13/11877 (daylight inside-only), cam12/19244 (outside-colour bushes), cam12/6352 (the known
  bright-branch lock), and cam15/16208 plus 16564 (loose-camera/fence motion). The first three are
  geometry/daylight ambiguity, while only the cam15 pair has strong global-shift evidence. Blob
  count is therefore not their alert cause, and no blanket low-count suppression was added.

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

- Historical detector work is on `main`; the approved cam15 update is deployed as the source
  bundle described above. Reconcile the server's deployed changes before any Git update.
- `EXTRACTOR_VERSION` is `motion-features-v9` (2026-10-19: added direct detection and scored-frame
  counts; the prior v8 added `warmup_dynamic_inside_bottom_left_fraction`).
- Ruff is clean and the full suite is **941 passing tests** as of 2026-10-19; rerun both before
  deployment or changing behavior because the count grows with each layer.
- `DEBUG_RENDER_VERSION` is 3. It passes camera identity to the classifier so the cam15 environment
  rule shown by debug matches the live analysis path; v2 introduced shared warmup observations.
- Historical 2026-09-21 labelled extraction/classifier replay: **817 clips**, TP 23 / FP 29 /
  FN 25 / TN 740, precision 0.442, recall 0.479, F1 0.460 (unknowns counted negative).
  This historical snapshot is not directly comparable to the older 708-clip snapshot or the
  October expanded corpus.
- The standing fixture retains all five incident and 20 animal events; the full October labelled
  corpus retains all five incident and 24 animal events and all 39 previously alerting clips.
- cam12/9162 remains `animal_candidate`: it is labelled `unknown`, but the user's
  review says it may contain a small animal as well as a camera artifact.
- Historical workstation deployment, 2026-09-21: Telegram and Docker production paths were
  implemented as summarized above. BotFather setup was
  complete, the token and alert destination are configured, one trustee is authorized, and the
  workstation watcher was rebuilt/recreated from `4ce3141` on 2026-09-21. It is healthy on schema
  10, reports `DEBUG_RENDER_VERSION=2`, and runs the Linux `amd64` image on `x86_64`. Its staged
  camera alerts, durable system notifications, and four-category Telegram/ntfy parity are active.
  No security-role user is configured. The pre-upgrade state is preserved at
  `data/backups/pre-schema-v10-20260921T064510Z` and image tag
  `nature-ridge-perimeter-watch:pre-v10-20260921T064510Z`.
- The live Oracle deployment now runs the cam15 update as recorded above. Fresh installations
  still initialize directly at schema 10; see the [Oracle Cloud runbook](oracle_cloud_deployment.md).
  Preserve the active server's environment, database and session during updates.
<!-- Legacy schema-copy migration script search terms:
  `scripts.migrate_schema_v9.py` and then `scripts.migrate_schema_v10.py` with the newly built
-->
- The representative Telegram smoke test successfully sent cam08/7360 (animal) and cam06/21520
  (incident), including their `Debug view` buttons. Telegram has normal or silent delivery but no
  Bot API priority level. For camera events, ntfy receives incidents at its configured `urgent`
  priority and animals, residents, and neighbours at default priority; the separate system outbox
  sends every recognized transition at default priority.
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

For the fresh deployment, use [the Oracle Cloud runbook](oracle_cloud_deployment.md); an empty
Oracle data directory initializes directly at schema 10. The v8-to-v9-to-v10 chain and rollback
boundary apply only when retaining the workstation database. Keep credentials in `.env`; the
logging configuration redacts Telegram bot tokens and suppresses URL-bearing HTTP client INFO logs,
but secrets must still never be pasted into reports or chat.

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

- Refreshed legacy false-alert set:
  `data/reports/debug/archive/legacy-debug-render/review_false_alerts_2026-09-19-v2/`
- Warmup bottom-left recovery validation and exact cam07/22289 render:
  `data/reports/debug/2026-09-20/warmup-bottom-left-exit/`
- Low-blob environment audit, exact production-input renders, and clearer event representatives:
  `data/reports/debug/2026-09-20/low-blob-environment/`
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
- `src/db.py` — schema v10 persistence, including camera resolution plus camera/system outboxes.
- `src/config.py`, `config/thresholds.yaml` — typed classification thresholds.
- `config/cameras.yaml` — dated geometry and camera configuration.
- `scripts/backtest.py` — labelled/full-corpus evaluation harness.
- `scripts/render_debug.py` — visual detector inspection; its HUD shows classification, shared
  warmup-object evidence, rectangular artifact evidence, and global shift evidence. Use
  `--exact-message-id` for a named clip and `--message-id` for its clearer event representative.
- `scripts/label.py` — label workflow using the shared production event key.
- `scripts/watch.py` — Telethon process entry point and heartbeat loop.
- `scripts/migrate_schema_v8.py`, `scripts/migrate_schema_v9.py`, and
  `scripts/migrate_schema_v10.py` — additive v7-to-v10 migration chain.
- `scripts/send_test_alert.py` — manual text-transport smoke test; `--examples` sends the standing
  labelled animal and incident videos with debug buttons.
- `.env.example` — existing Telegram, Bot API, ntfy, DB, and operating-window settings.
- `docs/handoff.md` — full historical investigation log. Start with session #22; older sections
  are retained for provenance and sometimes explicitly superseded.
- `docs/detection_capability_map.md` — feature inventory and scope.

`tests/test_architecture.py` enforces that `src` does not import `scripts`.

## What remains

There is no remaining implementation blocker for running the watcher. The remaining work is
operational setup, evidence-driven follow-up, or optional tooling.

### Deployment inputs and housekeeping

1. Keep operating the live Oracle watcher described above. Use the migration chain only for older
   schema copies; the cam15 update required no schema migration.
2. Fill the optional security-company/control-room/armed-response contact values when known, and
   add a `security` role user if trustees want that access tier. One trustee is already configured.
3. Decide whether the storage-constrained server should enable `MEDIA_RETENTION_ENABLED`; leave it
   off on analysis machines.
4. Retry Telegram restoration for `cam02/15475` and `cam01b/17386`. They are the only two known
   missing local clips out of 16,899 and do not block live operation.

The current Oracle server and deployed image are ARM64; the development workstation is AMD64.
Native runtime/media and classifier checks passed on the Oracle host on 2026-10-19. Validate on
the actual target host when changing architecture rather than relying on a cross-build alone.

### Evidence-driven detector work

The 2026-09-21 cold audit ran the current production detector across all 8,079 clips that had local
media, no class label, no startup metadata, and were not `Initial` captions. It produced 298
non-guard classifications, of which 14 would notify: 12 incident candidates and two animal
candidates. Human review labelled those 14 unique events as four animals, five guards, and five
environment events. The guard notes were faint flashlight, leaving frame, and mixed
guard/vegetation/bug footage; environment notes were bushes, rain, and spider web. This is ten
false notifications across the full 2023-11-17 through 2026-09-21 corpus window, roughly 3.5 per
year. It is acceptable for deployment under the conservative security policy, but it establishes
the next evidence set for reducing nuisance alerts. Recall is not established by this review
because the 8,065 suppressed predictions were not exhaustively watched.

1. Observe real live alerts and label concrete false positives/false negatives before changing
   thresholds. The expanded labelled snapshot still has limited rare-class support.
2. Do not add a low-blob blanket veto. The five audited leaks split across daylight inside
   geometry, outside vegetation/branch ambiguity, and camera motion; low blob count is not their
   classification reason.
3. Camera-shift evidence may support a future cam15-specific maintenance route, but it is not yet
   an alert veto: large genuine subjects can also produce a high global-shift score. Any proposal
   must be measured at protected-event level, including siblings.

### Optional tooling and product work

- Run-comparison CLI and projected false-pages-per-night reporting.
- Browser zone editor and a compact zone-independent raw-track cache.
- Generic migration framework; targeted migrations already cover schema v10.
- Per-camera classifier thresholds only if labelled evidence supports them.
- YOLO/ONNX resolver, activity heatmap, trend analytics, and retrospective probe reporting.
- Automatic proactive attachment of rendered debug video; on-demand `Debug view` is already live.

## Optional architectural refinement

The persistent cache currently stores final feature mappings rather than fully zone-independent raw
tracks. Extracting a compact raw-track payload would allow fence geometry to be rescored without
rerunning detection. The user accepts reprocessing after detector changes, so this refactor is not
required before the event-policy work and should not displace it.

Likewise, do not create global suppression thresholds for storms, bushes, spider webs, rectangular
artifacts, or camera movement without protected-class margin. Their evidence currently overlaps
real animal/incident footage.

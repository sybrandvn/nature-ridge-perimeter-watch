# Next-agent handoff: operate and improve the production watcher

Updated 2026-09-19 on `main` after the detector, Docker, and Telegram work was completed.

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
- The measured sibling wait is 300 seconds: only 6 of 8,274 observed sibling groups exceeded it;
  99.9% completed within 286 seconds. Completion captions finalize immediately, and a late urgent
  sibling reopens an already-finalized suppressed event.
- The resolver preserves any `animal_candidate` or `incident_candidate` sibling. The proposed
  completed-guard and longest-only overrides remain rejected because measured protected events
  lost alerts under those policies.

The sections below preserve the implementation brief and detector constraints for audit context.
Do not treat their future-tense statements as current repository status.

Independent code review, later the same day:
[code_effectiveness_review_2026-09-19.md](code_effectiveness_review_2026-09-19.md).
Read it before implementing the resolver below. It reproduces validation, renderer,
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
- `EXTRACTOR_VERSION` is `motion-features-v7`.
- Run the current full suite and Ruff before changing behavior; the count grows with each layer.
- Cold labelled backtest plus final classifier replay: **708 clips**, TP 32 / FP 24 /
  FN 15 / TN 637, precision 0.571, recall 0.681, F1 0.621 (unknowns counted negative).
- Protected-event regression: all 5 incident and 20 animal events pass with all 49
  protected sibling clips present and no exception.
- cam12/9162 remains `animal_candidate`: it is labelled `unknown`, but the user's
  review says it may contain a small animal as well as a camera artifact.
- Telegram and Docker production paths are implemented as summarized above.

Run the normal checks with:

```bash
uv run ruff check .
uv run pytest -q
uv run python scripts/backtest.py \
  --out /tmp/perimeter-labelled.csv --labelled-only --no-record
```

Use `--no-record` for experiments. Bump `src.motion.EXTRACTOR_VERSION` whenever cached feature
semantics change; classifier-only changes do not require an extractor bump.

## Original implementation sequence (completed; retained for audit context)

### 1. Measure and implement the completed-sibling event resolver

Do this offline first, then use it as the event-policy core of the live Telegram service.

Many camera events produce a short `(Initial)` preview followed by a longer `(Stopped)` or
`(Timeout)` clip. The short preview can lose a guard during IR warmup, then classify a later bush,
pole, noise patch, or camera artifact as outside motion. The longer sibling often identifies the
guard correctly. Confirmed examples:

| short clip | fuller sibling | short result | fuller result |
| --- | --- | --- | --- |
| cam01b/17500 | cam01b/17501 | `animal_candidate` | `guard_candidate` |
| cam07/18641 | cam07/18642 | `incident_candidate` | `guard_candidate` |
| cam03/20521 | cam03/20522 | `incident_candidate` | `guard_candidate` |

Do not simply make “guard wins” or “longest clip wins” the production rule. First measure every
labelled sibling event, including animal and incident events where the short clip may hold the only
useful evidence.

The experiment should compare at least:

1. current `classify_event` behavior, where any animal/incident sibling wins;
2. the fuller/completed sibling alone;
3. a conservative resolver that may let a clearly completed guard result override a weak Initial
   alert while preserving corroborated animal/incident evidence.

Report event-level TP/FP/FN/TN and list every animal or incident event changed by the proposed
policy. Only after that evidence should reusable, transport-independent resolution logic be added
under `src/`. Keep caption parsing and report generation in `scripts/` unless they are truly part
of the production-domain API.

Resolver definition of done:

- the policy is measured on all labelled sibling events;
- every protected animal/incident change is manually inspectable;
- tests cover event order, missing siblings, duplicate Initial clips, and an Initial alert whose
  fuller sibling is guard;
- the per-clip classifier remains usable unchanged;
- the resolver has a small transport-independent API suitable for the live watcher.

### 2. Build the live Telegram watcher

Use Telethon to subscribe to new messages from `SOURCE_CHANNEL`. Reuse `src.message_parsing` and
the existing camera configuration rather than duplicating caption/camera matching. For each clip:

1. upsert its metadata with `source='live'`;
2. download media to a stable path under `data/history/<camera>/` using a temporary file followed
   by an atomic rename;
3. group sibling messages by the embedded camera-event timestamp;
4. buffer an Initial preview for the measured/configured sibling window;
5. run detection, scoring, per-clip classification, maintenance checks, and the event resolver;
6. dispatch the final outcome once, even across a restart.

Move the reusable embedded event-timestamp/key parsing out of the private
`scripts.label._event_key` helper into `src`; production code must not import `scripts`. Likewise,
factor a pure `src` clip-analysis entry point that resolves dated camera geometry/reference data,
extracts features, returns the detailed classification and maintenance flags, and can be called by
both the watcher and offline tools. Do not make the watcher import `scripts.backtest`.

Derive the sibling wait window from the observed message-gap distribution instead of guessing it,
and make it configurable. A missing sibling must eventually flush rather than leave an event
pending forever. Unknown cameras and malformed/non-video messages must be logged and handled
without killing the watcher.

Add durable processing state to SQLite for pending events, processed source messages, final event
decisions, and delivery attempts. The current unique `(channel_id, message_id)` clip key is useful
but is not enough to guarantee exactly-once event delivery after a crash. Any schema change must
use an explicit migration following the existing `scripts/migrate_schema_v*.py` pattern; export
labels before destructive migration work.

The service must support graceful shutdown, structured logs, bounded retries/backoff, recovery of
pending events after restart, and a health signal that Docker can check. Keep network clients and
the clock injectable so the behavior is testable without Telegram access or real waiting.

### 3. Complete Telegram and ntfy delivery

`src.telegram_alert.send_telegram_alert` currently sends text only. Extend the transport so an
animal/incident alert can include the source clip (or a reliably playable H.264 derivative), the
camera, event time, category, reason, and useful evidence. Preserve a text-only fallback.

Keep routing policy explicit:

- `animal_candidate` and `incident_candidate` go to the urgent alert path;
- maintenance is orthogonal and may accompany any category;
- guard/resident/environment results are recorded but do not enter the urgent alert channel;
- delivery failure never marks an event delivered.

Use idempotency keys/durable delivery rows so a retry cannot post duplicate alerts. Existing
Telegram and ntfy functions are injectable and unit-tested; extend that pattern for media and
retry orchestration. `scripts/send_test_alert.py` should become the safe manual end-to-end
credential/media smoke test, not the live service itself.

### 4. Attach the watcher to the existing Docker deployment

`Dockerfile`, `compose.yaml`, `.dockerignore`, and `scripts/container_healthcheck.py` already cover
the reproducible/non-root image, persistent bind mount, locked dependencies, signal-forwarding
init, read-only root filesystem, config/DB readiness, optional watcher-heartbeat contract,
interactive Telethon bootstrap, and H.264 write/read verification. Keep those controls.

Once the watcher from step 2 exists, add it as the default production Compose service. At minimum:

- receive credentials through environment/secrets, never bake `.env` or a session into the image;
- use the healthcheck's `--heartbeat` option so health is tied to the real watcher heartbeat;
- use a restart policy and graceful stop period compatible with flushing SQLite and pending state;
- document unattended watcher startup, upgrade, backup, and recovery around the existing bootstrap
  and operator commands;
- keep the existing H.264 smoke check in deployment validation.

If ntfy runs in a separate container, configure its Compose service hostname; the example
`http://localhost:80` points back into the watcher container and will not reach a sibling service.

Add container-level tests or smoke checks for startup with fake transports and a temporary mounted
data directory. Do not require live credentials in the automated test suite.

### Production definition of done

- a new source clip can flow from a fake Telethon event through persistence, sibling resolution,
  classification, and a fake Telegram/ntfy delivery in an integration test;
- replaying the same source messages or restarting during delivery produces no duplicate alert;
- an Initial-only event flushes after the configured deadline;
- a completed guard sibling prevents the known preview false alerts without losing measured
  animal/incident events;
- Docker starts as non-root with persistent state and reports healthy only when the watcher and DB
  are ready;
- setup, session bootstrap, normal operation, upgrade, backup, and recovery are documented;
- the full detector/backtest regression and all unit/integration tests remain green.

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
- `scripts/send_test_alert.py` — current manual text-transport smoke test.
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

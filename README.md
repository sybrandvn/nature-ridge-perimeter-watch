# Nature Ridge Perimeter Watch

Motion-based perimeter fence monitoring built from a Telegram security-camera feed. The reusable
detection, scoring, classification, persistence, calibration, and alert-transport code lives
under `src/`. `scripts/watch.py` runs the durable Telegram watcher: it downloads new videos,
analyzes them, resolves Initial/Stopped siblings, and delivers camera-classified event alerts.

For a new coding agent, start with
[docs/next_agent_handoff.md](docs/next_agent_handoff.md). It describes the current detector state,
the measured detector behavior, completed service integration, and remaining improvement work.

The original development sequence used two gates before building the production pipeline:

1. **Gate 1 — camera order.** The fence camera order is unknown and must be inferred from patrol
   timestamp patterns alone (`src/sequence.py`).
2. **Gate 2 — CV feasibility.** Classical computer-vision features must actually separate
   guard/environment clips from animal/incident clips on a hand-labelled sample (`scripts/spike.py`).

Both gates were resolved, and the backfill, detector, watcher, Docker service, alert delivery, and
query bot are now implemented. The gate workflow remains below because it is the reproducible path
for recalibrating the detector from source evidence.

## Setup

```bash
uv sync
cp .env.example .env   # fill in Telegram credentials, channel id, etc.
```

Config lives in two places:

- `.env` — secrets and environment-specific paths (see `.env.example` for every variable).
- `config/cameras.yaml` / `config/thresholds.yaml` — camera roster, fence geometry, and
  motion/classification thresholds. Both start as skeletons and get filled in as you work through
  the steps below.

Run tests and lint after any change:

```bash
uv run ruff check . --fix
uv run pytest -q
```

## Docker

The image packages the locked runtime dependencies, watcher, and operator tools. It runs as a
non-root user, keeps the root filesystem read-only under Compose, and bind-mounts `./data` for the
SQLite database/WAL, Telethon session, downloaded history, references, reports, and logs.

Set `PUID` and `PGID` in `.env` to the owner of the host `data/` directory, then build and verify
the container:

```bash
printf 'PUID=%s\nPGID=%s\n' "$(id -u)" "$(id -g)" >> .env
docker compose --profile tools build toolbox
docker compose --profile tools run --rm toolbox
docker compose --profile tools run --rm toolbox \
  python -m scripts.container_healthcheck --readiness --media-smoke
```

The media smoke check writes a real H.264 file through `Mp4Writer` and decodes it again with
OpenCV. Readiness validates the mounted directory, schema version, camera configuration, and
threshold configuration. The production service healthcheck also requires a fresh watcher
heartbeat at `/app/data/live/watcher.heartbeat`.

Create the Telethon session once in an interactive terminal. It is written to
`data/session.session` on the host and is excluded from both Git and the image build context:

```bash
docker compose --profile bootstrap run --rm session-bootstrap
```

Run existing workflows in the same reproducible image, for example:

```bash
docker compose --profile tools run --rm toolbox python scripts/meta_backfill.py
docker compose --profile tools run --rm toolbox \
  python scripts/download_clips.py --camera cam06 --limit-per-camera 10
docker compose --profile tools run --rm toolbox python scripts/backtest.py --labelled-only --no-record
```

Back up `data/perimeter_watch.db`, its `-wal`/`-shm` companions, `data/session.session`, and any
irreplaceable source clips before upgrades. Rebuild with `docker compose --profile tools build
--pull toolbox`; the bind-mounted data survives image replacement. Stop active writers before a
filesystem-level SQLite copy, or use SQLite's backup API.

Before the first watcher start on an existing schema-v7 database, run all additive migrations,
then start the default service:

```bash
docker compose --profile tools run --rm --build toolbox python -m scripts.migrate_schema_v8
docker compose --profile tools run --rm --build toolbox python -m scripts.migrate_schema_v9
docker compose --profile tools run --rm --build toolbox python -m scripts.migrate_schema_v10
docker compose up -d --build watcher
docker compose ps
docker compose logs -f watcher
```

An existing schema-v8 deployment needs v9 followed by v10; an existing v9 deployment needs only
v10. Stop the watcher and back up the SQLite database before migrating; do not start newer watcher
code against an older database schema.

`watcher` restarts unless stopped and is healthy only while configuration, SQLite, persistent
storage, and its heartbeat are healthy. `toolbox` and `session-bootstrap` run only through their
profiles. For an upgrade, stop `watcher`, back up SQLite and the session, rebuild, run any new
explicit migration with `toolbox`, and start `watcher` again. Pending events and known failed
deliveries resume. Schema v9 adds staged preliminary, final, and resolution deliveries plus the
event resolution state. A startup incident can therefore produce an early warning, a confirmation,
or a clearly marked conflicting/likely-resolved update without replacing the original evidence. A
schema-v10 system-event outbox durably delivers maintenance/security notifications and retries
bounded failures independently of camera events. Any delivery interrupted after its external call
began is marked `ambiguous` for manual review so a restart cannot post a duplicate.

The default 300-second Initial-sibling window is measured from the local corpus: 8,268 of 8,274
paired events completed within it, and 99.9% completed within 286 seconds. Stopped/Timeout clips
finalize immediately. Override `EVENT_WAIT_SECONDS` only with newer measured evidence.

### Telegram alert and query bot

Create the bot once in Telegram with `@BotFather` (`/newbot`) and put the returned token in
`TELEGRAM_BOT_TOKEN`. The same bot can post urgent clip alerts and answer private commands. Add it
to the alert destination, grant permission to post, and set `ALERT_CHANNEL_ID` if alert delivery is
wanted. Never paste the token into source files, logs, or chat.

The bot enforces `BOT_TRUSTEE_IDS` and `BOT_SECURITY_IDS` in every handler. Trustees and security
can open the grouped inline interface with `/menu` or `/start`. The top level contains Monitoring,
Events, System, Site, and Info sections. Camera and history entries are selectable buttons. Every
visible slash command performs an immediate action: `/menu`, `/tonight`, `/health`, and `/about`.
Specialized status and parameterized commands stay in the inline menu. Existing
direct commands remain supported: `/about`, `/contacts`, `/tonight`, `/health`, `/power [count]`,
`/batteries`,
`/panel [count]`, `/faults [count]`, `/animal`, `/animals [count]`, `/incidents [count]`,
`/residents [count]`, `/neighbours [count]`,
`/history <animal|incident|resident|neighbour> [page]`, `/event <id>`, `/debug <video-id>`,
`/map`, and
`/last <camera_id>`; only trustees
can use `/patrols`. Power history separates failures from restorations and collapses identical
repeated notifications. `/history` groups Initial/Stopped sibling clips into events and lists a
stable representative message ID; `/event` retrieves that chosen video. Battery status pairs
low/restore messages where possible and flags unresolved device warnings for confirmation at the
alarm panel.
Direct `/history` results include one-tap video buttons labelled with the local date, time, and
camera, so mobile users do not need to copy message IDs.
Every proactive alert, event, or camera video sent by the bot includes a **Debug view** button. It
renders the annotated detector video on demand with the clip's timestamp-specific camera geometry,
reference background, and complete production motion configuration. Renders are cached under
`data/live/debug/{camera_id}/` using an input fingerprint, so repeated taps are fast and detector
or geometry changes invalidate the old result. `/debug <video-id>` provides the same operation
directly.
Warmup frames use the scorer's own brightness-corrected greyscale view. A solid yellow box is the
object used by the warmup feature; dashed cyan boxes passed the gates but were not selected, and
dashed grey boxes were excluded by a configured ignore region or the size gate. The HUD reports
all three counts. For offline review, `scripts/render_debug.py --message-id ID` still selects the
clearer event sibling; use `--exact-message-id ID` when auditing the named clip itself with its
database timestamp, geometry, and reference background.
To send clearly marked examples using the standing labelled animal and incident clips, run
`uv run python scripts/send_test_alert.py --examples`. The normal no-argument form remains the
text-only Telegram and ntfy credential test.
Panel history reports arm/disarm transitions; faults report tamper, supervision/device-missing,
and control-room communication-test failures. Media commands restore a cleaned clip from the
source channel through Telethon when it is not local. Media counts are capped at five per request.
Incoming maintenance/system notifications remain stored in `system_events`, are available via
`/health`, `/power`, `/batteries`, `/panel`, and `/faults`, and are also sent proactively through a
separate durable outbox. Telegram receives every recognized transition. ntfy uses urgent priority
only for incident alerts; animal and system-event notifications, including failures and recovery
updates, use default priority. Detector-only `blinding_foreground` maintenance findings remain in
the offline maintenance review queue rather than paging Telegram or ntfy.
The Events menu provides paged Animal, Incident, Resident, and Neighbour histories. Telegram and
ntfy receive all four live categories. Only incidents use urgent ntfy priority; animal, resident,
and neighbour observations use default priority. This camera routing is separate from the
system-event notifications described above.
Info contains the system description and an environment-configured security contact directory.
Fill `SECURITY_COMPANY_NAME`, `SECURITY_COMPANY_PHONE`, `CONTROL_ROOM_PHONE`, and
`ARMED_RESPONSE_PHONE` in `.env`; blank values display as `Not configured` until the real details
are available. Store phone numbers in international `+27...` format for mobile clients.

After adding or changing system-message parsing, safely import recognized non-video history without
touching clip rows:

```bash
docker compose stop watcher
docker compose --profile tools run --rm toolbox python -m scripts.sync_system_events
docker compose up -d watcher
```

Media retention is **disabled by default** for analysis/development machines. Storage-constrained
servers can opt in with `MEDIA_RETENTION_ENABLED=true`; cleanup then runs at startup and hourly by
default. It keeps human-labelled or live-finalized animal/incident evidence, pending live events
until they are resolved, and the newest local video for each camera. Other database-referenced
videos under the configured data directory are deleted and their `file_path` is cleared; message
metadata remains, so a bot request can retrieve the video again. The regenerable `data/live/debug`
cache follows the same policy: keep animal/incident and latest-camera renders, retain only the
newest renderer/fingerprint variant for each kept clip, and remove other managed renders. Paths
outside the data directory are never deleted. Configure the interval with
`MEDIA_RETENTION_INTERVAL_SECONDS`.

To restore cleaned media on a development machine, stop the watcher so it releases the Telethon
session, then run the resumable bulk restore. Successful files are committed one at a time, so the
same command resumes safely after interruption or rate limiting:

```bash
docker compose stop watcher
docker compose --profile tools run --rm toolbox python -m scripts.restore_media
docker compose up -d watcher
```
The map is deliberately an approximate ordered list, and patrol output carries a confidence
caveat. Unlisted users receive only `Not authorized` and their numeric user ID is recorded in the
structured watcher log. This provides a bootstrap path: configure the token, message `/about`,
read the denied ID from `docker compose logs watcher`, then add that ID to the appropriate
comma-separated allowlist and restart the service.

```bash
docker compose up -d --build watcher
docker compose logs --tail=100 watcher
docker compose --profile tools run --rm toolbox python scripts/send_test_alert.py
```

When an alert transport is configured after events were already finalized, the restart recovery
step creates missing delivery rows for animal/incident camera events. Existing delivered or
ambiguous rows are not duplicated.

## Setup and calibration workflow

### 1. First Telegram auth

The first time any script touches Telethon, it will prompt interactively for your phone number
and login code, then persist a session file at `TELEGRAM_SESSION_PATH`. Do this once, in a
terminal you control directly (not piped through anything that would swallow the prompt).

### 2. Metadata-only backfill (Phase 0a)

```bash
uv run python scripts/meta_backfill.py
```

Walks the full channel history and records **metadata only** — no video downloads yet — into the
`clips` and `system_events` tables: camera id (resolved from the caption via
`config/cameras.yaml` aliases, or `unknown` if unresolved), timestamp, and caption text. Camera
health notifications (battery, power loss/restore) are captured separately in `system_events`.
Safe to re-run; it's idempotent (upserts on `channel_id, message_id`).

If most clips resolve to `unknown`, add the real caption patterns you're seeing as aliases in
`config/cameras.yaml` and re-run.

### 3. Camera-order inference (Phase 0b, gate 1)

```bash
uv run python scripts/infer_camera_order.py
```

Infers the fence order purely from patrol timing patterns in the backfilled metadata — segments
patrol passes, builds a transition-time matrix, and seriates it (spectral + reversal-minimising
local search, cross-checked against an independent greedy-chain ordering). Prints:

- the inferred order and reversal cost,
- cross-check agreement between the two independent methods,
- entry-point candidates (should match your 2 known gates),
- any co-located or unplaced cameras,
- split-half stability.

**This is a hypothesis, not an authority.** Confirm it against your own mental map of the fence
before writing it into `config/cameras.yaml`'s `order` fields. If agreement is low or entry
points don't match the known gates, don't proceed to the spike — revisit the backfill data first
(enough patrol passes? gaps too large? `max_gap_seconds`/`min_cameras` tuned for your patrol
cadence?).

### 4. CV feasibility spike (Phase 0c, gate 2)

This is the step that decides whether classical CV is viable at all before any of the rest of
the pipeline gets built.

1. Pick 2-3 cameras with the richest incident history, ideally including whichever camera caught
   the known crawl incident.

   **Found from the real backfilled data** (see `data/perimeter_watch.db`) by matching your
   recollections against clip timing patterns — an isolated single-camera trigger reads as a
   probe/animal, a dense multi-camera sweep in a tight window reads as a guard patrol:

   - **Crawl incident, 2026-07-21 — visually confirmed on `cam06`**: nightly clip volume jumped
     from a 30-day average of 12.6 to 28.65 (2.3x, sustained) starting this night, unlike every
     other spike day in the history which reverts the next night. **The incident is
     `data/history/cam06/21520.mp4`** (paired with the "Initial" alert `21519.mp4`), 20:31-20:36
     UTC — a low, crouched silhouette is visible at the base of the IR-lit diagonal fence rail,
     clearest from around frame 10 onward. That sighting is what kicks off the wider response: a
     dense multi-camera sweep follows across cam06, cam09, cam10, cam01b, cam05, cam04, cam07,
     cam03 through 21:41 UTC as guards react. `cam01b`'s earlier isolated trigger at 20:11-20:16
     UTC (message ids `21517`/`21518`) turned out to be unrelated ambient activity, not the
     incident — see below. **Important orientation caveat**: unlike the other perimeter cameras,
     `cam01`/`cam01a`/`cam01b` face the opposite way — the open grassy area in frame is the
     complex/interior side, not outside the fence — which is what led to the initial
     misidentification. Frame-by-frame review of `21517.mp4`/`21518.mp4` shows only the ambient
     grass/fence scene, a faint distant light of unclear origin, and then a torch flooding the
     frame from the guard/interior side — no intruder visible there. (Earlier passes here
     mistakenly cited `21491.mp4`, from the wrong night entirely, then mistakenly read the faint
     light in `21518.mp4` as the intruder — both corrected; the real incident clip is `cam06`'s
     `21520.mp4`.)
   - **2024-03-16 `cam04`/`cam13`/`cam12`/`cam07` re-trigger night — retracted, not a probe**:
     `cam04` re-triggered 7 times between 19:33 and 03:47 the next morning, with `cam13`
     re-triggering 3 times, plus single triggers on `cam12`/`cam07` — looked like a probe timing
     pattern. **Visual review disagreed** across all 4 cameras (8-16 sampled frames each, all 24
     clips): no person in any of them, just fence/foliage scenes with a small bright green blob
     recurring near the lens on `cam12`/`cam13`/`cam07`, i.e. the IR-insect false-positive pattern
     the plan calls out (probably an insect-heavy night after rain), not an intruder.
   - **`cam10` 2024-03-14/15 re-trigger night — retracted, not a probe**: `cam10` alone
     re-triggered 9 times over ~4 hours; visually it's a floodlit pole/ladder-style fence
     structure with dense foliage, and the ambiguous shape near the bottom-left of
     `4048.mp4`/`4049.mp4` is wind-agitated ground texture (storm/gusty night), not a person —
     the pattern was a false positive despite looking clean in the timing data.
   - **Probe, 2024-03-16 — visually confirmed on `cam08`**: identified from a WhatsApp forward a
     trustee sent, received 2024-03-16 00:56 local, saying he'd "just picked up" someone putting
     on a backpack. The closest matching trigger is **`data/history/cam08/4054.mp4`** (paired
     with `4055.mp4`), 2024-03-16 00:47:30-00:50:18 local (msg ids `4054`/`4055`) — a few minutes
     before the WhatsApp receipt, consistent with forwarding delay. Confirmed.
   - **Animal, 2026-01-06**: tree fell over the fence during the day (cameras don't run then); an
     animal was seen climbing it "while still light out". `cam05`'s very first alert of the night
     fired at 18:21 SAST, right at dusk startup — the earliest trigger of any camera that
     evening.
   - **Animal, 2025-07-15**: `cam15` fired a single isolated trigger at 02:28 SAST, deep in the
     night, no other camera active nearby in time.
   - **Animal, 2024-08-26**: `cam08`'s first alert of the night fired at 18:04 SAST, again right
     at dusk, "not as dark out" per recollection.

   All 3 animal sightings, the crawl incident, and the probe are visually confirmed against the
   downloaded footage. `docs/plan.md`'s site facts mention 2 probes, but only one (`cam08`) had
   enough independent evidence to pin down; the second isn't being chased further. Timing
   patterns alone aren't reliable enough on their own — every purely timing-based candidate this
   session (`cam04`/`cam13`/`cam12`/`cam07`, `cam10`) turned out to be a false positive; a visual
   check (or independent ground truth like a forwarded message) is required before trusting a
   candidate.
   Recommended spike cameras: `cam06` (incident), `cam08` (probe + animal), plus `cam05` (animal,
   dusk-lit so easier to see).
2. Download a small clip subset for those cameras and record `clips.file_path` automatically:
   ```bash
   uv run python scripts/download_clips.py --camera cam01b --camera cam07 --camera cam05 \
       --since 2026-07-20 --until 2026-07-22 --limit-per-camera 15
   ```
   Pulls from rows already in `clips` (from step 2 above) that don't have a file yet, downloads
   each message's video via Telethon to `data/history/{camera_id}/{message_id}.mp4`, and updates
   `clips.file_path`. Drop `--since`/`--until` to widen the window (e.g. to also pull ordinary
   guard/animal nights for contrast, not just the incident window) — `--limit-per-camera` still
   caps it to a spike-sized sample, not a full backfill.
3. Hand-enter fence polylines for those cameras in `config/cameras.yaml` (`fence`, `outside`,
   `depth_cutoff`; 2-4 points is enough to start).

   **How to draw one** (no editor yet — see Phase 2 step 24 in `docs/plan.md`):
   1. Extract a still frame from a downloaded clip:
      ```bash
      uv run python scripts/extract_frame.py data/history/cam08/7360.mp4 --out /tmp/cam08.png
      ```
      Defaults to the clip's middle frame; pass `--frame-index N` if that frame is unhelpful
      (e.g. too dark, or before/after the subject enters frame).
   2. Trace the fence line in red, by hand, on that frame. Upscale it first so it's drawable
      (`cv2.resize(..., interpolation=cv2.INTER_NEAREST)` at 4x), open it in any paint tool
      (Snipping Tool, Paint, Preview), draw a red line along the fence, and save over the file.

      **Do not try to read the coordinates off the image by eye.** These are low-contrast IR
      frames and it is genuinely hard to tell the fence from a lit cable, a spider thread, or a
      guy wire — three separate attempts on `cam05`/`cam06` locked onto the wrong feature before
      this approach was adopted. A human tracing the line directly is both faster and correct.
   3. Recover the polyline from the red pixels — exact, no eyeballing:
      ```bash
      uv run python -c "
      import cv2, numpy as np
      img = cv2.imread('/tmp/cam05.png')
      h, w = img.shape[:2]
      b, g, r = (img[:,:,i].astype(int) for i in range(3))
      mask = (r > 150) & (g < 100) & (b < 100) & (r - g > 60) & (r - b > 60)
      ys, xs = np.where(mask)
      for i in range(3):  # top, middle, bottom samples
          y = ys.min() + (ys.max() - ys.min()) * i / 2
          band = (ys >= y - 3) & (ys <= y + 3)
          print(round(xs[band].mean() / w, 4), round(y / h, 4))
      "
      ```
      Three points captures the slight curve of a hand-drawn line; use more if the fence bends.
   4. Work out `outside`: pick any point you know is outside the fence (vegetation, sky) and
      check which side it falls on:
      ```bash
      uv run python -c "
      from src.zones import side_name
      fence = [(0.03, 0.96), (0.47, 0.06)]   # your normalised points, in order
      print(side_name((0.78, 0.42), fence))  # a point you know is outside the fence
      "
      ```
      Whatever it prints (`left` or `right`) is your `outside` value: plain screen position --
      point x compared to the fence's x at that same row. Point order doesn't matter (reversing
      it gives the same answer), but re-run this check after any edit to the points anyway.
   5. Set `depth_cutoff` to the row (as a `y` fraction) beyond which perspective makes the fence
      line too thin/distant to reliably judge a side — 0.0-0.1 is typical for a camera looking
      down a long fence run.
   6. Draw the resulting polyline back onto the clean frame and look at it before committing, as
      a final check that the extraction landed where you intended.

   `cam08` in `config/cameras.yaml` is a fully worked example from `data/history/cam08/7360.mp4`
   (dusk frame, fence line clearly visible running bottom-left to top-middle). `cam05` and
   `cam06` were both produced with the red-trace method above.

   Note that cameras can shift on their mounts over time, so a polyline is only valid for the
   era it was drawn in — `docs/plan.md` (Geometry model) covers the dated-versioning scheme this
   needs before Phase 2.
4. Label ~150 clips, oversampling animal/incident:
   ```bash
   uv run python scripts/label.py --camera cam_north
   ```
5. Extract features to CSV:
   ```bash
   uv run python scripts/spike.py --camera cam_north --out data/reports/spike_cam_north.csv
   ```
6. Inspect the CSV by hand (or a notebook). Does any threshold combination separate
   guard/environment from animal/incident at usable precision? Explicitly count flashlight and
   IR-insect clips landing outside the fence. **If separation fails, stop and revisit scope**
   (earlier ML, IR/brightness handling, multi-frame reference modelling) rather than pushing on
   to Phase 1.

### 5. Manual alert transport test

`src/telegram_alert.py` and `src/ntfy_alert.py` are pure, injectable-transport functions (mocked
in tests, never touch the network there). Verify your real credentials with one test message on
each channel:
```bash
uv run python scripts/send_test_alert.py
```
Uses `TELEGRAM_BOT_TOKEN`/`ALERT_CHANNEL_ID` and `NTFY_BASE_URL`/`NTFY_TOPIC`/`NTFY_TOKEN` from
`.env`; either pair is skipped (not failed) if unset.

### 6. Backfill, label, backtest, and live operation

These paths are implemented. Use `scripts/meta_backfill.py` and `scripts/download_clips.py` for
source history, `scripts/label.py` for review, and `scripts/backtest.py` for measured classifier
changes. The production entry point is `python -m scripts.watch`, run by the default Compose
`watcher` service. The role-restricted query bot runs in that same process when
`TELEGRAM_BOT_TOKEN` and an allowlist are configured.

The authoritative operating and next-work summary is `docs/next_agent_handoff.md`.
`docs/plan.md`, `docs/handoff.md`, and the dated review documents preserve the investigation and
decision history; their older future-tense sections are historical rather than current setup
instructions.

## Fail-safe classification policy

Bounding-box aspect ratio for a crawling person overlaps large animals — classical CV cannot
reliably tell them apart from a short clip. So: shape/trajectory evidence may only **escalate** an
outside alert to `outside_priority`. Nothing about blob shape is ever allowed to downgrade or
suppress an outside alert. The implemented classifier and event resolver preserve this policy.

## Repo layout

```
config/            cameras.yaml, thresholds.yaml
src/               reusable runtime/domain code: motion, scoring, classification, DB,
                   configuration, calibration, references, parsing, and alert transports
scripts/           operator/offline CLIs: ingestion/backfill, labelling, backtesting,
                   review rendering, reference builds, migrations, and diagnostics
tests/             unit/integration suite; automated tests use fake transports and no live credentials
docs/              current plan, handoff, capability map, and historical findings
data/history/      backfilled source clips, grouped by camera (local, gitignored)
data/live/clips/   source-channel clips kept for live operation (local, gitignored)
data/live/debug/   regenerable production debug cache, grouped by camera (local, gitignored)
data/reference_bg/ derived production references (local, gitignored, reproducible)
data/reports/      generated review/analysis artifacts (local, gitignored)
data/reports/debug/YYYY-MM-DD/topic/camera/
                  deliberate human-review renders; `unfiled/` is the CLI fallback
data/reports/debug/archive/
                  preserved legacy debug sets; do not add new work here
data/reports/scratch/
                  disposable experiments and intermediate output
data/backups/      retained DB/label backups (local, gitignored)
data/logs/         retained operational logs (local, gitignored)
```

Keep operational cache files in `data/live/`, deliberate visual reviews in a dated/topic
`data/reports/debug/` directory, and short-lived experiments in `data/reports/scratch/`.

The dependency direction is deliberate: `scripts` may import `src`, but `src` must never import
`scripts`. `tests/test_architecture.py` enforces that boundary. Some scripts are necessarily
substantial because they implement interactive review or report orchestration; the production
watcher uses the reusable APIs under `src/`.

## Sensitivity note

Patrol-compliance analytics profile identifiable individuals' movements and working patterns.
The `/patrols` command is trustee-only. Guards are data subjects under POPIA, so trustees should
agree the retention and access scope before enabling the bot for routine use.

# Nature Ridge Perimeter Watch

Motion-based perimeter fence monitoring built from a Telegram security-camera feed. This round
covers discovery, feasibility validation, and calibration tooling — **not** a live alerting
service (see [docs/plan.md](docs/plan.md) for the full plan and what's explicitly deferred).

Two gates decide whether the rest of the pipeline gets built:

1. **Gate 1 — camera order.** The fence camera order is unknown and must be inferred from patrol
   timestamp patterns alone (`src/sequence.py`).
2. **Gate 2 — CV feasibility.** Classical computer-vision features must actually separate
   guard/environment clips from animal/incident clips on a hand-labelled sample (`scripts/spike.py`).

Either gate failing changes the plan rather than getting silently worked around.

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

## Workflow

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

   - **Crawl incident, 2026-07-21**: nightly clip volume jumped from a 30-day average of 12.6 to
     28.65 (2.3x, sustained) starting this night, unlike every other spike day in the history
     which reverts the next night. `cam01b` alone triggered at 20:11-20:16, then a dense
     multi-camera guard sweep followed at 20:31-21:41 across cam06, cam09, cam10, cam01b, cam05,
     cam04, cam07, cam03.
   - **Security probe, 2024-03-16**: no volume spike, but `cam04` re-triggered 7 separate times
     between 19:33 and 03:47 the next morning — a lingering, repeated re-trigger pattern on one
     camera, not a sweep — with `cam13` re-triggering 3 times nearby in time, and single triggers
     on `cam12` and `cam07`. Matches the plan's probe signature (no multi-camera sequence, just
     dwelling near one point on the fence).
   - **Animal, 2026-01-06**: tree fell over the fence during the day (cameras don't run then); an
     animal was seen climbing it "while still light out". `cam05`'s very first alert of the night
     fired at 18:21 SAST, right at dusk startup — the earliest trigger of any camera that
     evening.
   - **Animal, 2025-07-15**: `cam15` fired a single isolated trigger at 02:28 SAST, deep in the
     night, no other camera active nearby in time.
   - **Animal, 2024-08-26**: `cam08`'s first alert of the night fired at 18:04 SAST, again right
     at dusk, "not as dark out" per recollection.

   That's 1 crawl + 2 probe-pattern cameras (cam04, cam13) + 3 animal sightings (cam05, cam15,
   cam08) — matching the site facts in `docs/plan.md` exactly. Recommended spike cameras:
   `cam01b` (incident), `cam04` (probe), plus `cam08` or `cam05` (animal, dusk-lit so easier to
   see). Confirm against your own memory before committing labelling effort.
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
3. Hand-enter fence polylines for those cameras in `config/cameras.yaml` (`fence`, `far_side`,
   `depth_cutoff`; 2-4 points is enough to start).

   **How to draw one** (no editor yet — see Phase 2 step 24 in `docs/plan.md`):
   1. Extract a still frame from a downloaded clip:
      ```bash
      uv run python scripts/extract_frame.py data/history/cam08/7360.mp4 --out /tmp/cam08.png
      ```
      Defaults to the clip's middle frame; pass `--frame-index N` if that frame is unhelpful
      (e.g. too dark, or before/after the subject enters frame).
   2. Open the PNG and pick 2-4 points along the fence line itself, in pixel coordinates, in
      either order (top-to-bottom or bottom-to-top — just be consistent for step 4).
   3. Normalise each point to `(x / image_width, y / image_height)`, both in `[0, 1]`.
   4. Work out `far_side`: pick any point you know is outside the fence (vegetation, sky) and
      check which side it falls on:
      ```bash
      uv run python -c "
      from src.zones import side_name
      fence = [(0.03, 0.96), (0.47, 0.06)]   # your normalised points, in order
      print(side_name((0.78, 0.42), fence))  # a point you know is outside the fence
      "
      ```
      Whatever it prints (`left` or `right`) is your `far_side` value.
   5. Set `depth_cutoff` to the row (as a `y` fraction) beyond which perspective makes the fence
      line too thin/distant to reliably judge a side — 0.0-0.1 is typical for a camera looking
      down a long fence run.

   `cam08` in `config/cameras.yaml` is a fully worked example from `data/history/cam08/7360.mp4`
   (dusk frame, fence line clearly visible running bottom-left to top-middle).
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
   IR-insect clips landing on the far side. **If separation fails, stop and revisit scope**
   (earlier ML, IR/brightness handling, multi-frame reference modelling) rather than pushing on
   to Phase 1.

### 5. Manual alert test (Phase 4 stub, no gate dependency)

`src/telegram_alert.py` and `src/ntfy_alert.py` are pure, injectable-transport functions (mocked
in tests, never touch the network there). Verify your real credentials with one test message on
each channel:
```bash
uv run python scripts/send_test_alert.py
```
Uses `TELEGRAM_BOT_TOKEN`/`ALERT_CHANNEL_ID` and `NTFY_BASE_URL`/`NTFY_TOPIC`/`NTFY_TOKEN` from
`.env`; either pair is skipped (not failed) if unset.

### 6. Backfill, label, and backtest (Phase 1+, after both gates pass)

Not built this round — see `docs/plan.md` for the full Phase 1-5 plan (full video backfill,
`src/motion.py`/`src/classify.py`, the backtester and threshold-iteration loop, and the later
trustee/security query bot).

## Fail-safe classification policy

Bounding-box aspect ratio for a crawling person overlaps large animals — classical CV cannot
reliably tell them apart from a short clip. So: shape/trajectory evidence may only **escalate** a
far-side alert to `far_side_priority`. Nothing about blob shape is ever allowed to downgrade or
suppress a far-side alert. This must be preserved in `classify.py` when it's built.

## Repo layout

```
config/            cameras.yaml, thresholds.yaml
scripts/           CLI entry points (meta_backfill, infer_camera_order, label, spike)
src/               config, db, sequence, zones, features, message_parsing, errors
tests/             pytest suite (unit tests only; no live Telegram/video needed)
docs/plan.md       full implementation plan, scope decisions, and what's deferred
```

## Sensitivity note

Patrol-compliance analytics (later phase) profile identifiable individuals' movements and working
patterns. Guards are data subjects under POPIA — worth a word with trustees on retention and
access scope before that bot ships.

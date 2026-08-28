# Handoff: Phase 0c labelling and the gate-2 decision

Written 2026-08-28 for an agent picking this up fresh. `docs/plan.md` is the full plan and stays
authoritative; this is the short version of where things actually stand and what to do next.

## Where the project is

Everything is on branch `feat/phase0-foundations`. Working tree clean, 147 tests passing
(`uv run ruff check . && uv run pytest -q`).

Phase 0 gates all downstream work. Gate 1 is resolved (see below); gate 2 has a written finding
pending user sign-off.

- **Phase 0a (metadata backfill) — done.** 16,887 clips in `data/perimeter_watch.db` with camera,
  timestamp, caption. Camera roster (`cam01`-`cam16`, plus `cam01a`/`cam01b`) is populated in
  `config/cameras.yaml` from real captions.
- **Phase 0b / gate 1 (camera-order inference) — resolved without inference (2026-08-28).**
  User decided fence order is just numerical/alphabetical by camera id; `order` is set directly
  in `config/cameras.yaml` (cam01=0 ... cam16=17). `scripts/infer_camera_order.py` still exists
  for optional later validation but is off the critical path.
- **Phase 0c / gate 2 (CV feasibility spike) — in progress, steps 1-3 of 6 done.**

## Phase 0c status

| Step | State |
| --- | --- |
| 1. Pick spike cameras | done — `cam06` (crawl incident), `cam08` (probe + dusk animal), `cam05` (dusk animal) |
| 2. Download clips | done — 142 clips on disk (`cam05` 50, `cam06` 46, `cam08` 46), spanning 2023-2026 |
| 3. Fence polylines | done — all three cameras have `fence`/`outside`/`depth_cutoff` in `config/cameras.yaml` |
| 4. Hand-label ~150 clips | **tooling ready, 0 rows labelled — waiting on the user to run it** |
| 5. Run `scripts/spike.py` | blocked on step 4 |
| 6. Read the CSV, decide gate 2 | blocked on step 5 |

## The next task, and the blocker that was in front of it

Step 4 is hand-labelling. The user does this, not the agent — it is ground truth about their own
property. The agent's job is to make it painless.

**Blocker fixed (2026-08-28):** `src.db.iter_unlabeled_clips` now takes `with_file_only: bool =
False`. `scripts/label.py` defaults its CLI to `with_file_only=True` (pass `--include-no-file` to
get the old metadata-only behaviour walking all 16,887 rows). It also surfaces the known rare-class
clips from `README.md` section 4 first (`PRIORITY_MESSAGE_IDS` in `scripts/label.py`: cam06
21519/21520, cam08 4054/4055/7360, cam05 18269) so the crawl incident, probe, and animal sightings
come up early in the session instead of possibly not at all if the user stops partway through.
Priority clips are still labelled by the user, not pre-filled — this only changes order.

The user confirmed: no configured video player integration needed, printing the path is enough
(and it's clickable via OSC 8 terminal hyperlinks in most terminals -- confirmed working). Each
prompt also shows single-letter shortcuts (`g`/`a`/`i`/`e`/`u`) and a one-line example per label,
since typing the full word every clip was slow.

**Short/blank clip handling (2026-08-28):** many triggers send a very short (<2s, empirically
0.2-1.8s in the downloaded set) near-blank clip immediately, followed ~3-5 minutes later by the
real clip -- the camera waking up, usually nothing visible. `--short-clip-seconds` (default 2.0)
flags these via `_clip_duration_seconds` (cv2 frame_count/fps); `default_prompt` shows the
duration and a note. After `--confirm-short-count` (default 3) of them get labelled the same in a
row, the user is asked once whether to bulk-apply that label to the rest of the session's short
clips without reviewing each one. Priority clips (see above) are never eligible for this bulk
path even if short -- cam06's `21519` ("Initial" alert) is itself a short clip but is a known real
event, not blank.

**Ready for the user to run:**
```bash
uv run python scripts/label.py --camera cam06
uv run python scripts/label.py --camera cam08
uv run python scripts/label.py --camera cam05
```
or omit `--camera` to go through all three spike cameras' downloaded clips (142 total) in one
session, priority clips first per camera. Labels are one of
`guard`/`animal`/`incident`/`environment`/`unknown`/`startup` (`src.db.VALID_LABELS`). `startup`
(added 2026-08-28, schema_version 2) is for the short (<2s) near-blank clip many triggers send
before the real clip a few minutes later -- `scripts/label.py` flags these by duration and can
bulk-confirm a run of them.

Once ~150 clips are labelled, the remaining Phase 0c steps are still blocked in sequence:

## Things that will bite you

- **Do not read fence coordinates off an image by eye.** These are 320x240 IR night frames. Three
  separate attempts on `cam05`/`cam06` locked onto a bright diagonal cable instead of the fence,
  including after zooming and overlaying a coordinate grid. What worked: give the user a clean
  upscaled frame, they trace the fence in red in a paint tool, then colour-threshold the red
  pixels back out. Full recipe is in `README.md` section 4 step 3. Reuse it for any further
  cameras.
- **`outside` is direction-relative.** It depends on which way the polyline runs, not on absolute
  screen position, so reversing the point order flips `left`/`right`. Recompute with
  `src.zones.side_name` against a known-exterior point every single time the points change. It
  has silently flipped twice already.
- **Timing patterns alone do not identify incidents.** Every purely timing-based incident
  candidate in this repo's history turned out to be a false positive on visual review (see the
  retractions in `README.md` section 4). Always confirm visually or against independent ground
  truth.
- **No migration framework.** Schema changes mean: export labels to JSONL, delete the DB,
  reimport. Once the user has hand-labelled 150 clips, that data is expensive — get the label
  export path working before any schema change touches the `labels` table.
- **`cam01b`/`cam16` face the opposite way** to the other perimeter cameras: the open ground in
  frame is the interior, not outside. (Corrected 2026-08-28: `cam01`/`cam01a` actually face the
  *same* way as most cameras — the earlier note blaming the whole cam01 family was wrong.) This
  caused a misidentified incident once.
- **STALE (superseded 2026-08-28): `cam15` is deliberately left without `fence`/`outside`.** This
  is no longer true — cam15 now has real fence geometry (see
  `/memories/repo/nature-ridge-conventions.md`). The original reasoning below is kept for
  history only: it's pointed down at a fence post close-up, foliage both sides, camera
  reportedly loose/moving in the wind — no guard has ever shown up in ~1.5 years of sampled
  clips and the spot likely isn't walkable.

## Known open items, not yet scheduled

- **Fence polyline versioning.** Cameras can shift on their mounts, which moves the fence line in
  frame. `docs/plan.md` (Geometry model, and step 24) documents that `fence`/`outside`/
  `depth_cutoff` need to become a dated per-camera history keyed off clip timestamp. Not needed
  for the spike; required before Phase 2's zone editor and live classification are trustworthy.
- **Compass orientation.** The exterior/outside of the fence is compass south property-wide,
  noted at the top of `config/cameras.yaml`. Captured as documentation only — the per-camera
  `left`/`right` values still do the actual geometry work.
- **Gate 1 (camera order)** resolved 2026-08-28 — numerical/alphabetical by id, no inference run.

## Conventions

- `uv` for everything: `uv run ruff check . --fix && uv run pytest -q` after every module change
  and before every commit.
- Commit logically per change, stay on `feat/phase0-foundations`, never auto-merge.
- `src/` is flat, no package. `scripts/` are standalone entry points that `sys.path`-insert the
  repo root.
- DB timestamps are UTC; site local time is UTC+2 (SAST).
- Secrets live in `.env` — never grep or cat it, and never scope a search broadly enough to
  match it.

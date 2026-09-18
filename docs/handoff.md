# Handoff: animal-event regressions fixed; detector extraction still next

Written 2026-08-28, updated repeatedly since; last updated 2026-09-17. **If you are a new agent
picking this up, search this file for "Handoff for a new agent (2026-09-17, session #19)"
and start there.** (This file's session sections are not in one consistent order: #1-#5 are the
oldest, kept in their original forward-chronological spot further up; starting from #6, each new
session's entry is instead inserted directly above its predecessor, so the chain from #6 to the
latest reads newest-first. #17 is the current latest.) On branch
`feat/phase2-refactor` (cut from `main`, which has all of Phase 1 and Phase 2's empirical
detection/classification work merged). The detection work itself is in a good, validated,
actively-improving state — see session #16's entry for the latest detection review and object
linking work, and session #17 below for the new persistent extraction cache. The cache's complete
identity prevents stale reuse; the remaining plan-step-22 refinement is extracting the detector
itself and replacing the zone-aware feature payload with genuinely zone-independent raw tracks.
See `docs/plan.md`'s "Phase 2 refactor brief" for that distinction. `docs/plan.md` is the full plan;
this file is the short version of
where things actually stand and what to do next. `docs/detection_improvement_review.md` is the
authoritative record of everything session #16 measured and shipped, with its own itemised
"Implementation status" section at the top.

## Handoff for a new agent (2026-09-17, session #19)

Continued the raw-track extraction without changing detector behaviour. `src.motion.GeometryObservations`
is a compact JSON-safe DTO for the fence-side evidence only: selected contour vertices, genuine
per-frame contour vertices, centroid path, frame dimensions, and persistent multi-object boxes.
It excludes images, masks and OpenCV contours. `scripts.spike.geometry_observations_from_detection()`
reduces an in-memory `ClipDetection` to that DTO; `geometry_features_from_observations()` replays the
eight current fence/depth features (`outside_*`, multi-object outside readings, crossing and median
fence distance) after a fence/side/depth edit without opening video. The normal extractor now uses
that same replay path, and tests round-trip its JSON payload and prove exact agreement with normal
feature extraction.

This is intentionally not the final raw cache yet: colour/texture, warmup and metric-calibration
features still require the in-memory imagery/contours, and `zone.ignore` remains a detector input.
Full tests: 723 passed. Incident/animal regression: all 5 incident events and all 8 animal events
pass with the existing documented cam10/7632 exception; the labelled warm run remains 29 TP / 22 FP
/ 18 FN (F1 0.592). The next safe seam is compact metric observations (genuine contour bounding
boxes plus frame indices), then a cache record can combine those geometry observations with stable
image-derived scalars.

**In progress 2026-09-17:** `src.motion.detect_clip` is now the public detector entry point;
debug/zone-rendering callers use it directly. It currently delegates to `scripts.spike._detect_clip`
while the large tracking-helper cluster moves in behaviour-preserving slices, so this is API
ownership rather than a completed physical move. `scripts.backtest --reference-background-primary
--no-record` is a separate, uncached experimental mode: it differences scored frames against an
aligned cross-clip reference to test recovery of subjects absorbed by a short clip's own median.
It remains off by default and must be judged on the labelled corpus before becoming a detector
option. The first targeted check (the five labelled animal clips currently reporting `no_motion`:
cam05/9692, 18269, 18271 and 18916; cam12/9720) found a genuine but insufficient result: all five
now produce detector features, but all five classify as `environment_candidate`. It therefore
recovers foreground without yet selecting/describing the animal reliably; do not enable it as a
default or count it as an animal-recall improvement. The next experiment, if pursued, must compare
competing reference-derived candidates rather than merely substitute the whole difference mask.

**Progress, next physical slice:** `src.motion` now owns contour selection, contour centroids and
the corner-form bounding-box primitives (area, IoU, centre, size plausibility and relative search
margin). `scripts.spike` re-exports them for compatibility while `track_contour` and the larger
tracking passes still use them. The next slice also moved template reacquisition and recovered-box
contour construction to `src.motion`; the high-level single/multi-object tracking passes remain in
`scripts.spike`. `track_contour`, the single-subject continuation policy (overlap, size-scaled
distance, area plausibility and flashlight override), now also lives in `src.motion`; the next
substantial extraction is `_run_track_pass`, which owns the state-machine sequencing around it.
That pass is now active from `src.motion` too, along with its patch-similarity/scenery guard; the
previous `scripts.spike` body is retained temporarily as a private parity oracle until its active
equivalent has a clip-level comparison harness. Full tests remain 723 passing and the incident/
animal regression check remains green after the move.

## Handoff for a new agent (2026-09-16, session #18)

Closed the three unaccepted animal-event regression failures surfaced in session #16. The
classifier now has three narrow recovery branches, all measured through the existing rule order:
`near_fence_animal` recovers cam10/9405 plus three other labelled animal clips; its compact-blob
ceiling excludes the daylight wind in cam15/9644. `fence_straddle_no_colour` uses temporal outside/crossing evidence to
recover cam15/15454's 50/50 fence-line porcupine and is the only newly alerting full-corpus clip in
that branch; `inside_elevated_animal` handles perched daylight animals before the generic
ground-plane implausible-height gate and reaches only the cam10 bird event plus cam05/18788 in the
16,886-clip report.

Labelled-corpus delta: exactly 7 category changes, all animal clips moving into the alert channel.
Alert precision/recall/F1 move from 0.500/0.468 to 0.569/0.617/0.592. `scripts/check_incident_regression.py` passes all 5
incident events and all 8 animal events, with only the already signed-off cam10/7632 exception.
The debug renderer also now rounds fractional dead-reckoned multi-track boxes before OpenCV drawing
and mask slicing; this is renderer-only and does not change cached feature semantics.

The next architectural work remains session #17's raw-track detector extraction. The operational
feature cache is correct and fast; changing geometry still invalidates/recomputes instead of cheaply
reapplying zones to a zone-independent raw-track payload.

**Progress 2026-09-16:** the detector-owned in-memory DTOs (`TrackedObject`, `FrameDetection`,
`ClipDetection`) now live in `src.motion` and are re-exported by `scripts.spike` for compatibility.
`render_debug.py` consumes them from their new home. This is deliberately only the first seam:
`ClipDetection` still contains NumPy frames/masks/contours and is not a cache payload; detector
implementation and zone-specific feature reduction remain in `scripts.spike` pending a compact
serialisable raw-track design.

**Progress, second seam:** `scripts.spike.features_from_detection()` now re-scores an existing
`ClipDetection` for a zone without reopening the video or calling `detect_clip`. The standard
`extract_clip_features()` path and this helper share the same established scoring body via a private
bridge, so output remains byte-identical while a future raw payload gains a real consumer. The only current
non-reusable geometry input is `zone.ignore`, because it is intentionally applied during detection;
fence/side/depth/calibration are consumed at scoring time.

## Handoff for a new agent (2026-09-13, session #17)

Picked up plan step 22, `motion.py` caching. Clean baseline: ruff green and 688 tests passing.
The operational feature cache is now complete. The stale MOG2-shaped `motion:` YAML was replaced
by all 22 real detector settings; `MotionThresholds` loads it strictly, and both detector/extractor
signatures are coupled to those values by a test. `src/motion.py` defines
`EXTRACTOR_VERSION=motion-features-v1`, builds a cache identity from video bytes, motion config,
resolved camera geometry, daylight hint, and reference-background pixels, and persists both real
feature mappings and explicit no-motion results through the existing `blob_tracks` table.
`scripts/backtest.py` uses it by default; `--no-cache` bypasses reads and writes.

Corpus verification on all 678 labelled clips: uncached, cold-cache and warm-cache CSVs had the
same SHA-256 (`899354925d35676f...`) and identical metrics (precision 0.500, recall 0.468). The
cold pass populated 678 `motion-features-v1` rows. Warm runtime was 3.7 seconds versus roughly
6 minutes uncached. Full tests: 695 passing. Incident/
animal regression output was unchanged: all 5 incident events alert; the same 3 already-known
animal event exceptions remain.

Honest remaining architectural refinement: this caches final JSON-safe feature mappings and
therefore includes zone/reference identity. It is correct and fast, but not the plan's ultimate
zone-independent raw-track payload—editing geometry invalidates and recomputes rather than cheaply
reapplying `zones.py`. Do not serialize `ClipDetection` wholesale (NumPy frames/masks/contours).
A later extraction of the detector from `scripts/spike.py` into `src/motion.py` should define a
smaller raw-track DTO; bump `EXTRACTOR_VERSION` when that payload or extraction semantics change.

## Where the project is

On branch `feat/phase1-finalisation` (branched off `main` at the `phase0-checkpoint` tag;
`feat/phase0-foundations` is retired but kept). 398 tests passing
(`uv run ruff check . --fix && uv run pytest -q`) as of 2026-09-05, with one deliberate
uncommitted change in `config/cameras.yaml` (cam06 fence base line — see session close #3).

Phase 0 is merged to `main` as a checkpoint, not a clean sign-off — see `docs/plan.md`'s
"Checkpoint (2026-08-30)" section. Gate 2 pass/fail is still the one open decision affecting
calibration/threshold work (Phase 3). That did not block Phase 1's foundation scaffolding (steps
15-19: project scaffolding, config/db, label JSONL durability, promoting Phase 0 scripts into
`src/backfill.py`/`src/sequence.py`), which is now done — see "Phase 1 foundation — done
below.

- **Phase 0a (metadata backfill) — done.** 16,887 clips in `data/perimeter_watch.db` with camera,
  timestamp, caption. Camera roster (`cam01`-`cam16`, plus `cam01a`/`cam01b`) is populated in
  `config/cameras.yaml` from real captions.
- **Phase 0b / gate 1 (camera-order inference) — resolved without inference (2026-08-28).**
  User decided fence order is just numerical/alphabetical by camera id; `order` is set directly
  in `config/cameras.yaml` (cam01=0 ... cam16=17). `scripts/infer_camera_order.py` still exists
  for optional later validation but is off the critical path.
- **Phase 0c / gate 2 (CV feasibility spike) — all 6 steps done, gate decision pending user
  sign-off.** Fence geometry was extended from the original 3 spike cameras to all 18 labelled
  cameras (2026-08-28), and the spike was rerun on that wider geometry plus a corrected
  background-subtraction detector (see "Things that will bite you" below). 281 labels rows as of
  2026-08-29 across all 18 cameras, not just the original 3 (see Phase 0c status table).

## Phase 0c status

| Step | State |
| --- | --- |
| 1. Pick spike cameras | done — `cam06` (crawl incident), `cam08` (probe + dusk animal), `cam05` (dusk animal) |
| 2. Download clips | done — 142+ clips on disk across the spike cameras, later downloads extended to more cameras |
| 3. Fence polylines | done — extended from 3 to all 18 labelled cameras (2026-08-28) |
| 4. Hand-label ~150 clips | done, and ongoing past the original milestone — 281 labels rows as of 2026-08-29: 174 with a known event class (guard 126, environment 22, unknown 11, incident 10, animal 5), 121 with only a `startup_state` so far (event class pending its partner clip's review — see schema v5 note below) |
| 5. Run `scripts/spike.py` | done — rerun 2026-08-30 with the corrected zone geometry (see below); `data/reports/spike_all_2026-08-30.csv` is current |
| 6. Read the CSV, decide gate 2 | **written finding done (`docs/gate2_separability_finding.md`), pass/fail call is the user's, not yet made — see staleness note below before signing off** |

## The next task: gate 2 pass/fail

`docs/gate2_separability_finding.md` has the full writeup. Summary of what it found:

**Staleness warning:** the doc's main body (label counts table, green-flashlight rule, shape/motion
table, the 53%/58% flashlight-false-positive count) was written 2026-08-28 on 110 guard clips with
the *old*, direction-of-travel-relative geometry -- it predates both a day of extra labelling
(guard is now 194 rows) and the 2026-08-30 `zones.py` fix. Only "Addendum 2 (2026-08-30)" is
current. A quick re-check on today's data found the coarse outside-majority rate barely moved (51%
vs. the old 53%), but that's not a controlled comparison -- label volume changed too. Rerun the
whole main-body analysis on `spike_all_2026-08-30.csv` before treating any of its specific numbers
(other than Addendum 2's) as the basis for the gate 2 sign-off.

- **Green flashlight is a strong guard identifier:** `green_light_ratio > 0.05 OR
  green_light_flicker > 0.02` catches 61% of guard clips, 2% of non-guard, 0/12 animal+incident.
- **Both predicted failure modes are confirmed and quantified:** 53% of guard clips have an
  outside-majority pixel fraction from geometry alone (flashlight beam crossing the fence line);
  55% of `environment` clips match the IR-insect signature.
- **Shape/motion separates animal/incident from guard/environment in the right direction**
  (aspect ratio, jitter, path length) but only 12 positive (`animal`+`incident`) clips exist
  against 132 negatives — too few to certify precision/recall. Best rule found: 75% recall / 22%
  precision (~2.7x the 8.3% base rate).
- **Recommendation in the doc:** usable as an escalation signal layered on outside geometry (the
  design `docs/plan.md` already specifies), not usable as a standalone classifier. Whether that
  clears the bar for "gate 2 passes" is an explicit call the finding leaves to the user.

Next step for whoever picks this up: get the user's pass/fail call on gate 2. If it passes (as an
escalation signal, per the recommendation), Phase 1 work can start. If not, the plan needs
revisiting before any further implementation.

## Things that will bite you

- **Do not read fence coordinates off an image by eye.** These are 320x240 IR night frames. Three
  separate attempts on `cam05`/`cam06` locked onto a bright diagonal cable instead of the fence,
  including after zooming and overlaying a coordinate grid. What worked: give the user a clean
  upscaled frame, they trace the fence in red in a paint tool, then colour-threshold the red
  pixels back out. Full recipe is in `README.md` section 4 step 3. Reuse it for any further
  cameras.
- **`outside` is plain screen position (fixed 2026-08-30).** `src.zones.side_name` used to be
  direction-of-travel-relative (depended on which way the polyline ran, flipping `left`/`right`
  if point order reversed) -- confusing enough that it was mistaken for a config bug once. It's
  now a plain point-x vs. fence-x-at-that-row comparison; point order no longer matters. Verified
  against real footage that this alone fixed cam06/cam09/cam10's known crawling intruders reading
  as "inside" with zero `cameras.yaml` edits. Still recompute with `src.zones.side_name` against
  a known-exterior point after any edit to the fence points, as a sanity check.
- **Guard/environment clips on cam06/cam09/cam10, and cam01b's 6 resident clips, still read
  `outside_pixel_fraction` inconsistently even after the above fix** -- same clip label, same
  camera, split between reading "inside" and "outside". That's a separate, still-open issue, most
  likely the fence line traced too high (top rail vs. base) rather than anything left/right --
  see the crawl incident's own description below (crouched silhouette at the base of the rail,
  not the rail itself).
- **Timing patterns alone do not identify incidents.** Every purely timing-based incident
  candidate in this repo's history turned out to be a false positive on visual review (see the
  retractions in `README.md` section 4). Always confirm visually or against independent ground
  truth.
- **No migration framework by default.** The documented fallback is: export labels to JSONL,
  delete the DB, reimport. In practice, once `clips` holds real backfilled volume (16,887 rows),
  that's disproportionate for a change scoped to one table -- `scripts/migrate_schema_v5.py` did
  an in-place rebuild instead (rename, recreate from the current schema, copy+transform, bump
  `schema_version`, keep the old table rather than dropping it), run via a raw `sqlite3`
  connection since `db.connect()` enforces an exact version match. Still export JSONL and
  file-copy the live `.db` first regardless, and stop every writer (background downloader AND
  any idle interactive `label.py` session sitting at a prompt) before migrating -- an idle
  session still holds an open db connection.
- **`cam01b`/`cam16` face the opposite way** to the other perimeter cameras: the open ground in
  frame is the interior, not outside. (Corrected 2026-08-28: `cam01`/`cam01a` actually face the
  *same* way as most cameras — the earlier note blaming the whole cam01 family was wrong.) This
  caused a misidentified incident once.
- **`cam15` has real fence geometry and is reversed-mount**, same as `cam01b`/`cam16`: open
  ground in frame is the interior side, not exterior (`config/cameras.yaml` is correct on this —
  `outside: left`). Rough terrain, confirmed animal sighting (2025-07-15, see
  `/memories/repo/incident-findings.md`), no human ever seen there in the sampled history.

## Schema v5 (2026-08-29): labels split into class + startup_state

`labels.label` used to conflate two different things: what the event was, and whether a
short/early pre-alert clip's own content was visible on its own. That collided for real — a
short clip auto-labeled `startup` (or hand-labeled `startup_clear`/`startup_blank`) silently
discarded the event's actual class whenever the paired, decisive clip hadn't been reviewed yet.
Found via a live-db audit: 123 startup-family rows, 107 with their partner still unlabeled.

Now `label` (nullable, 5 real classes) and `startup_state` (`clear`/`blank`/`duplicate`) are
separate columns. Labeling any clip in an event propagates its class to every sibling sharing
the same embedded alert timestamp that doesn't have one yet. `docs/plan.md`'s Ground truth
labels section has the full column semantics; `scripts/migrate_schema_v5.py` is the migration
that ran on the live db (158 rows unchanged, 16 startup-family rows inherited an already-known
partner class, 107 left `label=NULL` pending — those events' partner clips still need review,
which is the bulk of what's left in the current `--message-ids-file` shortlist session).

The frame-match rule for auto-detecting `startup_state=duplicate` was also tightened the same
day: the last decoded frame of a short/early clip is frequently a torn/truncated write from the
recording being cut off (not real content difference) — tolerated now, but only when every
earlier frame is a tight match, checked against a confirmed real incident (cam02#8699) that
fails the same way but with looser earlier-frame differences (moving subject, not a torn frame).

## Known open items, not yet scheduled

- **Fence polyline versioning.** Cameras can shift on their mounts, which moves the fence line in
  frame. `docs/plan.md` (Geometry model, and step 24) documents that `fence`/`outside`/
  `depth_cutoff` need to become a dated per-camera history keyed off clip timestamp. Not needed
  for the spike; required before Phase 2's zone editor and live classification are trustworthy.
- **Compass orientation.** The exterior/outside of the fence is compass south property-wide,
  noted at the top of `config/cameras.yaml`. Captured as documentation only — the per-camera
  `left`/`right` values still do the actual geometry work.
- **Gate 1 (camera order)** resolved 2026-08-28 — numerical/alphabetical by id, no inference run.
- **121 labels rows have only a `startup_state`, no event class yet** (schema v5 migration, see
  above) — their partner clip needs review. `scripts/label.py --message-ids-file` now pulls a
  shortlisted clip's partner into the queue automatically when this applies.
- **Real-time pairing logic is designed but not built.** For the live pipeline (not this Phase 0
  spike): pair a new clip with the last one for the same camera when their embedded alert
  timestamps match exactly (not a fuzzy time guess — the timestamp string is the real key), with
  a ~15min gap as a sanity bound only. Real pair gaps in the db: min 5s, median 3.2min, p95
  4.75min, max 8.6min across 8,274 pairs.

## Phase 1 foundation — done (2026-08-30)

Steps 15-19 of `docs/plan.md`. `src/config.py`/`src/db.py` (config loading, SQLite schema incl.
`blob_tracks`/`backtest_runs`/`backtest_results`, label JSONL export/import) already existed from
the Phase 0 spike, so there was nothing to add there. What was actually built:

- `src/logging_setup.py`: a `JsonFormatter` + `configure_logging()` on the root logger, replacing
  the old per-call-site `logger.info(json.dumps({...}))` pattern in `scripts/meta_backfill.py` and
  `scripts/download_clips.py` (now `logger.info(event, extra={...})`).
- `src/backfill.py`: `run_backfill`/`extract_primitives`/`RawMessage` promoted out of
  `scripts/meta_backfill.py`, which is now a thin Telethon-wiring wrapper around it — matching the
  existing `src/sequence.py` + `scripts/infer_camera_order.py` split (algorithm in `src/`, script
  keeps only I/O glue). `tests/test_meta_backfill.py` renamed to `tests/test_backfill.py`.
- "Console entry points" (from step 15) deliberately not added as real `[project.scripts]`: this
  repo is `uv init --app --no-package` (no `[build-system]`), so installed entry points aren't
  meaningful without converting to a packaged layout, which would contradict the flat-layout
  decision. Each script's own `main()` + `if __name__ == "__main__"` is treated as satisfying it.

Next up is Phase 2 (corpus and CV: camera-ID parsing hardening, resumable full backfill, cached
blob tracks, zone application, rule-engine classifier) — still gated on the user's gate 2
pass/fail call for anything threshold-dependent.

## Detection exploratory tools (2026-08-30/31)

Off the phased plan — a debugging/tuning pass on the spike-2 detector prompted by specific hard
clips (a camouflaged rooikat, a man whose accomplice's hand was drowning him out, a barely-visible
dassie). `scripts/motion_heatmap.py` (new) and `scripts/spike.py` both changed; 306 tests passing.

- **`scripts/motion_heatmap.py`** — standalone CLI, not wired into `spike.py`. Accumulates a
  per-pixel change signal across a whole clip and colorizes it, for judging where a marginal
  subject moved when per-frame detection only shows a dot. Normalises by the 90th percentile of
  *active* pixels, not the frame max — on the two-person clip a hand near the camera peaked ~4x
  the second man's whole body, so max-normalisation discarded 85% of the body's genuinely
  detected signal under the final display gate.
- **`detect_clip` anchor exemplar pass** (`_anchor_exemplar_index`/`_anchor_trace`, on by default
  via `anchor_refine=True`) — picks the real (non-appearance-recovered) detection closest to the
  clip's median tracked size and sweeps that one fixed crop forward and backward, filling frames
  the existing forward/backward passes leave with no box at all. 60 labelled clips: boxless
  frames 36 → 12, box-size jitter and centre-path smoothness unchanged.
- **Checked and rejected, with numbers recorded in the code so they aren't retried:**
  multi-scale template matching (box flickers between per-frame best scales); overwriting
  already-recovered boxes with the anchor exemplar (worse than leaving the drifting per-frame
  template alone); using the motion heatmap as a detection search prior (60-68% of frame lights
  up on vegetation-heavy clips — it's real wind motion, not signal); growing the tracked contour
  via a sliding-window "corroborated by neighbouring frames" test (only 2.2% of motion pixels sit
  within 20px of the tracked box across 60 clips, 47.3% sit >60px away — the existing
  `fragment_close_kernel_size` MORPH_CLOSE already recovers what's reachable).
- Debug clips re-rendered via `scripts/render_debug.py` after all of the above (2026-08-31):
  cam08/7360 (rooikat) and cam08/4053 (man) both track correctly; cam05/18270 (dassie) still only
  a 9px recovered box, expected given how faint its signal is; cam10/4042 correctly shows an IR
  flare frame with 12 spurious multi-tracks scattered across a bush, not a real subject — this is
  the same clip that lit up the motion heatmap and is why it was rejected as a prior.
- Full findings and every rejected variant's measurements: `/memories/repo/nature-ridge-
  conventions.md`.

**Anchor sweep runaway lock, fixed (2026-08-31).** User feedback after watching re-rendered debug
clips: cam08/7360 (rooikat) tracking is acceptable given the noisy footage; cam15/15454
(porcupine) recovery lingers slightly past when the subject is actually gone (mild); cam10/21524
(two men exiting frame) was the real bug — after they leave, the recovery locked onto a static
background patch and held it, unmoving, for the last 16 of 43 frames, having jumped there via a
fast, wrong-direction leap first. Root cause: `_anchor_trace` (added earlier this session) had no
cap on consecutive matches, unlike `_run_track_pass`'s existing `max_recovered_streak` — once the
real subject leaves the search window for good, nothing stops the fixed exemplar from matching an
unrelated static patch that merely resembles it, and because that patch never moves it keeps
re-matching itself at high confidence indefinitely. Fixed with a new required `max_streak` param
on `_anchor_trace` and a `detect_clip` kwarg `max_anchor_streak` (default 4). Verified on
cam10/21524: the false-locked frames are now correctly `None` (no box) instead of a confident
wrong answer. Measured on 60 labelled clips: geometry proxies are flat to slightly better than
baseline at this cap (jitter .2226→.2190, jerk 12.17→12.01) — capping the rare runaway case costs
nothing in the general population. Tried reducing `max_recovered_streak` itself first (12→4) —
rejected, it made general-population tracking measurably worse (jitter .2226→.320, jerk
12.17→18.70) because a shorter forward-pass streak more often hands off to the backward pass
instead, and that handoff discontinuity is itself worse than a longer single streak. Full
measurements: `/memories/repo/nature-ridge-conventions.md`.

## Guard/environment separation: environment_candidate rule added (2026-08-31)

Picked up the "next up" item below. Ground truth re-checked unchanged (guard 200, environment 22,
incident 10, animal 5, resident 6, unknown 15, 96 pending). Full detail and every rejected
alternative's measurements: `/memories/repo/nature-ridge-conventions.md`.

- The actual problem wasn't "environment reads as guard" — `guard_candidate` never fired on any
  environment clip. It was that neither class had a working positive identifier under the current
  tracker (`guard_candidate` recall only 8.5%, `insect_candidate` now fires on 0/22 environment
  clips — dead, the persistent-tracking work smoothed out the jitter it used to key on), so both
  landed mostly in `unclassified`, and 7/22 environment clips were polluting
  `animal_or_incident_candidate` on top of that.
- Also found (not fixed): `animal_or_incident_candidate`'s `aspect_ratio < 0.95` premise has
  inverted under the current tracker — animal+incident median aspect_ratio is now 1.24-1.41,
  higher than guard's 1.05, the opposite of the original gate-2 doc. Still fires on 74/200 (37%)
  of guard clips. Flagged in `scripts/backtest.py`'s docstring, not re-derived this pass.
- Fix shipped: new `environment_candidate` category in `scripts/backtest.py::classify`,
  `blob_count > 10` (max simultaneous motion blobs in a frame — the strongest univariate
  discriminator found, AUC 0.933 environment-vs-guard). Checked after `guard_candidate` but before
  the shape-based rules. Measured: 59% environment recall, 4.5% guard false-fire, **0/15 leak**
  against the full labelled animal+incident set (hard constraint held) — verified both via the
  sweep and a direct re-check against animal/incident alone. 3 spot-checked debug renders
  (`scripts/render_debug.py`) confirm the rule behaves as expected, including its known limit (a
  near-blank clip too short to accumulate the blob_count signal).
- 309 tests passing (was 306), committed on `feat/phase1-finalisation`.
- **Next natural step, not done this pass:** re-derive `animal_or_incident_candidate` now that its
  aspect_ratio premise is known-inverted — the 37% guard contamination of that candidate pool is a
  bigger unresolved problem than the guard/environment split was. Same measured-not-guessed
  discipline as this pass; re-check against the full incident+animal set before adopting anything.

## HUD timestamp, more debug clips, ignore-region wiring (2026-08-31, later same session)

Follow-on from the environment_candidate work above, picked up directly from the user reviewing
those debug renders. 316 tests passing.

- `scripts/render_debug.py`'s HUD title now includes the clip's embedded local (SAST) camera
  timestamp, reusing `scripts.label._EVENT_TS_RE` (already used for event-sibling grouping) rather
  than a new regex, e.g. `cam08/4292 guard @ 24-03-24 18:39:45`.
- 12 more guard/environment debug clips rendered across cameras not previously spot-checked
  (`data/reports/debug_render/guard_env_batch2/`), sampled across env_TP/env_FN/guard_TP/
  guard_unclassified/guard_FP_env categories for variety.
- **cam12/4033's edge-of-frame green, investigated**: measured (not guessed) via a per-pixel
  green-hue hit-frequency heatmap across all frames of the clip. 63-75% of green-hue pixels sit
  within 10px of the frame's top/bottom border (median edge distance 6-8px), concentrated in a
  band across the top and bottom edges of the right two-thirds of the frame — while the visually
  identical foliage filling the middle two-thirds of that same half of the frame shows almost no
  green hits at all. Cross-checked against a second cam12 clip (4032), same pattern. This is
  consistent with lens edge/vignette colour fringing (chromatic aberration, worse at the frame
  periphery regardless of scene content), not the guard's flashlight and not real foliage colour.
  Candidate ignore region (normalised): roughly `y < 0.30` and `y > 0.85`, `x > 0.35` — not yet
  written to `config/cameras.yaml`, pending user confirmation.
- **cam10's stationary light, investigated**: an attempted automated cross-clip photometric-
  persistence sweep (which pixel is bright in >50% of frames across 40 cam10 clips) was too broad
  to be useful (peak only 62%, huge bbox — picks up "generally lit foliage", not a specific
  fixture). Falling back to the clip the user actually flagged (cam10/4030) worked much better: a
  green glow just above the fence post, present in 12/13 frames, tight bbox normalised
  x:[0.37,0.50] y:[0.0,0.26]. Cross-checked against two more cam10 clips (4034, 4049) — same
  position each time (x:[0.37,0.50], y:[0.03,0.26]), confirming it's a genuinely fixed light, not
  a one-off. Candidate ignore region: normalised x:[0.35,0.52] y:[0.0,0.30] (padded) — not yet
  written to `config/cameras.yaml`, pending user confirmation.
- **`CameraZone.ignore` wired into detection for the first time.** It existed in `src/config.py`
  and `src/zones.py` (`in_ignore_region`/`classify_zone`, built for the future zone-classification
  pipeline) but `scripts/spike.py`'s actual blob-detection/feature-extraction loop never consumed
  it — confirmed zero references in `scripts/spike.py` before this change. Now:
  - `src/features.py::ignore_region_mask(frame_width, frame_height, ignore_polygons)` rasterises
    normalised ignore polygons to a boolean pixel mask (`cv2.fillPoly`).
  - `scripts/spike.py::detect_clip` takes `ignore_polygons=()`; when non-empty, the mask is zeroed
    out of every frame's motion diff before `cv2.findContours`, so a known fixed artifact region
    can never itself become a tracked blob or inflate `blob_count`/`motion_pixel_fraction`.
  - `src/features.py::green_light_ratio` takes `exclude_mask=None`; `extract_clip_features` builds
    the mask once from `zone.ignore` and passes it to both the whole-frame check (feeds
    `green_light_flicker`) and the tracked-subject check.
  - No camera currently has any `ignore` polygons configured, so this is a no-op today — all 309
    pre-existing tests still passed unchanged. New tests added in `tests/test_features.py` and
    `tests/test_spike.py` cover the mask rasterisation and the end-to-end suppression.
- **Not yet resolved: dusk/dawn/night time-of-day design.** User asked whether to use camera
  saturation or a seasonal sunrise/sunset curve to identify dusk/dawn/night. Saturation-only is
  likely a poor choice — already disproven by the "Daylight-gate fix investigated and ABANDONED"
  finding (2026-08-30): the guard's own flashlight raises whole-frame saturation the same way real
  ambient dusk/daylight colour does, and four saturation-based discriminators were all tried and
  failed to separate the two. A seasonal sunrise/sunset curve would sidestep this (doesn't depend
  on image content at all) but needs the site's actual latitude/longitude, which isn't stored
  anywhere in the repo (only the UTC+2/SAST offset is known) — needs to be asked for before
  implementing.
- **cam08 fence re-trace: done.** User traced clip 4292's fence-top over the 4x-upscaled reference
  frame; re-extracted via the standard red-trace/color-threshold recipe and verified with
  `scripts/visualize_zone.py` — line tracks the mesh post edge, not the diagonal cable distractor.
  `config/cameras.yaml`'s cam08 entry already reflects this.

## Camera mount drift confirmed + dated fence-history feature (2026-08-31, later same session)

User noticed cam01a's fence looked "way off" on clip 18124 but correct on clip 22635, and asked to
find where the camera moved. Bisected both cam01a and cam06 via `scripts/visualize_zone.py` over a
spread of historical clips (message_id + real DB timestamp), comparing scene framing frame by frame.

- **cam01a: confirmed hard remount**, precisely bounded: old framing (fence post on the left,
  vertical white plant/cable clutter) holds through message_id 21980 (2026-08-05T17:18:31Z); new
  framing (diagonal wall/roof edge, zoomed in) is already in place by message_id 21981
  (2026-08-05T17:22:15Z) — a ~4-minute window, same session, not gradual drift. User traced the
  new fence over the middle frame of clip 22635 (post-remount, daylight); extracted via the
  standard recipe: `[[0.4365, 0.1052], [0.1135, 0.999]]`.
- **cam06: inconclusive.** Sampled framing across message_ids spanning 2024-04-15 through
  2026-07-21 — post position/size looked broadly consistent, no obvious jump like cam01a's. Either
  there's no real remount, or the drift is too subtle to judge from compressed thumbnails (this is
  exactly why the repo's rule is "never read fence coordinates off a frame by eye" — turns out it
  applies to *verifying* alignment, not just tracing it). Not resolved; needs either a sharper
  quantitative method or the user's own side-by-side comparison.
- **Dated fence-history schema implemented** (`docs/plan.md`'s "Geometry model" open item, now
  built): `config/cameras.yaml` camera entries can use a `zones:` list of
  `{effective_from, fence, outside, depth_cutoff, ignore}` entries instead of flat top-level
  fields — mixing both on the same camera is a `ConfigError`. `effective_from` (ISO 8601, `Z` or
  offset) is optional on the first/oldest entry (`None` sorts first, "applies from the start").
  `Camera.zone` stays the current/most-recent geometry (every existing call site not yet updated
  keeps working unchanged); new `Camera.zone_at(timestamp)` picks the latest entry whose
  `effective_from` is at or before `timestamp`, falling back to `zone` if there's no history or the
  timestamp is `None`/unparseable. `scripts/backtest.py`, `scripts/rank_candidates.py`,
  `scripts/render_debug.py`, `scripts/spike.py` and `scripts/visualize_zone.py` (new `--timestamp`
  flag) all updated to call `zone_at(clip["timestamp"])` instead of the static `.zone`. cam01a is
  the first (and so far only) camera using `zones:`. 321 tests passing.
- **Not yet done:** cam06's drift is still unresolved (no dated entry added — would need a second
  reference trace once/if a boundary is found). No other camera has been checked for drift yet.

## Next up: storm and guard identification (session closed 2026-08-31, picked up above)

This session's tracking work (anchor sweep + its streak-cap fix, above) is done and committed.
The next session's focus, per the user: separating `environment` false-triggers (storm/wind/rain
shaking foliage or the camera) from real `guard` sightings — these are the two most confusable
classes on this corpus (see the "DAYLIGHT GATE IS SELF-DEFEATING" and "Daylight-gate fix
investigated and ABANDONED" findings in `/memories/repo/nature-ridge-conventions.md`, both about
this exact confusion, both worth reading before proposing a new discriminator).

- **There is no separate `storm` label.** User confirmed 2026-08-31: a storm trigger is just a
  cause of an `environment` clip, not its own `VALID_LABELS` entry — do not add one. A prior
  attempt at a storm-specific metric (fraction of frames with whole-frame-ish motion, tried
  against 4 user-flagged storm clips: cam15/9944, 9954, 9956, 18948) failed to separate them from
  ordinary guard/environment clips; see the `scripts/backtest.py` note in
  `/memories/repo/nature-ridge-conventions.md` before retrying that specific approach.
- **Plan for next session:** add more debug clips (`scripts/render_debug.py`) covering
  guard/environment cases, especially ones the current rules misclassify, and iterate on
  detector/feature/rule changes against them.
- **Hard constraint: do not regress incidents or animals while tuning guard/environment.** As of
  the last full count (`/memories/repo/nature-ridge-conventions.md`, 2026-08-30 section), ground
  truth had 10 `incident` rows and 5 `animal` rows against 200 `guard` / 22 `environment` — re-run
  the count at the start of the next session, don't trust this number as still current. User does
  not expect more incidents to turn up, but does expect the backtest (`scripts/backtest.py` /
  `scripts/rank_candidates.py`) to surface additional animal sightings over time — any
  guard/environment rule change should be checked against the full labelled incident+animal set
  (not just guard/environment precision) before being adopted, same discipline as the gate-2 and
  daylight-gate investigations already in memory.

## QA pass over guard_recent_long/guard_batch2, wrap-up (2026-09-03)

User did two full passes watching through the debug-render reference clips (`guard_recent_long/`
and a rebuilt `guard_batch2/`, both under `data/reports/debug_render/`, gitignored/local-only) and
reported one issue per clip. Full findings are in
`/memories/repo/nature-ridge-conventions.md` under "QA pass, cont. (2026-09-03)" (and the section
above it, "Wide QA pass ... (2026-08-31, cont.)" from the prior session) — read both before
picking this back up. One config fix landed this session (see below); everything else is a
confirmed finding or an open design question, not yet implemented.

**Only code/config change made:** `config/cameras.yaml` cam08 gained a SECOND `ignore` polygon
(a second stationary light found on the same camera, different clip than the one already
excluded). Verified via `uv run ruff check . --fix && uv run pytest -q` (334 passed).

**Recurring theme worth prioritizing next:** `max_recovered_streak=12` forcing a track reset,
followed by unconstrained re-acquisition onto whatever tiny or textured blob is nearby, hit 3
separate clips in one sitting (a mounting pole, a literal spider web strand, and a 3x3px noise
speck). No shared fix implemented yet — the single highest-value open item from this pass.

**Two design questions raised by the user, not yet designed or implemented:**
1. A per-camera `expect_illumination` config flag (verify a known light is actually visible in a
   given clip).
2. Auto-detecting stationary lights ("green + no movement") instead of hand-tracing an `ignore`
   polygon per light per camera — the robust version of this needs stability checked ACROSS MANY
   clips on the same camera over time, not just within one clip (a guard holding a flashlight
   steady for a short clip looks identical to a fixed light within that single clip alone).

Both connect to the same open background-model-architecture question raised again this session
for cam01/16167 and cam05/4822 (dwelling/slow-drift subjects the median-background diff can't
see) — worth one shared design conversation rather than three separate patches.

**Not investigated further, treated as already-resolved or user-confirmed:** cam01a/22635
(matches a prior R/G-ratio finding), cam03/21536, cam07/4097 (both user-confirmed-correct),
cam06/4875 (already-documented dilution finding from the prior pass), cam11/4310 (hue collision
checked, confirmed not a problem).

## Continued QA pass: dashed-box marking, hue fix, min_reacquire_area fix, new labels (2026-09-03, later same session)

Full technical detail is in `/memories/repo/nature-ridge-conventions.md` (search for the dated
subsections named below) — this is the short version. Validated throughout with
`uv run ruff check . --fix && uv run pytest -q`, which stayed green the whole way
(334 -> 345 passed by the end). All 51 clips across every standing `data/reports/debug_render/`
sample folder were re-rendered at the end, more than once, against the final code.

**Implemented and shipped:**
1. **"Mark instead of ignore" + auto-detect stationary lights.** `FrameDetection` gained an
   additive `suppressed_light_box` field (motion inside an ignore region, captured before it's
   zeroed for real tracking) — `render_debug.py` now draws this as a dashed box labelled
   "IGNORED (stationary light region)" instead of showing nothing. New
   `src.features.detect_stationary_light_mask()` auto-detects a fixed light per-clip (pixel
   green-lit in >=80% of frames, excluding the tracked subject's own region) and feeds both the
   render and `extract_clip_features`'s exclude_mask — a new light no longer needs a hand-traced
   `zone.ignore` polygon first. The existing STATIONARY LIGHT marker changed from a solid contour
   outline to a dashed box, matching the same "mark, don't hide" principle.
2. **cam01/16167 flashlight hue miss** — `green_light_mask`/`green_light_ratio`'s `hue_low`
   35 -> 33 (cam01's flashlight measured 34-35, one unit under the old cutoff). Validated against
   3 reference/false-positive clips, no measurable change.
3. **cam01/16167 light-drift fix — tried twice, both REJECTED.** Global brightness-shift
   compensation didn't reduce noise at all. High-pass illumination filtering did clean up the
   pole artifact but devastated real small-subject detection on 3 known-fragile animal clips
   (rooikat -80% hit pixels, porcupine below its area gate entirely, dassie -30-50%) — rejected,
   not shipped. cam01/16167's drift issue and the related cam05/4822 / cam11/4310
   wrong-artifact-tracked family remain open; the only still-credible direction is a real
   cross-clip/per-camera reference-background architecture change, deliberately not attempted
   without explicit sign-off given its corpus-wide blast radius.
4. **cam13/4101 min_reacquire_area fix.** New floor on `track_contour`'s fresh/unconstrained
   reacquisition path (clip start, or right after a `max_recovered_streak` force-drop) stops a
   dropped track from landing on a noise-speck-sized contour. This went through **two rounds of
   correction after a full-corpus reassessment caught real regressions** a narrow reference-clip
   validation missed (cam10's genuinely tiny, single-digit-pixel real animals were losing
   detection entirely, or worse, locking onto a wrong static artifact) — fixed by (a) only
   applying the floor once a real track was already established once, and (b) threading
   chronological knowledge from the forward pass into the backward pass so a trailing noise
   speck can't look like a legitimate "first-ever" pickup from the backward scan's own reversed
   point of view. Final default: `min_reacquire_area=20.0`. **Lesson recorded in repo memory:
   always sweep the full standing sample set (old vs new, all 51 clips) before trusting a
   `detect_clip`-level change, not just a handful of reference clips.**
5. **New/corrected labels** (channel -510921049): cam10/17146 (bird on fence), cam10/7632 (animal
   in flare region), cam10/9405 (animal, rocky wall), cam10/4308 (guard, added to
   `guard_batch2/`), cam10/18786+18787 (animal, event pair). cam03/9273 was briefly mislabeled
   `animal` then corrected to `guard` per user review — also confirmed a render-only false
   FLASHLIGHT overlay on that clip from dawn foliage sitting just under the 0.15 daylight gate
   (matches the already-documented, already-abandoned cam03 daylight-gate confound — not a new
   bug, not re-attempted). Confirmed-animal clips moved from `animal_check/` into
   `animal_recent_long/`; `animal_check/` now only holds the 2 still-`unknown` cam03 clips.

6. **Static ignore-region overlay removed.** The `zone.ignore` hatch + outline + "IGNORE" label
   was baked into the *static* per-clip `ink` layer in `_zone_layers`, so it drew on every frame
   regardless of whether the light was lit — including in broad daylight, which is what the user
   spotted on cam10. Removed. The region is still masked out of detection/scoring exactly as
   before; it is now only *visually* marked when something real happens there, via the
   already-per-frame-gated STATIONARY LIGHT and "IGNORED" dashed markers from item 1.
7. **cam12 remount — dated fence history added.** User's trace on `cam12/19245.mp4` (2026-03-02)
   did not match the configured fence, but every earlier sampled clip back through `12201.mp4`
   (2025-02-21) still did. cam12 triggers rarely, so there are no clips between those dates to
   bisect. `effective_from` is `2026-03-02T16:13:54Z` (message 19244, the earlier of the two
   same-night clips confirming the new position) — the earliest timestamp there is real evidence
   for. `outside: right` re-verified for the new polyline via `side_name`.
8. **Two investigations that correctly ended in "do not change anything":**
   - *cam04/4210* ("tracking 1 frame only, maybe the IR flare should be longer?"): median grey is
     already flat (~30) from frame 13 on, exactly matching the current `warmup_dropped=13`, so no
     flare is being missed and lengthening the window would only discard real frames. The actual
     problem is 13 consecutive appearance-recovered frames frozen on a static ~143px feature —
     the same static-texture-lock family as cam01/16167 and cam02/11264, and *above* the new 20px
     floor, so `min_reacquire_area` does not catch it.
   - *cam05/4822* ("maybe still too much IR flare, we can see the guard earlier"): the step that
     sets `warmup_dropped=8` is a genuine ~10-level IR gain jump (8 -> 18). Swept
     `flare_tolerance` 2.0/3.0/4.0/5.0 — settle index is 8 for every value in 3.0-5.0, and *worse*
     (11) at 2.0. There is no safe tolerance change available, and forcing a larger one risks
     accepting real ongoing flare as settled corpus-wide.

**Committed** in 6 logical commits on `feat/phase1-finalisation` (`258fa5e`..`2b05875`): config
ignore regions, detect-layer, render-layer, a leftover `rank_candidates.py` fix, docs, and the
cam12 dated fence. The working tree had been carrying all of this uncommitted since the previous
session — worth checking `git status` early next time.

## Handoff for a new agent (2026-09-04, session close)

**State.** Branch `feat/phase1-finalisation`, working tree clean, 367 tests passing
(`uv run ruff check . --fix && uv run pytest -q`), 21 commits ahead of `origin/main` and
unpushed. 432 label rows (guard 252, environment 39, unknown 18, resident 10, incident 10,
animal 10, plus 93 with only a `startup_state`). 52 debug renders across 6 folders under
`data/reports/debug_render/` (gitignored, local-only). 133 per-camera reference backgrounds in
`data/reference_bg/` (also gitignored) — rebuild with `scripts/build_reference_bg.py` as the
corpus grows.

**What changed on 2026-09-04.** The per-camera reference background (item 2 below) was built,
swept, kept and committed. The cam03 daylight confound (item 3) was partially resolved by a
sun-time gate. Neither is wired into scoring yet — see item A.

**Do not "complete" the incident sample set.** Every incident is a pair of Telegram clips: a short
startup/trigger segment followed by a longer continuation that already contains it. Only the
second of each pair belongs in `incident_recent_long/`. All 5 pairs are now represented
(cam08/4053, cam08/4055, cam06/21520, cam09/21522, cam10/21524); 4053 was a genuine omission,
fixed 2026-09-03. The 5 missing first-of-pair clips are correct to leave out.

**Read before touching detection:** `/memories/repo/nature-ridge-conventions.md` is the real
engineering log — every failed approach is recorded there with *why* it failed, and several
plausible-sounding ideas in this area have already been tried and rejected on measurements. The
two most expensive lessons from this session:

- **Validate `detect_clip`-level changes against the full 51-clip standing sample**, diffing old
  vs new per-frame boxes, not a handful of reference clips. A narrow validation passed cleanly
  this session while silently destroying detection on cam10's smallest real animals.
- **When the user corrects your reading of a frame, discard the evidence you gathered under the
  wrong assumption** rather than averaging it in. A second fence trace collected while misreading
  which line was which nearly ended up blended into the final config.

### Remaining work, in suggested order

Items A and B are the current priorities and were set on 2026-09-04. Items 1-5 below them are the
older list, kept because their measurements and rejected approaches are still the reference
record; 1 and 2 are now closed, 3 is partially closed.

**A. Identify `environment` correctly, using the reference background as the candidate source.**
This is the user's chosen next direction and the highest-value open item.

The reference-background veto shipped this session currently *throws away* the information it
computes. When it drops a frozen run because the box matches the per-camera reference, it has
just established something valuable: **there is motion here, and it is sitting on top of scenery
that is always there.** That is close to a definition of wind-blown vegetation.

The proposed signals, in the user's framing:

- *If it moves, it may be wind.* Motion that lands on reference-background regions is vegetation
  being shaken, not a subject. A candidate feature is the fraction of a clip's bg-diff motion
  pixels whose location matches the reference background above the `scenery_correlation`
  threshold.
- *If it isn't movement, track it across the screen.* A subject **translates**; wind
  **oscillates in place**. Net displacement over the clip, measured against path length, should
  separate a person walking a fence line from a branch swinging back and forth. Note this is
  distinct from the existing `path_length`/`jitter` features, which measure magnitude, not net
  direction \u2014 and which the batch-5 ablation found to be noise.
- *Larger surface areas.* Wind moves a whole bush; a subject is compact. Total area (or bounding
  extent) of scenery-matching motion is a third candidate.

**Read these three prior results before writing any code \u2014 two of them are near-misses on this
exact idea:**

1. **A closely-related attempt already failed.** An ad-hoc "fraction of frames with whole-frame-ish
   motion" metric was tried against the 4 known storm clips (cam15/9944, 9954, 9956, 18948) and did
   *not* separate them from guard/environment clips \u2014 some guard clips scored as high or higher.
   The reference background is the genuinely new ingredient (it distinguishes "motion everywhere"
   from "motion everywhere *on top of known-static scenery*"), but treat this as unproven, and
   expect to have to beat that prior negative explicitly.
2. **`motion_heatmap` was rejected as a detection prior precisely because of this behaviour** \u2014 it
   lights 60-68% of the frame on vegetation clips (cam10/4042: 52655 of 76800 px) because
   wind-blown foliage is real motion everywhere. That failure as a *prior* is exactly the signal
   wanted as an *environment feature*. The tool already exists and is already validated.
3. **The bar to beat is `blob_count`**, AUC 0.933 for environment-vs-guard, currently used at
   threshold 10 in `scripts/backtest.py::classify` for 59% environment recall at 4.5% guard
   false-fire and zero animal/incident leak. Any new feature must be measured against that
   baseline, and must be checked against the **full** animal+incident set \u2014 the user's standing
   constraint is that no tuning may lose an animal or an incident.

Practical caveats: there are only 39 environment labels, and just 4 environment clips in the whole
DB carry any notes, so labelling more wind/storm clips with notes is a cheap prerequisite. Four
of the standing sample's clips have no reference background at all (cam12/19245 by era,
cam11/4309, cam11/4310, cam09/21522 by sparsity), so any reference-derived feature needs a defined
value when no reference exists \u2014 do not let "no reference" silently read as "no scenery motion".

**B. Decide whether the reference-background veto feeds scoring.** It is currently wired only into
`scripts/render_debug.py`. `scripts/backtest.py` and `scripts/rank_candidates.py` still pass no
reference, so **no feature value has moved corpus-wide** and every recorded AUC remains
comparable. Turning it on there would change `persistence`, `longest_detection_run`,
`recovered_fraction` and the other provenance-derived features on the ~7% of clips it affects, and
requires a fresh leave-one-out AUC measurement to justify. This is a deliberate decision, not an
oversight \u2014 and item A may well settle it, since item A wants that information as a feature
rather than as a silent veto.

**1. The static-texture lock — ADDRESSED 2026-09-04 via item 2; kept for its rejected approach.**
A track gets sustained
indefinitely by appearance-matching against a fixed, high-texture background feature — a mounting
pole (cam01/16167), a spider-web strand (cam02/11264), a fence-post junction (cam04/4210) — while
the real subject is elsewhere or gone. `max_recovered_streak` and this session's
`min_reacquire_area` each catch a slice of it; neither catches these, because the false feature is
larger than the floor and re-matches its own template forever.

**Frame-local discrimination has now been measured and REJECTED — do not retry it.** The idea was
that a run of appearance-recovered frames whose box never moves *and* whose contents correlate
with the clip's own median background must be scenery rather than a subject. Implemented and swept
over all 52 standing clips: it changed 23 of them and gutted precisely the fragile-animal set it
was supposed to leave alone (cam10/17146 21->9 boxed, cam10/7632 17->5, cam10/9405 13->7,
cam08/7360 11->6). The measurement explains why, and the numbers are worth keeping:

| | frozen-recovered runs | background similarity |
|---|---|---|
| known static locks | 6 | 0.87 – 0.99 |
| real guards/animals | 54 | 0.54 – 1.00 (18 of 54 at >= 0.97) |

There is no separation. Worse, the *worst* target (cam04/4210, 11 frozen frames) sits at 0.87 —
**below** most genuine subjects. The reason is structural, not a tuning problem: a frame is only
appearance-recovered *because* bg-diff found nothing there, which already means its contents
resemble the background. High background similarity is therefore a property of every recovered
frame, subject or not. Run length doesn't separate either (12 of the 54 real runs are >= 11
frames). Any frame-local signal derived from "does this look like the background" is measuring the
precondition for being in that state at all.

That leaves the discriminator genuinely outside a single clip's median: a static feature is in the
*same place across different clips from the same camera*, and a subject never is. That is item 2's
per-camera reference background, which is why these two items are really one.

**Stage 1 of item 2 has now been measured, and it works — see below.**

**2. Per-camera reference background — BUILT 2026-09-03, pending keep/discard.** The signal that
item 1 lacked: score each frozen-recovered run against a reference background built from the 15
clips *nearest in time from the same camera and lighting*, excluding the clip under test
(nearest-in-time stands in for both camera era and season). Unlike the clip's own median, a
different clip's background carries no guarantee of resembling a recovered frame, which is what
breaks item 1's circularity.

It separates. Verified by eye on the review sheets in
`data/reports/investigations/static_lock_review/` (60 sheets, one per frozen run, each showing the
clip crop beside the reference crop at the same coordinates):

- Every high scorer inspected was scenery — cam02/4302 (fence rail, 13 frames), cam05/4695 (razor
  wire, 13), cam06/4875 (palisade, 11), cam05/4822 (rail corner, 13) — in each the clip crop and
  the reference crop are visually identical.
- Low scorers are real subjects: cam10/21524 at 0.143 is the crawling men, clip crop plainly
  unlike the reference.

**The scope is roughly four times what we thought.** At a 0.94 threshold, 18 of 60 frozen runs
across 15 of the 52 sample clips are scenery locks — ~139 wrongly-boxed frames — not the 4 clips
originally listed. Two specific consequences worth knowing:

- **cam05/4822's first 13 frames are locked on the fence rail**, which is exactly why the guard
  "appears late". That complaint was diagnosed twice as an IR-flare-window problem and it is not
  one; the warmup window was already correct both times.
- cam13/4091, cam07/4067, cam06/4701, cam08/4306, cam07/4097, cam04/4103, cam11/4309 all carry
  previously-unnoticed locks.

**Caveats to carry into stage 2.** The margin is real but not generous: the lowest flagged run is
0.944 and the highest unflagged is 0.938, and cam10/9405's genuine small animal sits at 0.922 —
about 0.02 of headroom. This is not the clean bimodal split item 1 hoped for, so the threshold has
to be chosen on a labelled set, not by eye. **Alignment is required, not optional**: phase
correlation lifted cam01/16167's two weaker runs from 0.847/0.844 to 0.953/0.954, and cameras
whose framing shifts between clips need it far more (cam01a/22635 measured a 36px shift,
cam14/21501 27px — without alignment those score as false locks).

**Stages 2-4 are now built and committed (`be45a20`, `6b45155`, `65f18f7`) — awaiting a keep/discard
decision.** What shipped:

- `src/reference_bg.py` + `scripts/build_reference_bg.py`. References are bucketed by camera, mount
  era, daylight and calendar quarter, sparse quarters folding into the nearest populated sibling;
  up to 40 clips sampled evenly per bucket. **133 references built** into `data/reference_bg/`
  (gitignored). `reference_for` returns `None` rather than crossing an era boundary — cam12's
  post-remount clips get no reference at all, which is correct.
- The veto in `scripts/spike.py`: `_run_track_pass(reference_background=..., max_scenery_streak=2,
  scenery_correlation=0.94)`. Applies only to appearance-recovered frames that are also
  near-stationary, and nulls the frozen run retroactively. With no reference passed the behaviour
  is byte-identical to before, which is the fallback for sparse cameras.
- `scripts/render_debug.py` resolves each clip's reference automatically; `--no-reference-bg`
  renders the old behaviour for comparison.

**Full-sweep result (52 clips, old vs new): 7 clips changed, 44 frames dropped, 0 gained**, 4 clips
had no reference (cam12/19245 by era, cam11/4309, cam11/4310, cam09/21522 by sparsity).

| clip | frames | lost |
|---|---|---|
| cam05/4695 | 16 -> 6 | 6-15 (razor wire) |
| cam02/4302 | 27 -> 18 | 18-26 (fence rail) |
| cam05/4822 | 15 -> 6 | 6-14 — **the "guard appears late" complaint, now actually fixed** |
| cam08/4306 | 43 -> 35 | 31-38 (dark ground under the palisade) |
| cam01b/18884 | 14 -> 9 | 9-13 (empty black corner) |
| cam06/4875 | 31 -> 29 | 11-12 (palisade) |
| cam04/4065 | 43 -> 42 | 29 |

All seven were inspected as clip-vs-reference crops and are scenery; renders for review are in
`data/reports/debug_render/reference_bg_check/`. **Nothing was gained, and no real subject was
lost.** It is much more conservative than stage 1 predicted (44 frames, not ~139) because the
production references are coarser than stage 1's nearest-15, era filtering removes cam12, and
`max_scenery_streak=2` needs 3+ consecutive frames.

Against production references the named targets score cam01/16167 0.99, cam04/4210 0.99,
cam05/4822 0.99; real subjects cam13/4101 0.57 and cam10/9405 0.72/0.90 stay below. **One known
miss: cam02/11264's 13-frame lock scores 0.922, under the 0.94 threshold.** Do not lower the
threshold to catch it by eye — cam10/9405's genuine animal is at 0.896, leaving only ~0.014 of
headroom. Pick any new value on the labelled set.

**cam11 has only 2 clips in the entire corpus, so cam11/4310 can never be fixed this way** — it
needs separate handling and should come off this item's list.

**3. cam03 daylight-gate confound — PARTIALLY FIXED 2026-09-04 (`2664c22`).** `color_fraction`
around 0.12-0.15 left dawn/dusk cam03 clips just under the 0.15 gate, so the green overlay drew a
false FLASHLIGHT marker. Four threshold-based fixes had already failed, all of them re-deriving
daylight *from the pixels* — which is exactly why they failed, since the guard's own flashlight
perturbs every such statistic the same way ambient colour does.

The fix was an **exogenous** signal: `src.features.is_daylight` (a month-keyed Pretoria
sunrise/sunset table that already existed but was only used as an eyeball column) is now OR'd into
`render_clip`'s `daylight_gated`. A clock and a date cannot be fooled by a flashlight. cam03/9237
fires at 05:54 local in November, 49 minutes after sunrise, at `color_fraction` 0.095.

Measured cost: **zero** — all 28 labelled clips that `is_daylight` flags already reported
`flashlight_subject_fraction` and `green_light_ratio` of exactly 0.000.

**Still open, and do not call this solved:** it fixes dawn/dusk only. cam03/9066 (02:54 local) and
cam03/8767 (23:55 local) carry the confirmed 0.42/0.30 corruption at genuine night and are
untouched. Note also that the sun table is a month-granular step function rounded to 5 minutes.

**4. Gate 2 pass/fail is still the one open decision.** See "The next task: gate 2 pass/fail"
above, including its staleness warning — the main body of
`docs/gate2_separability_finding.md` predates both the `zones.py` fix and roughly 150 additional
labels, so re-run its analysis on current data before signing anything off.

**5. Smaller, well-defined loose ends.** The two design questions from the prior pass — a
per-camera `expect_illumination` flag, and cross-clip stability checking for auto-detected
stationary lights (the current detector only checks within a single clip, which cannot
distinguish a fixed light from a guard holding a torch steady) — remain open and are cheap next
steps. cam11/4310 (flashlight colour + stationary light) needs separate handling: cam11 has only
2 clips in the entire corpus, so no cross-clip method can ever help it. cam03/9237 and 9239 were
reviewed on 2026-09-04 and deliberately left `unknown`; their notes record why, so don't
re-investigate them.

## Handoff for a new agent (2026-09-04, session close #2)

**State.** Branch `feat/phase1-finalisation`, working tree clean, 372 tests passing
(`uv run ruff check . --fix && uv run pytest -q`). Same label counts as the session above (guard
252, environment 39, unknown 18, resident 10, incident 10, animal 10, 93 startup-only).

**This session picked up item A from the section above** ("Identify `environment` correctly,
using the reference background as the candidate source") and got a decisive, well-explained
**negative** result for its first candidate signal. Full numbers and mechanism are in
`/memories/repo/nature-ridge-conventions.md` under "Environment-via-reference-background: signal
(a) MEASURED AND REJECTED (2026-09-04)" — read that before touching this again.

**Summary: naive per-blob reference comparison scores WORSE than random for environment-vs-guard
(AUC 0.42-0.45, vs. blob_count's existing 0.813-0.933 baseline on the same data), and it's not a
tuning problem.** `_patch_similarity` (NCC) is invariant to linear brightness rescaling, so a
guard's flashlight lighting up known-static scenery (a fence rail, the ground) still "matches"
the reference structurally, just brighter — while wind-shaken foliage genuinely changes the local
texture pattern and scores LOW. The signal points backwards for a real, structural reason. This
is exactly why the narrower, already-shipped version of this idea (the frozen-box scenery veto,
section above) works and this broader per-blob version does not: the veto additionally requires
the box to be near-stationary across several consecutive frames before comparing it to the
reference at all, which a translating illuminated guard never satisfies.

**What shipped anyway, harmlessly:** `scripts/spike.py` gained `scenery_motion_fraction` +
`has_reference_background` (the latter diagnostic-only, added to `rank_candidates.py`'s
`_NON_FEATURE_COLUMNS`) — purely additive dict keys / dataclass fields with defaults, zero
regression, 5 new tests. Kept as real infrastructure for a future refined attempt (see next
paragraph), but **not** wired into `scripts/backtest.py::classify` — doing so would make
guard/environment separation worse, not better. `environment_candidate` is unchanged
(`blob_count > 10`).

**Also quick-checked (no code shipped): net displacement / path length ("subject translates, wind
oscillates in place").** AUC 0.665 guard-vs-environment — right direction this time, but still
below the blob_count baseline and confounded by short tracks (exactly 2 genuine centroids always
scores ratio=1.0, trivially, regardless of what the subject actually did). Same family of noise
this repo's batch-5 ablation already found for `jitter`/`path_length` alone. Not implemented as a
real feature. "Total area of scenery-matching motion" (the third candidate signal) was not
separately tested — it shares signal (a)'s NCC-brightness-invariance flaw as its core ingredient.

**Credible next direction for item A, not attempted this session:** restrict the reference
comparison to blobs that are themselves near-stationary/recurring at the same location across the
clip (or across several frames within it), rather than scoring every transient blob regardless of
motion — i.e., borrow the veto's own "is this box actually holding still" gate rather than
applying the reference check to everything. That is the one structural difference between the
already-working veto and this session's failed broader attempt.

**Gate 2 pass/fail (item 4 above) is still the standing, unrelated open decision** — nothing this
session changes that.

## Handoff for a new agent (2026-09-05, session close #3)

**State.** Branch `feat/phase1-finalisation`, 398 tests passing. **One uncommitted change in the
working tree**: `config/cameras.yaml` cam06's `fence` field currently holds a quick-test BOTTOM
(base) trace, with the original TOP-rail line preserved in a YAML comment. This was only a fast
way to validate the base-line idea — `fence` is, and stays, the TOP rail for all 18 cameras
(unchanged meaning). **Do not commit this as-is.** When implementing the next feature, restore
`fence` to the commented-out top line and move the bottom trace into the new `fence_bottom`
field (see below) — see `/memories/session/plan.md`. Labels grew this session: guard 268,
environment 96, animal 10, incident 10, resident 10, unknown 18.

**Shipped this session (5 commits).**

- `1680f29` — cross-camera storm/environment corroboration. New `src/storm_events.py`
  (`ClipSignal`, `neighbor_cameras`, `find_corroborated_events` via union-find) and
  `scripts/find_storm_events.py`. A clip is only surfaced when spatial neighbours fire inside a
  time window, which is what wind/rain does and a single guard does not.
- `924937a` — continuous dawn/dusk signal. `src/features.py` gained
  `minutes_from_daylight_boundary()` (signed minutes from the nearer of sunrise/sunset) and
  `is_twilight()`. Also fixed cam03/18512's wrong flashlight overlay by OR-ing `is_twilight` into
  `render_debug.py`'s daylight gate.
- `4082444` — `--exclude-twilight-minutes` filter for `find_storm_events.py`.
- `fd2a886` — re-derived the broken `animal_or_incident_candidate` rule (the "next natural step"
  flagged back on 2026-08-31) and split it into `animal_candidate` / `incident_candidate`.
- `d254f8b` — recorded the fence-height calibration and flashlight-divergence design ideas in
  `docs/plan.md`.

**Findings worth not rediscovering.**

- *"The guard false-positives are all in daylight" was wrong, but pointed at something real.*
  `is_daylight()` returns `False` for all 8 of them; they sit 13–56 minutes **after** the sun
  table's January sunset (18:55 local). Real twilight, past a binary cutoff — hence the continuous
  signal. Bucketing guard clips by minutes-after-sunset shows the confound directly: +15→+60 min,
  n=21, mean `blob_count` 9.05, **38.1%** false-fire on `blob_count > 10`; +60→+120 min, n=20,
  mean 4.80, **0%**; +240 min and beyond, n=80, mean 4.40, 6.2%. Dusk grain, not wind.
- *The `aspect_ratio < 0.95` premise was measurably dead* (AUC 0.518). Re-derived on 378 labelled
  + detected clips: `outside_pixel_fraction` AUC 0.700, `median_fence_distance` 0.660. Neither is
  usable alone — `outside_pixel_fraction` alone false-fires on 40%+ of guards at *every* threshold
  from 0.3 to 0.9, because **guards genuinely read as "outside"** under top-rail geometry. Requiring
  both moved recall 26.3% → **73.7%** and guard false-fire 31.8% → **15.9%** (environment false-fire
  21.1% → 31.6%, accepted). Animal-vs-incident splits on `color_fraction > 0.15` (AUC 0.828): 0/10
  incidents exceed 0.04 (all night IR), 7/9 animals exceed 0.15.
- *`blob_count > 10` survived re-measurement* on the larger label set: AUC dropped 0.933 → 0.856,
  but at the shipped threshold recall is 54.2%, guard false-fire 6.72%, and the animal+incident
  leak is still **0/20**. Threshold unchanged.
- *Storm sweep, tightened settings* (30 min window, neighbour distance 3, `exclude_twilight_minutes=60`)
  produced 6 events, all already-confirmed environment, **zero false positives** — but two real
  storms (2024-02-08, 2024-03-05) drop out because their members land 41–57 min after sunset. The
  twilight filter is a precision/recall dial, not a free win.
- **Known bug, diagnosed but NOT fixed:** `blob_count` is a *max over frames*, so a single noisy
  leading frame trips it. cam04/10887 reads `[16, 5, 5, 4, 3, ...]`, cam03/11096 `[14, 6, 6, 2, ...]`,
  cam03/18512 `[20, 1, 1, 1, ...]` — all spike on the first scored frame then decay. Contrast a
  genuinely windy clip, cam01a/18347: `[0, 1, 0, 2, 6, 23, 10, 8, ...]`, peaking mid-clip and
  *staying* elevated. Likely leftover IR flare-settle grain that `flare_settle_index`'s single-step
  test doesn't clear; `motion_heatmap.py::_residual_settle_index` already does the multi-step
  version and was never ported into `detect_clip`. Fixing it is a detector-level change and needs a
  full-corpus re-validation, so it was left alone.

### The fence base-line finding (this is what the next feature is built on)

Every camera's existing `fence` polyline was traced along the **top rail**, and that stays true —
it is not being redefined. Cameras are mounted near the fence top looking down and slightly
outward, so a guard walking *inside* the fence still projects to the outside of the top-rail line
— which is exactly why `outside_pixel_fraction` false-fires on 40%+ of guards no matter where you
put the threshold.

Tracing a brand-new line for cam06 along the **base of the fence where it meets the ground** (the
first bottom line to exist anywhere in the config) and re-running `extract_clip_features` over all
45 labelled cam06 clips, using it in place of the top line for the outside-fraction calculation:
33 of 45 changed, most collapsing to `outside_pixel_fraction = 0.0`. The crawl incident
cam06/21520 stayed at **1.0** under both geometries. cam06/21519 went 0.75 → 0.0 but is
`startup_state='blank'`, so not meaningful. A few kept real spread (3441 1.0→0.722, 4732
1.0→0.741, 22647 1.0→0.839, 4875 0.516→0.169).

The collapse initially looked like signal loss. **It isn't** — the user's call, and it's right: a
guard patrolling inside the fence *cannot* be outside, so 0.0 is the correct reading. The bottom
line is the correct reference for the inside/outside decision, once it exists. cam06's bottom
trace is near-vertical because the camera sits on top of the fence looking down its own line; that
is the real geometry, not a tracing error. `outside: right` was confirmed directly by the user.

### Next feature: two fence polylines (top + bottom) and metric features

Planned this session, **not implemented**. Full plan in `/memories/session/plan.md`. Summary:

Add a second polyline per camera — the base/bottom line — while the existing `fence` field keeps
its name AND its existing meaning: the **top rail**, unchanged, for all 18 cameras. Inside/outside
classification switches to prefer the new bottom line when a camera has one, falling back to the
top line (`fence`) when it doesn't — so the 17 cameras with no bottom line yet keep behaving
exactly as today. The pair of lines then gives a per-row pixel ruler (the fence is ~2 m) that
unlocks real height / size / speed features and a "crossed into the fence band from outside"
signal.

Design decisions already made, with the reasoning:

- `fence_bottom: tuple[Point, ...] | None = None` and `fence_height_m: float = 2.0` go on
  `CameraZone` as the **last fields, with defaults**. There are 25 literal `CameraZone(...)`
  constructions across 9 test files (16 in `tests/test_zones.py`); a required field would mean
  touching all of them for nothing. "Suite green with no edits to those 25 sites" is the
  acceptance test for phase 1. Do **not** call it `fence_top` — `fence` already is the top line.
- Don't rename `fence`, and don't repurpose what it holds. It would touch every call site plus all
  18 YAML entries, 17 of which have never had a bottom line traced.
- `signed_side`/`side_name`/`classify_zone`/`outside_pixel_fraction`/`track_crosses_fence`/
  `median_fence_distance` in `src/zones.py` all switch to reading `fence_bottom` when present,
  else `fence` — this is the one deliberate behaviour change, scoped to whichever cameras get a
  bottom line.
- The ruler is the horizontal separation between the two lines at a given row, reusing the existing
  `zones._fence_x_at_y` for both. Checked against cam06's two real traces: separation is 0.048
  (normalised x) near the top of frame and 0.31 at the bottom — it converges with distance, which
  is what a receding fence must do.
- **Units trap:** normalised x and y are not the same scale (frames are 320×240). Convert to pixels
  via `frame_width`/`frame_height` before forming any ratio.
- Guard the degenerate case: when separation falls under a floor, or `fence_bottom` is absent,
  return `None` ("uncalibrated") rather than dividing and emitting absurd metres.

Phases: (1) schema + parsing — add `fence_bottom`, wire the fallback logic above, and **fix
cam06's YAML** (restore `fence` to the commented-out top line, move the current bottom trace into
`fence_bottom`); (2) draw both lines in `render_debug.py` and `visualize_zone.py`, pilot cam06
only — it's the only camera with both lines traced; (3) pure helpers in `src/zones.py` —
`fence_separation_at_y`, `pixels_per_metre_at_y`, `estimated_height_m`, `estimated_width_m`,
`estimated_speed_mps`, `subject_base_y`, all using the subject's **feet row** (bbox bottom) as the
depth reference; (4) `in_fence_band` / `entered_band_from_outside`, but **discovery first** —
nobody knows whether the corpus contains an actual outside-to-band crossing, so run the 10
incident + 10 animal clips through it and report the count before building a rule on it; (5)
sanity-check calibration against the 268 guard clips (`estimated_height_m` should cluster around
1.6–1.9 m — if it doesn't, the ruler is wrong, fix or abandon, do not ship) and only then sweep
thresholds.

**Do not touch `scripts/backtest.py::classify` until phase 5 has numbers.** Out of scope: retracing
all 18 cameras, a zone editor, full perspective homography.

**Confound to encode as a regression fixture:** the bird that sat on the fence, then moved away
very fast and got smaller. Its feet row is on the *top* line, not the ground, so the depth
assumption breaks and height/speed will read absurd. Worth a "feet above the base line ⇒
uncalibrated" guard.

### Still open, unchanged by this session

- Gate 2 pass/fail — the standing decision, still the user's to make.
- `guard_candidate` recall is 8.7%; the daylight gate is self-defeating and four fix attempts have
  failed. See `/memories/repo/nature-ridge-conventions.md` before attempting a fifth.
- **`resident` has no rule at all** in `classify()`. Dog detection is the obvious route but zero
  dog-labelled clips exist — examples are needed first.
- The first-frame `blob_count` spike bug above.

## Handoff for a new agent (2026-09-05, session close #4)

**State.** Branch `feat/phase1-finalisation`, 430 tests passing
(`uv run ruff check . --fix && uv run pytest -q`). Same label counts as session close #3.

**Shipped: phases 1-3 of the fence base-line plan above, plus phase 4 (discovery) and phase 5
(calibration sanity check).** Full detail in `/memories/repo/nature-ridge-conventions.md`
("Fence base line + metric features -- phases 1-3 IMPLEMENTED"); short version:

- Fixed the uncommitted cam06 YAML (`fence` restored to the top rail, the base trace moved into a
  new `fence_bottom` field). `CameraZone.fence_bottom`/`fence_height_m` added as trailing
  defaulted fields — zero edits needed to any of the 25 existing `CameraZone(...)` test
  constructions.
- `src/zones.py::effective_fence()` prefers `fence_bottom` when present, else `fence`;
  `classify_zone`/`track_crosses_fence`/`median_fence_distance` all switched to it. Every camera
  without a bottom line (17 of 18) is byte-identical to before.
- New pure, tested calibration helpers: `is_grounded_at_fence`, `fence_separation_at_y`,
  `pixels_per_metre_at_y`, `estimated_height_m`/`estimated_width_m`/`estimated_speed_mps`,
  `subject_base_y`. All return `None` ("uncalibrated") rather than a guess whenever a camera has
  no `fence_bottom`, the row is outside the base line's traced range, or the two lines' pixel
  separation is below a floor.
- Discovery-stage `in_fence_band`/`entered_band_from_outside` added, per the plan's explicit
  "discovery first" requirement. Checked against the only calibrated camera's ground truth (cam06
  has 1 incident clip, 0 animal clips): `entered_band_from_outside` is `False` on the crawl
  incident (21520) — its whole track reads "outside" and never enters the band at its own row.
  n=1, nowhere near enough to build a rule on.
- Phase 5 sanity check run against all 41 cam06 guard clips' genuine (non-recovered) frames:
  per-clip median `estimated_height_m` clusters at 1.66m overall, tighter (1.5-1.7m) on
  higher-frame-count clips — inside the 1.6-1.9m target band. **Verdict: the ruler is trustworthy
  for cam06, kept.**
- `render_debug.py`/`visualize_zone.py` now draw both lines when both exist (verified visually on
  cam06/21520 night + cam06/8467 daylight).
- **Deliberately not done, matching the plan's own gating**: no threshold sweep, nothing wired
  into `scripts/backtest.py::classify`, no other camera retraced. Only cam06 is calibrated.

**Natural next step if this is picked up again**: trace a `fence_bottom` for a second camera (a
guard-heavy one with enough labelled volume to be worth calibrating) before attempting any
threshold sweep — a rule built on a single calibrated camera can't generalise, and phase 4's
discovery result (n=1 incident) is too thin to judge `in_fence_band` on its own merits yet.

### Still open, unchanged by this session

- Gate 2 pass/fail — the standing decision, still the user's to make.
- `guard_candidate` recall is 8.7%; the daylight gate is self-defeating and four fix attempts have
  failed. See `/memories/repo/nature-ridge-conventions.md` before attempting a fifth.
  **SUPERSEDED 2026-09-07 — see session close #5. That 8.7% is this ONE rule's own recall, not the
  system's: overall guard recall is now 71.8%.**
- **`resident` has no rule at all** in `classify()`. Dog detection is the obvious route but zero
  dog-labelled clips exist — examples are needed first.
- The first-frame `blob_count` spike bug above.

## Handoff for a new agent (2026-09-07, session close #5)

**Read this section first — it supersedes #4 for current state.** 523 tests passing, branch
`feat/phase1-finalisation`, incident regression 5/5 events. Nothing is uncommitted.

### What this session built

Triage now has **three** independent outputs, not one queue:

1. **Incident/animal queue** — `scripts/rank_candidates.py --out ...`. Now excludes
   `startup_state` blank/duplicate rows, sub-1s clips that have an event sibling, and
   blinding-foreground rows.
2. **Maintenance queue** — `--maintenance-out ...`, `write_maintenance_candidates()`. A
   "someone must physically clean this camera" report. Deliberately includes already-labelled
   clips (a labelled guard clip can still show an obstructed lens).
3. `blinding_foreground` column in `scripts/backtest.py` output, independent of `category` —
   a clip can be a real guard sighting **and** have an obstructed lens.

New features, all measured against the full labelled corpus before shipping:

| feature | what it means | key numbers |
| --- | --- | --- |
| `blob_white_fraction` | tracked blob's overexposed-pixel fraction | animal max 0.182, incident max 0.310 → 0.4 threshold is zero-leak |
| `long_flare_frames` | raw `warmup_dropped` count (NOT the fraction — that's 30-60% in every class and useless) | ≥18 |
| `post_flash_red_shift` | R/G after the brightness peak minus before it | **only guard ever exceeds 0.10** (guard max 0.581, environment 0.014, animal 0.038, incident 0.001) |

### Where classify() actually stands (measured 2026-09-07, 534 labelled clips)

**Guard recall is 71.8% (270/376), not the 8.7% quoted elsewhere in this file** — that stale
figure is the `green_light` rule's *own* recall and predates the two rules that now do the work:

| rule | guards caught |
| --- | --- |
| R7 inside-only movement | 131 (34.8%) |
| R2 warmup flashlight | 111 (29.5%) |
| R1 green light (scored frames) | 28 (7.4%) |

**Recall is no longer the bottleneck — leakage is.** Full confusion:

| true label | result |
| --- | --- |
| guard (376) | 71.8% correct, **8.8% (33) leak into the animal/incident alert channel**, 6.1% → environment, 12.0% unclassified |
| environment (100) | **only 50% caught**; 18% → incident channel, 15% → guard |
| animal (13) | 46.2% correct, 23.1% unclassified |
| incident (10) | 70% correct, **30% (3) suppressed as guard_candidate** |
| resident (10) | no rule exists; 60% land in guard |

Two things a new agent should weigh before adding more guard signal:

1. **`environment` is the weak link now, not guard.** Half of it is missed and 18% reaches the
   high-priority incident channel — a bigger false-alert source than guard misrouting.
2. **3 of 10 incident clips are suppressed as `guard_candidate`.** This is the known, accepted
   cost of ordering guard rules ahead of the geometry rules, and it is currently safe *only*
   because every one of those incidents has a sibling clip that still alerts (event check 5/5).
   That margin is thin — re-check it whenever a guard rule is added or an incident is labelled.

### The environment leak, diagnosed and FIXED (2026-09-07)

**29 environment clips reached the alert channel** (18 `incident_candidate` + 11
`animal_candidate`) before this fix — the largest false-alert source, bigger than guard
misrouting.

**It was heavily concentrated by camera:** cam10 ×16, cam12 ×4, cam15 ×3, cam07 ×3, cam08/cam09/
cam13 ×1 each. cam10 alone was 55% of the leak, consistent with its already-documented 55.6%
`environment_candidate` rate — a per-camera problem as much as a rule problem.

**`median_fence_distance` is the discriminator, AUC 0.840** separating leaking environment from
real animal+incident — and the direction is physically meaningful: an intruder approaches the
fence (positives p50 **0.227**) while wind-shaken vegetation sits out in the field (leaking
environment p50 **0.406**). The animal/incident rule previously bounded this feature only from
*below* (`> 0.1`); it had no upper bound.

Simulated through the **real rule order** (only rows the earlier rules don't already catch):

| upper bound | env leak | guard leak | incident events alerting |
| --- | --- | --- | --- |
| none (was current) | 29/100 | 33 | 5/5 |
| 0.50 | 22/100 | 32 | 5/5 |
| 0.45 | 17/100 | 29 | 5/5 |
| **0.40** | **14/100** | 27 | **5/5** |

**Incidents were entirely safe at any of these bounds** — the 10 incident clips run
0.117…**0.353**, so 0.40 still leaves headroom, and all 5 incident *events* keep alerting.

**The entire cost is one animal event: `cam10/7632`** (`median_fence_distance` 0.504), which is a
solo event with no sibling clip to cover it, so it is lost at 0.50, 0.45 and 0.40 alike. Its own
label note reads *"animal enters very early (during IR flare/warmup) ... visually marginal/hard
to confirm"* — i.e. precisely the case the IR-flare-tracking item above would help. The other
distant animal, `cam15/17949` (0.586), was **not** at risk: it reads `outside_pixel_fraction=0.00`
so it never satisfied the geometry rule anyway, and its sibling `17950` (0.251) carries the event.

**Decision (2026-09-07): the user accepted the tradeoff.** Shipped as `MEDIAN_FENCE_DISTANCE_MAX
= 0.40` in `scripts/backtest.py`, one extra clause in the animal/incident geometry rule.
`scripts/check_incident_regression.py` re-run afterward: still 5/5 incident events alerting,
`cam10/7632` now reads `unclassified` — the accepted cost, not a silent regression. If the
cam10-specific concentration (16/29 of the original leak) is revisited later, its `fence_bottom`/
geometry or its own environment base rate is the next thread to pull, separate from this
corpus-wide threshold.

**CORRECTED 2026-09-08: this "cost" was never real in a live system.** The claim above that
`cam10/7632` is "a solo event with no sibling clip to cover it" is wrong — its own short/duplicate
preview sibling, `cam10/7631`, already reads `animal_candidate` independently (`median_fence_
distance` 0.365-0.370, under the bound). `classify_event(['animal_candidate', 'unclassified'])`
returns `animal_candidate`, and a live system processing each Telegram message as it arrives would
alert on `7631` before the fuller `7632` is even relevant. Verified `7632`'s own reading isn't a
frozen-track artifact either: recomputing `median_fence_distance` from genuine (non-recovered)
centroids only moves it from 0.504 to 0.498 — the animal genuinely was that far from the fence.
Also checked whether `persistence`/`path_length` could recover `7632` and its 5 similarly-excluded
siblings (2 more animal, 3 unlabelled-adjacent) without the threshold change itself: no — even the
tightest combination tested re-admits 6-15 guard/environment clips per animal clip recovered.
`MEDIAN_FENCE_DISTANCE_MAX` stays at 0.40, unchanged.

### Immediate next step (highest value, well-evidenced)
`post_flash_red_shift` is a **zero-leak guard identifier corpus-wide** but is currently only used
by the maintenance queue. It is not yet a rule in `scripts/backtest.py::classify()`. Adding it as
a `guard_candidate` rule would be easy — but **weigh it against the confusion matrix above first**:
guard recall is already 71.8%, and every new guard rule sits *ahead* of the geometry rules, so it
can only push more incidents toward `guard_candidate` (already 3/10). Extra guard recall is not
clearly what this system needs right now; the environment leak above is the bigger win.

### IR flare tracking: IMPLEMENTED (2026-09-07), opt-in via `compensate_warmup`

**What shipped.** `scripts.spike.detect_clip(..., compensate_warmup=True)` (default `False`,
wired on only in `scripts/render_debug.py`): each dropped/warmup frame is photometrically matched
(`src.features.photometric_match`, a per-frame least-squares gain/offset fit) onto the settled
`background`, then diffed and tracked with the SAME threshold/morphology/contour pipeline used for
every scored frame — real per-pixel tracking, not the old appearance-template-only guess. Seeded
from `considered[0]`'s own established box for continuity when there is one; when there isn't
(e.g. `cam06/21520`'s slow-dwelling crawler, where the scored track never establishes at all), it
runs unseeded instead of giving up, so a real diff still gets a chance inside the warmup window on
its own merits. Falls back to the pre-existing appearance-trace box wherever this real-diff pass
finds nothing, so coverage never regresses. `src.features.photometric_match_color` additionally
colour/brightness-corrects every dropped COLOUR frame against a settled colour background, purely
for display (`ClipDetection.dropped_frame_compensated`) — this is the "even out the flare" ask.
`ClipDetection.dropped_frame_box_is_photometric` tags which mechanism found each box, so
`render_debug.py` can label a real diff-tracked box differently from an appearance-only guess.

**Revised same day, per the user's own framing ("it settles, should we not read from that settled
frame and normalise to that?"):** the fit walks backward from the settled frame rather than fitting
each warmup frame independently against it. Frame `drop-1` (closest to settled) is matched
directly against `background` — a small, well-conditioned fit — and every earlier frame is then
matched against its own already-corrected neighbour, one small step at a time, back to frame 0.
A frame near the start of a steep ramp (often near-black) fitting directly against the far-away
settled background is a much larger, less reliable jump than a chain of small steps between
frames that already resemble each other. The diff target for motion detection is unchanged
(still the real `background`) — chaining only changes what each fit is computed against.

**Deliberately scoped safe by construction, not just tested safe.** None of `frames`,
`background`, `considered`, or any `FEATURE_COLUMNS` value is touched — the new code only
populates `dropped_frame_boxes`/`dropped_frame_box_is_photometric`/`dropped_frame_compensated`,
fields already documented as render/diagnostic-only, never read by `extract_clip_features`. This
sidesteps the "main risk" below entirely, rather than accepting it: there is no merge into the
scoring pipeline for a residual gradient to hide in.

**Kill-check + validation, all before wiring it into the renderer:**
- Full test suite green (531 tests, +6 new: `photometric_match`/`apply_photometric_match`/
  `photometric_match_color` unit tests, two `detect_clip(compensate_warmup=...)` tests).
- `cam08/7360`, `cam15/15454`, `cam05/18270` (the three previously-fragile reference clips) —
  scored-frame detection byte-identical on/off, as expected by construction.
- Broader sweep: every clip referenced anywhere under `data/reports/debug_render/` (89 real
  files), `detect_clip(..., compensate_warmup=True)` vs `False` — **zero exceptions, zero
  scored-frame differences** (`largest`/`recovered`/`filled_by_reverse`/`centroid` compared frame
  by frame). 84/89 clips got at least one real photometrically-tracked warmup box.
- Visually confirmed on `cam06/21377` (the guard-visible-during-warmup clip from
  `_warmup_motion_features`'s own docstring): raw frame 0 is near-black, raw frame 6 has fully
  ramped up — the classic flare. The compensated render shows a consistently exposed scene
  throughout, with a real tracked box (solid cyan, "real diff, brightness/colour corrected")
  following what looks like a subject moving down along the fence frame to frame, not a static
  artifact. Demo renders in `data/reports/debug_render/ir_flare_compensation/` (gitignored).
- Re-render both attempts: the seeded/independent-fit version was numerically validated first
  (89 clips, 0 mismatches) without touching the standing `debug_render/` set, since the scored-frame
  diff already proved nothing scored could have moved. After the reverse/chained revision above,
  the user asked for a full re-render, so all 112 files under `data/reports/debug_render/` were
  regenerated in place (correct per-clip camera/timestamp/zone resolution, including the 4
  oddly-named dated-camera-era files) — re-verified 0 errors, and the full suite plus
  `scripts/check_incident_regression.py` (5/5) stayed green throughout.

**Known limitation, honestly not solved:** this does not fully rescue `cam06/21520` (the crawling
guy who dwells through most of his own clip's median background) — the unseeded fallback finds
*something* there now (7/15 frames), but the boxes are implausibly large (up to ~half-frame),
consistent with this clip's own already-documented root cause (the subject biases its own
whole-clip median background, so nothing genuinely diffs against it even inside the warmup
window). That is a background-MODEL problem (rolling/windowed background, not a whole-clip
median), not a warmup-compensation problem, and remains open.

**Not attempted:** wiring any of this into `extract_clip_features`/`FEATURE_COLUMNS` — the
`compensate_warmup` output is currently display/diagnostic-only, exactly like the appearance-trace
mechanism it improves on. Feeding it into scoring is the "main risk" scenario below and would need
its own full LOO-AUC-before/after validation pass, not attempted this session.

### Also requested: show the IR compensation in the debug renders — DONE

`scripts/render_debug.py`'s warmup-frame panel now shows `dropped_frame_compensated` (colour/
brightness-corrected) instead of the raw flare frame, and labels a real-diff-tracked box
("TRACKED (real diff, brightness/colour corrected)", solid outline) distinctly from the
appearance-only fallback ("INFERRED - TRACKED (reverse trace)", dashed, unchanged from before).



### Traps this session hit — read before proposing a new signal

Three signals looked clean on hand-picked examples and died on base rate:
`solidity==1.0` (60-92% in *every* class), the warmup *fraction*, and `warm_peak` (animal p50
0.429 — it leaks 38% of animals). **Always check the protected classes' own distribution before
believing a threshold derived from confirmed examples.**

And the opposite failure, which cost more: two ideas were rejected because *I measured them
wrong*, not because they were bad. `post_flash_red_shift` was dismissed on a clip-wide **average**
R/G (which gave a backwards result) until the user clarified the signal was temporal — flash,
*then* red. Measured as a transition it has a 0.58-vs-0.014 class separation. **When a user
describes a signal in temporal terms, measure the transition, not an aggregate.**

## Handoff for a new agent (2026-09-10, session close #16)

**Read this section first — it supersedes #15 for current state.** Same branch
(`feat/phase2-refactor`), 688 tests passing, working tree clean at `cf8ebbb`. This session (spans
2026-09-09 into 2026-09-10) wrote `docs/detection_improvement_review.md` from a full read of the
codebase and the labelled corpus, then implemented most of its prioritised recommendations,
in order, each measured against the full 678-clip labelled corpus before being trusted and
committed separately. **That review document's own "Implementation status" section at the top is
the authoritative, itemised record (11 items) — this entry is the short version and the things a
fresh agent needs to know that aren't obvious from reading the code.**

### What landed, in order

1. **A real bug, fixed**: `_multi_object_outside_features` was reading `TrackedObject.bbox`
   (corner form `(x0,y0,x1,y1)`) as `(x,y,w,h)`. Confirmed byte-identical `classify()` output
   before/after on the full corpus (not yet wired into any rule at the time).
2. **`check_incident_regression.py` now asserts on animal events too**, not just incidents —
   surfaced 3 previously-invisible animal-event misses. Each was root-caused, not just logged (see
   the script's own docstring and `KNOWN_ANIMAL_EXCEPTIONS`): `cam10-2024-12-01T16:12` (a
   fence-distance lower-bound miss), `cam15-2025-07-15T00:29` (a porcupine recovered only via its
   sibling clip's own fragile reading), `cam10-2025-10-28T03:51` (a bird on the fence rail —
   ground-plane assumption violated). **These 3 are still open, unfixed, exception-listed by
   design** — not a regression this session caused, real information that was always there.
3. **`scripts/backtest.py` gained real metrics**: confusion matrix, alert-channel precision/
   recall/F1, per-camera leak, written to a `.summary.json` alongside every CSV.
4. **`neighbour_candidate` rule shipped** — a person outside the fence in daylight, at plausible
   human height. Measured on its actual position in the rule chain: 4/5 labelled neighbour clips
   (3/3 events), zero cost to incident/animal/resident/guard, 3/159 environment clips added to
   this low-priority channel. **Needed schema v6→v7** (`predicted_class` CHECK constraint widened)
   — see "Schema is now v7" below if you have another copy of the DB.
5. **Per-object flashlight scoring, then a rule on it, then the debug render fix** — the
   `cam07/11174` story, worth knowing end to end since it's the throughline of most of this
   session: the single-track pipeline was following a bush 40% of the frame away from a real,
   plainly-visible flashlight, so `green_light_ratio` read 0.0. `_multi_object_flashlight_features`
   (reporting-only at first) scores EVERY persistently-tracked object independently, so it sees
   the beam regardless of which object the single-track pipeline follows. Measured clean (zero
   leakage into any protected class), then wired as a third, independent `guard_candidate` rule
   (`multi_object_max_flashlight_ratio`): alert-channel FP 35→23, precision 0.386→0.489, recall
   unchanged, every protected class byte-identical. Then found (via the user's own question — "does
   any flashlight object get traced as flashlight in the debug clip?") that `scripts/render_debug.py`
   never visually marked this: only the single main tracked box was ever checked for
   flashlight-ness. Fixed — every persistent object is now checked and marked.
6. **The multi-object tracker itself rewritten** (`track_multiple_objects` — object-linking stage
   1), per explicit user instruction ("do 1 and 2 ... i dont really see a need of seperating the
   flashlight head with the beam" — stage 3 explicitly declined). Global-optimal assignment
   (`scipy.optimize.linear_sum_assignment`, a real new dependency) replaces per-track greedy
   picking in dict-iteration order — a demonstrated real bug, not a theoretical one (two tracks
   could both claim the one candidate nearest their shared midpoint while a second, valid one sat
   unclaimed and spawned a spurious third id). Added an optional colour-histogram appearance
   tie-breaker, a cap on indefinite merge dead-reckoning, and switched the real pipeline to track
   every raw candidate rather than only min-area blobs (the fix for "cannot see small subjects at
   all" — the animal population). **This surfaced a real regression before it shipped**: the
   tentative-track `confirm_frames` gate needed to make that candidate-widening safe, at the value
   first tried (3), silently suppressed short-lived flashlight objects — regressing 16 guard clips
   including `cam07/11174` itself (ratio collapsed back to 0.0). Caught by re-measuring, not
   assumed; **`confirm_frames` now defaults to 1 (effectively off)** and is deliberately NOT the
   same value that would be needed to fully suppress tracker noise — see its docstring before ever
   changing it. At the shipped default, re-measured clean: alert-channel FP 23→22, precision
   0.489→0.500, recall unchanged, guard correctly classified 350→354, every protected class
   byte-identical, `cam07/11174` exactly restored.
7. **Per-track person/animal/artifact typing** (object-linking stage 2), reporting-only —
   `_multi_object_type_features`. Person/animal reuse the existing ground-plane calibration per
   object instead of only the single tracked subject; artifact reuses `blob_white_fraction`.
   **Measured and found NOT separable as first coded** — once the tracker was widened to track
   every candidate, `multi_object_animal_track_count > 0` fired on 138/158 environment clips
   (median count 53.5) vs 26/32 real animal clips (median 2.5): wind/insects mint dozens of
   one-frame "animal-height" tracks. Added a minimum-evidence filter
   (`MIN_TRACK_FRAMES_FOR_TYPE=3`) — helps, does not fix it. **Tried raising that bar further and
   it got WORSE, not better** (measured on cam10 specifically, its best-populated camera for both
   labels: at 12 frames, real animal clips drop to 0/6 with any qualifying track at all while
   environment stays at 39/83 — a real animal's track apparently does not outlast wind-shaken
   vegetation's). **Do not retry raising this threshold** — see the review doc's "do not retry"
   table and the feature's own docstring. `multi_object_artifact_track_count` looks more promising
   in the same pass but is flagged, not claimed, since it's likely correlated with the pre-existing
   `blob_white_fraction`. None of stage 2's features are wired into `classify()`.
8. **Also shipped along the way, lower-stakes**: `scripts/rank_candidates.py`'s LOO-CV
   sibling-leakage fixed (was fitting on both clips of one physical event); `has_reference_
   background` restored as a ranker input; per-camera feature standardisation added as an
   **opt-in, off-by-default** flag after a real measurement came back negative (LOO AUC 0.884→0.755
   on the only affordable test this session could run — see the flag's own docstring for exactly
   what was and wasn't measured). `optical_flow_direction_coherence` built and measured as a new
   reporting-only feature — the specific hypothesis (single-pair internal coherence) reads
   BACKWARDS on the full corpus (environment highest, not lowest) and is kept as infrastructure,
   not a validated discriminator — do not re-attempt that exact framing.

### Schema is now v7

`scripts/migrate_schema_v7.py` (already run against the live `data/perimeter_watch.db`, backed up
first). `predicted_class`'s CHECK constraint gained `neighbour_candidate`. **If you have another
copy of this database, run that migration against it too** — same hard-error-on-mismatch behaviour
noted in session #15's entry for v6.

### New dependency: `scipy`

Added for `scipy.optimize.linear_sum_assignment` (the tracker rewrite, item 6 above). Already in
`pyproject.toml`/`uv.lock`; nothing extra to do beyond the usual `uv run`.

### Verification discipline this session actually followed

Every item above was measured against the full 678-clip labelled corpus before being trusted —
`scripts/backtest.py --labelled-only --no-record`, comparing `.summary.json` confusion matrices
and per-clip category diffs against a captured baseline, not just "does it still pass the test
suite." Two regressions were caught this way BEFORE shipping (the `confirm_frames=3` flashlight
suppression in item 6, and the animal/person count near-universality in item 7) rather than being
discovered later or not at all. `uv run ruff check . --fix && uv run pytest -q` and
`scripts/check_incident_regression.py` stayed green (or at the same 3 documented exceptions)
throughout, checked after every single commit, not just at session end.

### Open decisions still pending — none of these were implemented without sign-off, on purpose

`docs/detection_improvement_review.md`'s "Open questions for the user" section (near the bottom)
has the full detail. Short version, in priority order:

1. **The fence-distance band** (`MEDIAN_FENCE_DISTANCE_MIN`/`_MAX` in `config/thresholds.yaml`).
   Removing it recovers 11/22 missed animal clips at a cost of 17 guard + 16 environment false
   alerts. Trade-off depends on the user's actual tolerance, not measurable from the data alone.
2. **Ship readiness criterion #3** (`docs/plan.md`) — a concrete guard/environment leak-per-night
   budget — is still unset and now blocks more decisions than it did in session #15.
3. **The what/where/when classifier restructure** (object-linking stage 4 — a clip carrying a SET
   of findings, not one winning category). The right long-term model per the user's own framing,
   but changes what an alert *is* and downstream notification isn't designed for it yet.
4. **Whether a supervised person/animal detector is in scope**, or the classical-CV commitment in
   `docs/plan.md` stands. Determines whether the metric-height approach (§2.2/§4.5) is the ceiling
   or a stepping stone.
5. **Using the reference background as the primary detection model**, not just a veto (§4.4) —
   repeatedly deferred, twice-evidenced as worth doing, still not attempted.

Stage 3 of object linking (beam vs torch, distinguishing a real flashlight head from
flashlight-lit vegetation) was explicitly declined by the user this session, not merely deferred —
do not build it without being asked again.

## Handoff for a new agent (2026-09-09, session close #15)

**Read this section first — it supersedes #14 for current state.** Same branch
(`feat/phase2-refactor`), 632 tests passing, working tree clean. This session executed
`docs/plan.md`'s Phase 2 refactor brief's two lowest-risk items — `classify.py` and
`backtester.py` — following a written step-by-step plan at
`docs/phase2_refactor_execution_plan.md`, which has the full per-step record (what moved, every
verification command, every result). This section is the short version of what changed and what a
fresh agent needs to know before touching this code.

### What landed

`src/classify.py` now exists: `classify_detailed()` holds the same rule chain
`scripts/backtest.py::classify()` always had, returning a typed
`ClassificationResult(category, reason, contributing)` — 15 reason codes, one per rule, so a
caller can finally say *which* rule produced a category, not just what the category was.
`classify()` is now a one-line wrapper (`classify_detailed(...).category`) kept for every existing
caller. `is_blinding_foreground()` moved alongside it. `scripts/backtest.py` keeps only the
screening harness around them (`run_backtest`, `iter_clips_with_files`, `write_csv`, `main`).

**Beyond the brief's original scope, at the user's explicit direction**: the 16 thresholds
`classify()` compares against — previously six named module constants plus nine inline magic
numbers — now live in `config/thresholds.yaml`'s `classification:` section, loaded via a new
`src.config.ClassificationThresholds` and `load_thresholds_config(...).classification_thresholds()`.
`classify()`/`is_blinding_foreground()` both take an optional `thresholds:
ClassificationThresholds | None` parameter, defaulting to a memoised parse of the real file. This
finally makes true what that file's header comment always claimed ("classify.py never hardcodes a
number") — **it was false before this session**: nothing loaded that section, and its old values
were stale and wrong (`outside_pixel_fraction.alert_min: 0.5` against the real, measured `0.6`; a
whole `aspect_ratio` section whose rule was retired 2026-09-04). **If you go looking at
`config/thresholds.yaml`'s git history and see those old values, do not treat them as a prior
"real" operating point to reconcile toward — the new 16 values were taken FROM `src.classify`'s
code, which was always the actual source of truth.** `motion:` is untouched and still unconsumed —
see "Next: motion.py caching" below.

One coupling worth knowing about if you ever touch `green_light_ratio_min`:
`src/features.py`'s `FLASHLIGHT_CANDIDATE_MIN_RATIO` is deliberately pinned to the same value (one
"is this a real flashlight" bar shared between `classify()` and `scripts.spike`'s track-scoring).
That used to be visible as two adjacent module constants; now that `classify()`'s side lives in
YAML the coupling is invisible in the code, so `tests/test_config.py::
test_flashlight_candidate_ratio_stays_coupled_to_classify_threshold` is what catches a future
desync. If it ever fails, the fix is almost certainly to also change `FLASHLIGHT_CANDIDATE_MIN_RATIO`
in `src/features.py`, not to silence the test.

`src/backtester.py::record_run()` wires the `backtest_runs`/`backtest_results` DB functions
(existing, unit-tested, previously called by nothing) into `scripts/backtest.py::main()`, which now
records every real run by default (`--no-record` to skip). Needed a schema bump — see next.

**Schema v6** (`scripts/migrate_schema_v6.py`, applied to `data/perimeter_watch.db`, backed up
first to `data/perimeter_watch.db.bak-2026-09-09-pre-v6`): `backtest_results.predicted_class`'s
CHECK constraint used to accept only the four-class routing vocabulary
(`guard_side`/`outside_alert`/`outside_priority`/`ambiguous`) that plan step 25 always specified but
was never built. `classify()` emits eight different categories instead. The constraint was
**widened to accept both**, not mapped — collapsing eight measured categories onto four routing
classes would encode an alerting policy nobody has validated, and picking one is gated on Ship
readiness criterion #3, still unset. **If you have another copy of this database (a laptop, a
backup, a second checkout), it needs `uv run python scripts/migrate_schema_v6.py` run against it
too** — `db.connect()` hard-errors on any schema_version mismatch, so every tool refuses to open an
unmigrated copy.

### Verification discipline this session actually followed

Every step was checked byte-identical against the pre-refactor code on the full 678-clip labelled
corpus (`uv run python scripts/backtest.py --labelled-only --out ...`, diffed against a Step-0
baseline captured before any edit) — extraction, the config wiring, and the reason-code addition
all came back with **zero mismatches** on every classification-affecting column. The one step
whose CSV output legitimately differs (adding the `reason` column) was checked by comparing every
*other* column instead. `scripts/check_incident_regression.py` stayed at 5/5 incident events
throughout. A full, unlabelled-corpus (16,886 clip) pass was also run after the last commit as an
additional smoke test — see the execution plan doc's Step 10 for that result if it matters to you;
it has no pre-refactor baseline to diff against (only a labelled-only one was captured at Step 0),
so it can only confirm "no crash, sane distribution," not byte-identical output, on the ~16,200
clips outside the labelled set.

### Next: `motion.py` caching (plan step 22) — the risky one, do it its own session

This is the one item left in the Phase 2 refactor brief that can make results silently *wrong*
rather than just slow, so give it a session of its own rather than folding it into something else.
`scripts/spike.py::detect_clip` takes 22 tuning keyword arguments and carries several sessions of
hard-won correctness fixes (warmup handling, anchor sweep, reference-bg veto, the
flashlight-candidate work) — extracting it to a cached `src/motion.py` needs the cache key to cover
every parameter that affects its output, or a config change will silently serve stale cached
results.

Two things worth knowing before starting, found this session while doing the equivalent work for
`classify.py`: the cache *machinery* is further along than the brief implies —
`src.config.ThresholdsConfig.motion_fingerprint()` and the `blob_tracks` table (composite key on
`extractor_version` + `motion_fingerprint`) already exist and are tested. What's missing is (1) no
`EXTRACTOR_VERSION` constant is defined anywhere, and (2) `motion_fingerprint()` hashes
`thresholds.yaml`'s `motion:` section, which is **not** where `detect_clip`'s real parameters
live — they're defaults in its own function signature, untouched by this session on purpose
(nothing in `scripts/spike.py` changed). Wiring `motion:` up is the same class of work this session
just did for `classification:` (real values from the code, a typed loader, `detect_clip` reads from
it instead of hardcoded defaults) — but against a function with 22 parameters instead of 16 values,
and a failure mode that's silent (stale cache) instead of loud (wrong category, caught immediately
by the incident regression check). Same discipline applies: byte-identical corpus verification at
every step, and don't retune anything while moving it.

## Handoff for a new agent (2026-09-09, session close #14)

**Read this section first.** `feat/phase2-corpus-and-cv` is merged to `main`; a new
`feat/phase2-refactor` branch is open with one explicit focus: **organise Phase 2's code, not
change its behaviour.** If you're picking this up fresh, go straight to `docs/plan.md`'s "Phase 2
refactor brief" section — it has the per-step reality check and a suggested order
(`classify.py` → `backtester.py` → `motion.py` caching, with the browser zone editor and a
backfill/parsing audit as separate lower-priority tracks). This section is the short version.

### Where detection/classification itself stands (closing out session #13)

`prefer_flashlight_candidate` (both the fresh-pick and active-track pieces) is validated at full
16,886-clip corpus scale with no disqualifying finding — see session #13 below for the full
numbers and for a retraction worth reading once: a "stationary light bug" I diagnosed from grainy
IR stills turned out to be a misread once the user checked the actual footage, and every flagged
clip was a genuine correction. The mechanism is still off by default, not because of any known
defect but because the one thing left is a business call, not an engineering one: **Ship
readiness criterion #3 in `docs/plan.md`** (how much guard/environment leak per night is
tolerable) is still unset. Nothing is blocking that decision from being made whenever it's wanted.

### The ask for the next agent: organise, don't rebuild

The empirical work across sessions #1-#13 solved real detection/classification problems but did it
inside `scripts/spike.py` and `scripts/backtest.py` rather than the module boundaries
`docs/plan.md` originally specified (`src/motion.py`, `src/classify.py`, `src/backtester.py`). That
was the right call at the time — gate 2 was undecided and refactoring a moving target would have
been wasted work — but gate 2 is retired now and the target has stopped moving as much. Time to
clean it up.

**This is explicitly a refactor, not new detection work.** The full test suite and
`scripts/check_incident_regression.py` must stay green throughout, and any extracted module's
output should be checked byte-identical against the pre-refactor version on the full labelled
corpus before being trusted — same discipline this repo has applied to every real detector change,
now applied to moving code around instead of changing what it computes. See `docs/plan.md`'s Phase
2 refactor brief for the concrete per-step breakdown; short version: `classify.py` (extract
`scripts/backtest.py::classify()`, add reason codes — nothing to build, just organise) and
`backtester.py` (wire in the `backtest_runs`/`backtest_results` DB functions that already exist and
already have tests, currently called by nothing) are both low-risk, high-value, and independent of
each other. `motion.py` (caching `detect_clip`'s output) is real but riskier — its cache key has to
cover every one of `detect_clip`'s 20+ tuning parameters or a config change will silently serve
stale results — and should come after the other two are done and stable, not first.

## Handoff for a new agent (2026-09-08, session close #13)

**Read this section first — it supersedes #12 for current state.** Branch
`feat/phase2-corpus-and-cv`, 577 tests passing, nothing uncommitted. This session picked a batch of
cam07 debug clips apart with the user (a mix of the "neighbour" investigation and a broader
`incident_candidate` sample) and turned three separate observations into one real fix plus one
diagnosed-but-not-fixed architecture question.

### The active-track flashlight override: cam07/11174 is now actually fixed

Session #11's `prefer_flashlight_candidate` only ever applied to a track's fresh/unconstrained
pick, which is exactly why it couldn't fix the clip that motivated it — traced why directly this
session: at cam07/11174's frame 0, the bush is the ONLY candidate large enough to matter (fresh
pick correctly has nothing to prefer yet), and the real flashlight only becomes its own separate
candidate at frame 1, by which point the bush is already the active track. A 113px candidate
90+px away from a 3025px active track fails both the IoU and distance continuation checks, so it
was correctly rejected as "not a continuation" and then wrongly discarded as a miss instead of
being treated as what it actually is: a new, independent detection.

**Fixed:** `track_contour`'s "no plausible continuation" branch (active-track case) now checks the
same `FLASHLIGHT_CANDIDATE_MIN_RATIO` bar before giving up — a candidate that clears it there
can't be the old track reappearing implausibly (it already failed the checks that would confirm
that), so it starts a new identity instead of being lost. Confirmed directly on cam07/11174: now
tracks the real subject for all 3 frames (an existing, unrelated mechanism — the `seed_index`
outlier-replacement logic that already existed for `considered[0]` — even retroactively fixes
frame 0 for free, once frames 1-2 establish what the "typical" tracked size actually is).
`green_light_ratio` goes 0.0 → 0.264, category flips `incident_candidate` → `guard_candidate`.

**Measured on the full 678-clip labelled corpus** (same methodology as session #11,
`data/reports/scratch/flashlight_active_override_2026-09-08/`):

| | count |
| --- | --- |
| guard clips corrected | **24** (up from 14 with the fresh-pick-only version) |
| guard clips regressed | 4 |
| environment clips affected | 2 |
| animal clips changed | 1 (lateral move, never in the alert channel either side) |
| incident clips changed | **0** |
| hard constraint (47 animal+incident clips) | 1 changed, **zero alert-channel impact** — identical to the smaller fix |

Two of the 24 corrections are already-named problem clips from earlier sessions: cam08/10852 and
cam01a/18603 (the truncated-preview pair examples from session #6), and cam01/16167 (flagged
across sessions #3/#8/#9 for flashlight-hue and light-drift issues).

**All 4 regressions share one root cause, checked directly on the actual frames**: the override
can jump onto flashlight-*illuminated vegetation* instead of the person, when the beam lights up a
bigger green-scoring patch than the subject itself (cam05/4738-4739, cam02/9053, cam07/21533).
This is a real, understood trade-off — the mechanism can't yet distinguish "green because it's the
torch" from "green because the torch is lighting up a bush," the same structural limitation
`green_light_ratio` has always had. Not fixed this session.

**Still off by default.** Zero cost to the hard constraint twice now (both the small and the big
version of this fix), but the same discipline as every other `detect_clip`-level change in this
repo's history applies: only the 678 *labelled* clips were swept, not the full ~17,000-clip corpus,
and the vegetation-hijack failure mode is diagnosed, not closed. Worth strongly considering for
promotion to default once (a) a full-corpus sweep confirms the vegetation-hijack rate is small, and
(b) either that failure mode gets a real fix (e.g. requiring the lit candidate to also be
person-shaped/sized, not just green-scoring) or is accepted as a known, bounded cost.

### The reference-background scenery veto: diagnosed why it misses cam07's short-clip bush hallucinations, not a bug in the veto

Investigating a 147-clip pool of cam07 `incident_candidate` reads (outside, human-plausible
calibrated height, no flashlight signal, all unlabelled) turned up three distinct sub-populations,
confirmed by the user watching a 9-clip sample: pure appearance-match hallucination on bush during
near-total darkness, real wind-blown vegetation, and (the one that led to the fix above) a real
guard whose flashlight arrives after the tracker's already committed elsewhere.

For the bush-hallucination population specifically: checked directly why the existing
reference-background veto (session #4, built for exactly this shape of problem) doesn't catch it.
**It isn't a bug — the veto only ever evaluates `recovered=True` runs (a track that's stopped
producing real diffs and is re-matching a stale template), and these clips' bogus detections are
`recovered=False`: genuine per-frame background-diff hits.** Confirmed on cam07/9593 and 5 others
— every hallucinated frame shows `recovered=False`. The real mechanism: these clips have only 3-5
"considered" frames after the warmup drop (4-8 total raw frames), far too few to build a stable
per-clip median background, so the bush's own natural texture wobble reads as spurious motion
against its own noisy, tiny-sample median.

This is the same architecture gap flagged and deliberately not attempted in session #4's continued
QA pass ("the only still-credible direction is a real cross-clip/per-camera reference-background
architecture change... deliberately not attempted without explicit sign-off given its corpus-wide
blast radius") — now with a concrete, reproducible trigger (very short clips specifically).
Extending the existing veto to also cover `recovered=False` frames was already tried in a related
form and rejected (session #2's "signal (a)": NCC scores a brightness-boosted real subject as
"matching" the reference just as well as real scenery) — so this needs a different, deliberate
design (e.g. using the per-camera reference AS the background model when a clip has too few frames
for its own median to be trustworthy, not just as a downstream veto), not a quick extension. Not
attempted this session — flagged as a well-evidenced next step, same "needs explicit sign-off"
caution as before.

### Telegram delivery-gap measurement, for the wait-for-sibling design question

Confirmed `clips.timestamp` in the DB is the real Telegram message delivery time (`msg.date` from
the raw backfill), not the embedded camera-clock caption timestamp — so the open "should a live
system wait for the fuller sibling clip" design question (sessions #6-#7) can be measured directly
without new instrumentation. Across all 8,274 real sibling pairs: min gap 5s, **median 190s**, p95
285s, max 514s. A short wait (the 30s originally proposed) only catches 6.1% of pairs; 60s catches
10.2%; you need ~300s to catch 99.9%. Recommended direction if this gets picked up: fire on the
short clip immediately (never suppress a real detection) and send a correction if a fuller sibling
later disagrees, rather than holding every alert for minutes to catch the rare case. Not built —
this is still Phase 4/5 (live service) scope per `docs/plan.md`, flagged here only because the
measurement was cheap to do now.

### Other user corrections/findings worth recording

- **The `resident`/daylight question is more subtle than "daylight = safe."** Empirically checked:
  the confirmed daylight-neighbour clip (cam01/10560-10561) already reads `environment_candidate`/
  `unclassified` today, not `incident_candidate`, because `color_fraction > 0.15` routes it away
  from the incident branch before shape rules ever apply. That gate is the same kind of raw image
  statistic that's failed as a daylight proxy four separate times elsewhere in this repo — it
  hasn't bitten here yet, but swapping it for the exogenous `is_daylight`/`daylight_hint` signal
  (already built, already used elsewhere) is the more robust long-term fix, not urgent.
- **13842's "something flies up into the camera"** — located precisely (raw frames 27-29, ~5.4-5.8s
  in, a bird/bat-near-illuminator motion-blur streak), confirmed geometrically outside the fence
  for its whole visible trajectory, and currently invisible to the pipeline for two compounding
  reasons: the brightness swing it causes trips `flare_frames()`'s gain-step heuristic, and even
  where a contour is found, the single-track design has no path to notice a second, unrelated,
  much bigger subject once a track is already active elsewhere. `track_multiple_objects`/
  `ClipDetection.multi_tracks` already exists for exactly this (every distinct blob gets its own
  id) but is never fed into `extract_clip_features` — purely a debug-overlay diagnostic today.
  Turning it into a real secondary-detection channel is the honest fix, not attempted this session.
- **21534's "different" flashlight colour, measured**: real hue is 37-56 (still the established
  green family, 33-85), just on the more yellow-green end (other cameras measured 34-40 to date).
  What's actually different: ~47% of the beam's brightest pixels blow out to near-white from sensor
  saturation while the rest stay clearly green-tinted — a brighter/closer beam than usual, not a
  different light source.
- **The neighbour-worker theory narrowed, not confirmed broadly.** Per the user directly: cam07's
  clean neighbour sightings are "in better [visibility], clearly the dude standing by the fence,"
  while the rest of the originally-flagged "outside" cluster turned out to be "just noisy startups
  or other artifacts" — i.e. the three separate problems above, not more neighbour instances. No
  relabelling done this session; the population of genuine cam07 neighbour sightings is still
  effectively just the one pair (22393/22394) confirmed so far.

### Suggested next steps

1. **Full-corpus (not just labelled) sweep of the active-track flashlight override**, per its own
   "still off by default" section above — the highest-leverage item, since the labelled measurement
   is already strongly positive.
2. **Size the three cam07 `incident_candidate` sub-populations properly** (bush-hallucination /
   wind vegetation / flashlight-arrives-late) beyond the 9-clip sample used to characterise them —
   only cam07 was checked; the same near-total-darkness-tiny-sample-background mechanism likely
   affects short clips on other cameras too, just less often given cam07's illuminated bush gives
   it more texture to hallucinate onto.
3. **The reference-background-as-primary-background-model architecture question**, now with a
   second concrete trigger (very short clips) alongside the original frozen-track one — still
   deliberately not attempted without explicit sign-off given the blast radius.
4. `track_multiple_objects` → real secondary-detection channel, for cases like 13842 where a
   second, real subject appears while an unrelated track is already active. Bigger scope than item
   1, same family of problem.
5. Everything in #12's list that's still open (gate 2 retirement follow-through, the resident
   population, `color_fraction`→`is_daylight` swap for the incident-branch gate).

### Full-corpus sweep (item 1 above), done same session — found a real, well-scoped bug before it shipped

Ran the active-track flashlight override against all 16,886 downloaded clips (not just the 678
labelled ones), flag off vs on, parallelised 12-way
(`data/reports/scratch/flashlight_active_override_full_2026-09-08/measure_full.py`, 75.2 minutes,
zero exceptions). **1,258 of 16,886 clips changed (7.5%).**

| transition | count |
| --- | --- |
| `unclassified` → `guard_candidate` | 436 |
| `incident_candidate` → `guard_candidate` | 242 |
| `guard_candidate` → `unclassified` | 236 |
| `environment_candidate` → `guard_candidate` | 163 |
| `animal_candidate` → `guard_candidate` | 32 |
| `incident_candidate` → `unclassified` | 30 |
| `guard_candidate` → `incident_candidate` | 27 |
| everything else | 92 |

By camera: cam07 518 (80% of the `incident_candidate`→`guard_candidate` moves, 194 of 242 — this
is the already-validated population from earlier this session), then cam05 195, cam06 146, cam04
128, cam03 83, and a long tail. The `animal_candidate`→`guard_candidate` transitions (the ones that
most need scrutiny, since they touch a protected class with no ground truth to check them against
at full-corpus scale) are heavily concentrated too: cam01b 15, cam04 14, cam05 3.

**Spot-checked the non-cam07 clusters directly (bounding boxes, then actual pixels) and found a
real, previously-unknown bug, not just "more of the same trade-off."** cam04's 14
`animal_candidate`→`guard_candidate` clips all show the identical shape: the OFF track sits on a
small, often near-identical box across genuinely different clips (a tell that it's tracking
nothing real — confirmed visually on cam04/6079: OFF's box is over blank, textureless ground; the
actual content in frame is a **small stationary green-tinted light fixture at the top of a fence
post**, clearly visible once you look at the full frame, not a crop). ON's override correctly finds
real motion adjacent to that light — except the light itself is what makes the candidate
"flashlight-scoring," not a person. Checked cam01b/16936 the same way: same shape exactly, a tiny
(11-13px!) genuine fixed light/reflection near a wire, not an animal or a guard.

**Root cause, confirmed by reading the code, not just inferring it: `prefer_flashlight_candidate`'s
per-candidate `green_light_ratio` scoring (in `detect_clip`, both the fresh-pick and active-track
paths) is called with no `exclude_mask` at all.** Every OTHER consumer of `green_light_ratio` in
this codebase (the whole-track scoring in `extract_clip_features`, the render overlay) is passed an
`exclude_mask` built from `zone.ignore` (hand-traced) and/or `detect_stationary_light_mask`
(auto-detected) — this is the exact, multi-session-old infrastructure built specifically so a known
fixed light can never read as "the guard's flashlight." `prefer_flashlight_candidate`'s
candidate-level scoring is the one place in the codebase that bypasses it, because it runs inside
`detect_clip` itself, and the auto-detected stationary-light mask isn't computed until
`extract_clip_features`, downstream of `detect_clip`. Confirmed neither cam04 nor cam01b (nor
cam07, which is NOT affected by this — its real flashlight sightings dominate and it apparently has
no confounding fixed light) has an `ignore` polygon configured that would have masked this by
accident.

**This is why cam07's numbers stand (nothing there was fooled by this bug) but the cluster on other
cameras cannot be trusted yet.** Not fixed this session — the fix is well-scoped (thread the same
exclude mask already computed elsewhere into the candidate-scoring calls, or run a lightweight
per-clip stationary-light check before scoring) but deserves its own implementation and validation
pass, not a rushed patch at the end of an already-long session. **This is exactly what "sweep the
full corpus before shipping" is for — the 678-clip labelled sample never touched a camera with this
confound, so it looked clean and wasn't.**

**Updated recommendation: still off by default, and now with a concrete blocker, not just an
abundance of caution.** Before this can be defaulted on:
1. Thread an exclude mask (ignore polygons + auto-detected stationary lights) into
   `prefer_flashlight_candidate`'s candidate scoring.
2. Re-run this same full-corpus sweep after that fix and confirm the cam04/cam01b-style clusters
   disappear while cam07's validated corrections survive unchanged.
3. Only then reconsider promotion to default.

Full per-clip results: `data/reports/scratch/flashlight_active_override_full_2026-09-08/results_full.csv`.
Spot-check renders: `data/reports/debug_render/full_sweep_spotcheck_2026-09-08/`.

**CORRECTED, same session, immediately after writing the above: there was no stationary-light bug.
The "fixed light" diagnosis was a misread of grainy night-IR frames, corrected by the user who
actually knows the footage.** cam04/6079 and cam01b/16936 are real guards shining a flashlight —
confirmed directly by the user, not inferred. Re-checked three more from the same clusters after
the correction: cam01b/17061 (a flashlight beam lighting up grass, box slightly off-center onto
adjacent plant texture — imprecise but real), cam04/6665 (same shape as 6079), and cam01b/17764
(the guard leaves frame early and the box locks onto "a bright bush" left in view afterward — per
the user directly — but the clip's own classification is still correctly `guard_candidate`,
unaffected, and its fuller sibling 17765 already read `guard_candidate` correctly on both flag
values regardless). **Every single spot-checked clip in the "concerning" clusters turned out to be
a genuine correction, not a false positive.** The exclude-mask code written to fix the
non-existent bug was reverted rather than kept as unvalidated insurance — matches this repo's own
standing discipline of not shipping a fix for a problem that isn't measured.

**Practical lesson, worth repeating for whoever reads this next:** a bounding box wandering onto
something that looks like foliage/an artifact in a single dark, low-resolution IR still is not
enough evidence on its own — this repo's whole history is full of "obviously wrong" reads that
turned out to be real guards, real animals, or real fixed lights depending on ground truth nobody
but the user has. When a short/startup clip's read looks strange, check the fuller sibling clip
before concluding anything (17764/17765 above is a clean example of exactly why). **Net effect: the
full-corpus sweep result stands as originally measured (1,258/16,886 changed, cam07 518, the rest
spread thin) with no confirmed blocker.** The "still off by default, needs a full-corpus sweep
first" caution from the smaller (labelled-only) measurement no longer applies — that sweep is now
done, at full-corpus scale, with no disqualifying finding. Promotion to default is a much more
live option now than the previous version of this section suggested; the remaining open question is
purely the design one already on record (does the guard/environment leak trade-off match the
"Ship readiness" bar in `docs/plan.md`), not a correctness blocker.

## Handoff for a new agent (2026-09-08, session close #12)

**Read this section first — it supersedes #11 for current state.** `feat/phase1-finalisation` is
merged to `main` and a new `feat/phase2-corpus-and-cv` branch is open for Phase 2 work. This
session closed out the branch, retired gate 2 as a blocking decision, checked the `resident`
population against a user hypothesis, and investigated cam07's geometry.

### Gate 2 retired as a blocking pass/fail gate

Per the user directly: "gate 2 keeps getting in the way... the classifier needs to be good enough
for us to ship." `docs/plan.md` now has a **"Ship readiness"** section replacing it — five
concrete, re-measurable criteria (incident regression passes, no silent animal loss, guard/
environment leak small enough to review, no known corpus-wide detector bug, tests green) instead
of a one-time decision on a stale snapshot. **One number in that list is still an open call for
you, not invented by me:** how much guard/environment leak into the alert channel is actually
tolerable per night. Set it in `docs/plan.md`'s "Ship readiness" item 3 once you have a number in
mind — everything else in that list is already measurable today.

### `resident` population: thinner and different from the hypothesis

You asked what the population looks like, on the hypothesis "resident can be inside in daylight,
might bleed into guard, but guards don't start too early." Checked against the real labelled data:

**Only 4 physical events (8 clips) exist in the whole corpus** — 3 on cam01b, 1 on cam09. That's
the whole population; any rule built on it is a rule built on 4 data points.

**None of them are clearly daylight.** Converting to local time (UTC+2) and checking
`src.features.is_daylight`:

| event | local time | `is_daylight` |
| --- | --- | --- |
| cam01b, 2025-10-29 | 22:31 | False (night) |
| cam01b, 2025-11-05 | 05:52 | True (dawn twilight, right at the boundary) |
| cam09, 2025-11-23 | 18:31 | False (dusk) |
| cam01b, 2026-02-18 | 18:34 → 18:39 | True → False (straddles the sunset boundary within the same 5-minute pair) |

So the actual population is dusk/dawn/night, not daylight — the daylight framing doesn't match the
labelled examples, at least not yet (could still be true of unlabelled residents nobody's reviewed;
this is only what's been hand-labelled so far).

**"Guards don't start too early" is directionally true on cam01b but the margins are thin.**
cam01b's 79 labelled guard clips run continuously 22:09→05:23 local with only two real gaps: a
~70-minute predawn gap (03:59→05:10) and a ~28-minute evening gap (22:23→22:51). Both cam01b
resident events happen to fall in one of these gaps — 22:31 (in the evening gap) and 05:52 (29
minutes after the last predawn guard clip) — which is consistent with your framing. But that's
**n=2**, not enough to trust as a boundary, and the margins (28-29 minutes) aren't the clean "well
before the shift starts" story the hypothesis implies.

**cam09 has zero labelled guard clips at all**, so its one resident event (18:31) has no
same-camera guard timing to compare against — the "guards don't start too early" idea can't even
be checked there yet.

**Recommendation: don't build a rule on this yet.** 4 events is too thin to fit anything without
overfitting to the exact 4 examples, and the daylight assumption doesn't match what's actually
labelled. The cheapest real next step is collecting more resident examples (and checking whether
any exist unlabelled in the corpus already — nobody's run a targeted search for them the way storm/
environment candidates have been) before designing a rule around time-of-day or guard-shift gaps.

### cam07 geometry: likely NOT broken — same pattern as cam01, redirect effort elsewhere

Checked the outside_pixel_fraction distribution over cam07's 36 labelled guard clips directly
(not just the extreme tail quoted in session #7): it's **genuinely bimodal** — 36.1% read fully
inside (0.0), 36.1% fully outside (1.0), the rest spread between. That alone doesn't distinguish
"correct" from "exactly inverted," so it needed a visual check, same as cam01's.

**One clip from each cluster, rendered and inspected:**
- **cam07/4097** (inside=0.0) — already user-confirmed correct in an earlier session (2026-08-31
  QA pass). Not re-litigated.
- **cam07/22393** (outside=1.0), rendered via `scripts/render_debug.py` and inspected frame-by-
  frame (`data/reports/scratch/cam07_geometry_2026-09-08/`): a person is clearly visible standing
  on open, textured ground on the side the geometry calls OUTSIDE, while a wire-mesh fence panel
  fills the near side of frame on the side called INSIDE — visually the same shape as cam01's
  already-resolved 10560 case (a real person on the clear side, away from the fence structure).

**Since both ends of the same fence line, same camera, same era check out visually, the geometry
itself is the more likely explanation to rule out, not confirm.** The more consistent reading:
cam07's guard genuinely patrols on both sides of this fence line across different nights (walking
the interior path most of the time, sometimes inspecting the fence from outside) — the same
resolution cam01 got. **No config change made.** This means cam07's real problem is what session
#10 already found and this session's #11 work partly addressed: the *tracker* following illuminated
ground/vegetation rather than the actual subject, not a fence-tracing error. Redirect any further
cam07 effort there, not at re-tracing the line.

Debug renders for anyone who wants to re-check this call: `data/reports/scratch/
cam07_geometry_2026-09-08/cam07_22393_debug.mp4` and the extracted stills alongside it.

**RETRACTED 2026-09-09, corrected by the user who actually knows the footage: cam07/22393 is not
a real outside patrol.** It's a recurring failure mode this repo already has a name for elsewhere
(the "guard walks out of shot BEFORE the IR gain settles" pattern `warmup_flashlight_ratio` was
built to catch, `src/classify.py`'s module docstring) — the guard exits frame bottom-left *during
warmup*, so no scored frame ever contains them at all. What the geometry rule is actually reading
as "outside" in the scored frames is a spider web (visible top-right in the stills), not a person.
"cam07's guard genuinely patrols on both sides of this fence line" above is wrong; do not cite it.
This was an AI visual misread of grainy IR stills, same failure class as the stationary-light
retraction earlier in this file, not a second independent confirmation of it. The redirect to "the
tracker follows illuminated ground/vegetation, not a fence-tracing error" happens to still be
directionally right, but for the wrong specific clip and the wrong specific object (a web, not
ground/vegetation) — treat it as unconfirmed rather than settled.

## Handoff for a new agent (2026-09-08, session close #11)

**Read this section first — it supersedes #10 for current state.** 573 tests passing
(`uv run ruff check . --fix && uv run pytest -q`), branch `feat/phase1-finalisation`. This session
picked up #10's suggested next step directly: the user's own framing was "we don't need to keep
only one contour, we can keep all of them and change how the pick is made — biggest doesn't
necessarily mean the real subject, especially bright objects vs. a flashlight." That is exactly
what the previous session had half-built and left uncommitted.

### What was actually there at the start of this session

The previous session's uncommitted diff (`scripts/spike.py`, `src/features.py`) had real
plumbing — `track_contour`'s `flashlight_scores` parameter, `_run_track_pass`'s
`flashlight_scores_per_frame`, and `src.features.FLASHLIGHT_CANDIDATE_MIN_RATIO` (deliberately the
same 0.02 as `classify()`'s own `GREEN_LIGHT_RATIO_MIN`) — all correctly wired to each other. But
`detect_clip`'s own `prefer_flashlight_candidate` parameter was a dead stub: declared in the
signature, mentioned nowhere in the docstring, and never read anywhere in the function body. No
tests existed for any of it. Flipping the flag did nothing.

### What shipped this session

- **Finished the wiring.** `detect_clip` now computes `green_light_ratio` for every raw motion
  candidate in every frame when `prefer_flashlight_candidate=True`, and threads the resulting
  per-frame score lists into both the forward and the backward `_run_track_pass` calls (reversed
  correctly for the backward one). Off (the default) is a true no-op — confirmed byte-identical
  behaviour, not just "should be."
- **Threaded through the two real callers.** `extract_clip_features` gained the same parameter
  (forwarded to `detect_clip`), and `scripts/render_debug.py` gained `--prefer-flashlight-candidate`
  so a specific clip (e.g. cam07/11174) can be rendered before/after for a human to look at
  directly, not just read numbers about.
- **6 new tests** (`tests/test_spike.py`): 5 pure `track_contour` cases (prefers a lit candidate
  over a larger unlit one; still prefers size among multiple lit candidates; falls back to
  largest-wins when nothing clears the bar; `None` is byte-identical to the old behaviour; the
  min-reacquire-area floor still applies before colour is even considered) and one end-to-end
  `detect_clip` test with two independently-moving synthetic contours (a big non-green blob, a
  small green one) confirming the flag actually changes which one gets tracked.

All of the above is additive and off-by-default — safe to ship regardless of what the measurement
below says. **Not yet committed as of writing this** — do that first if picking this up fresh.

### The real measurement: 678 labelled clips, flag off vs on, real footage

Script: `data/reports/scratch/flashlight_candidate_2026-09-08/measure.py` (gitignored, local-only,
per this repo's scratch-analysis convention). Runs `extract_clip_features` twice per labelled clip
(flag off, flag on) through the real `classify()` pipeline, with the same reference-background and
`daylight_hint` wiring `scripts/backtest.py` uses in production, and diffs the resulting category.
Full per-clip output: `data/reports/scratch/flashlight_candidate_2026-09-08/results.csv`.

**18 of 678 labelled clips changed category:**

| direction | count | clips |
| --- | --- | --- |
| guard clip corrected INTO `guard_candidate` | **14** | cam13/3617, cam04/8129, cam01/8987, cam08/10852, cam08/10860, cam08/13158, cam01/16167, cam01b/16998, cam01b/17513, cam07/18318, cam01a/18603, cam14/19550, cam14/19551, cam13/22532 |
| guard clip regressed OUT of a correct read | 1 | cam08/8538: `guard_candidate` → `unclassified` |
| environment clip regressed | 2 | cam09/3970: `environment_candidate` → `guard_candidate`; cam04/20227: `environment_candidate` → `unclassified` |
| animal clip changed, alert channel unaffected either way | 1 | cam01/16028: `environment_candidate` → `guard_candidate` (never alerting before or after) |
| incident clips changed | **0** | — |

Two of the 14 corrected clips are already-named problems in this file: **cam08/10852** and
**cam01a/18603** are exactly the short/truncated-preview clips session #6 used to illustrate "the
short clip reads `incident_candidate`, the full sibling reads `guard_candidate`" — this fix
independently corrects the short clip's own read, before any sibling-waiting design is even built.
**cam01/16167** is the clip repeatedly flagged across sessions #3/#8 for flashlight-hue and
light-drift issues — also now reading correctly.

**The hard constraint held exactly.** Of the 47 labelled animal+incident clips, only 1 changed
(cam01/16028 above), and it was not in the alert channel (`incident_candidate`/`animal_candidate`)
either before or after. Zero clips moved into the alert channel that weren't there before; the 7
guard clips that moved OUT of `incident_candidate` were false alarms being fixed, not real
detections being lost. `is_blinding_foreground` also flipped True→False on 8 of the corrected
guard clips plus cam04/20227 — a secondary, consistent improvement (the tracked box is the real
subject now, not an oversized bright obstruction blob).

### The catch: this does not fix the clip that motivated it

Checked cam07/11174 directly against the real file (`data/history/cam07/11174.mp4`), not just the
aggregate corpus number: it does **not** change category (`incident_candidate` both ways). Tracing
the actual per-frame candidates at the exact fresh-acquisition frame the session #10 investigator
found: the real flashlight contour scores `green_light_ratio = 0.0178`, the bush scores `0.0`, and
`FLASHLIGHT_CANDIDATE_MIN_RATIO` is `0.02` — a **0.002 miss**, not a wrong mechanism. The threshold
was borrowed unmodified from `classify()`'s `GREEN_LIGHT_RATIO_MIN`, which is calibrated on a
finished track's best frame across a whole clip — a different, generally-higher-scoring population
than one raw candidate in one single frame. The corpus-wide win above comes entirely from *other*
clips where a candidate happened to clear 0.02, not from the named motivating example.

### A real risk found, not yet realised on this sample

Spot-checking candidate-level `green_light_ratio` (not the whole-track feature) on clips already
known to be dangerous for green-hue detection:

| clip | why it's dangerous | max candidate `green_light_ratio` |
| --- | --- | --- |
| cam03/9066 | documented night-foliage-reads-as-flashlight confound | **1.0** |
| cam03/8767 | same confound | **1.0** |
| cam10/7631 | a real animal clip already flagged elsewhere as fragile to green-light changes | **0.0246** (over the 0.02 bar) |
| cam08/7360, cam10/9405 | real animal clips (rooikat, tiny animal) | 0.0 |

A single foliage sub-blob can score a full 1.0 — worse than the whole-track version of this same
confound (which tops out at 0.42-0.45 per session #9's item 3). None of these three clips'
end-to-end `classify()` output actually changed when checked directly (confirmed by rerunning
`extract_clip_features` on all three, both flag values) — but that is three clips, not proof of
safety, and it means the margin is thin enough that a different frame or a different clip on the
same camera could plausibly cross it. This is the same shape of lesson `min_reacquire_area` and the
static-lock scoring already taught this repo: a mechanism can look clean on the cases you thought
to check and still have a real, demonstrated failure mode sitting one clip away. **Only the 678
labelled clips were swept — the other ~16,400 downloaded, unlabelled clips were not.**

### Recommendation: keep it, keep it off by default, don't stop here

Net positive on every labelled clip checked (+14/-3, zero hard-constraint cost) is a real result,
not nothing — but it is not yet safe to flip `prefer_flashlight_candidate=True` on as the default,
per this repo's own established discipline (`min_reacquire_area` needed two correction rounds after
a narrower validation missed a regression; this validation is narrower still, since it never
touched the unlabelled majority of the corpus). Current state (flag defaults to `False`
everywhere) is the correct place to leave it for now. Concrete next steps, in order:

1. **Re-derive `FLASHLIGHT_CANDIDATE_MIN_RATIO` for this specific population** (per-candidate,
   per-frame) instead of reusing `GREEN_LIGHT_RATIO_MIN`. It is simultaneously too loose (crosses
   on a pure-foliage candidate at 1.0) and too tight (misses the named cam07/11174 case by 0.002)
   — strong evidence it's the wrong number for this job, not just an unlucky threshold.
2. **Sweep the full downloaded corpus, not just the 678 labelled clips**, watching specifically for
   the foliage/animal near-miss pattern found in cam03/9066, cam03/8767, cam10/7631 — those three
   didn't flip category this time, but nothing here guarantees no clip anywhere does.
3. **Investigate the 3 real regressions** (cam08/8538, cam09/3970, cam04/20227) before accepting
   them as an acceptable cost — none were looked at frame-by-frame this session.
4. Consider requiring the lit candidate to be evidenced across more than one frame (persistence,
   not a single-frame score) before it can override the largest-area pick — would likely kill the
   single-frame foliage-spike risk above while keeping the real corrections, but unmeasured.

Debug renders for direct visual comparison:
`data/reports/scratch/flashlight_candidate_2026-09-08/cam07_11174_{off,on}.mp4`.

## Handoff for a new agent (2026-09-08, session close #10)

**Read this section first — it supersedes #9 for current state.** 567 tests passing, branch
`feat/phase1-finalisation`, incident regression 5/5, nothing uncommitted. This session was a
targeted debug pass on cam07 (rendered clips + frame-level tracing) plus a re-check of the
`cam10/7632` "accepted cost" from session #7 — the user watched the debug renders and pushed
back with specific, correct technical observations on nearly every one.

### `cam10/7632` was never actually lost — corrected, no code change

Full detail in the `MEDIAN_FENCE_DISTANCE_MAX` sections of both `docs/plan.md` and higher up in
this file (search "CORRECTED 2026-09-08"). Short version: the "no sibling clip to cover it" claim
from session #7 was wrong. `cam10/7631`, the event's own short/duplicate preview clip, already
reads `animal_candidate` independently — a live system alerts on it before the fuller `7632` clip
is even relevant. Checked and ruled out two alternative explanations before concluding this:
`7632`'s own `median_fence_distance` (0.504) is genuine, not a frozen-track artifact (barely
moves to 0.498 when recomputed from non-recovered centroids only), and no `persistence`/
`path_length` combination recovers it without re-admitting 6-15 guard/environment clips per
animal clip saved. `MEDIAN_FENCE_DISTANCE_MAX` is unchanged.

### cam07: a real bug found and fixed, and a bigger one found and NOT fixed

Rendered 7 debug clips (`data/reports/debug_render/cam07_investigation_2026-09-08/`) of cam07's
21 (of 36) misclassified guard clips. The user's per-clip review nailed the mechanism on sight for
several of them; frame-level tracing confirmed each one exactly.

**Shipped: `_warmup_motion_features`'s normalisation was blowing up on near-black frames.** This
ranker-only feature (`warmup_outside_fraction`, never a hard `classify()` rule) used to divide
each dropped IR-warmup frame by its OWN median to cancel the gain ramp. On `cam07/22289` — the
user's example of "guard moving off screen bottom-left during warmup, clearly visible" — the
first 5 dropped frames have a whole-frame median of **2.0**, an 8-bit value that produces a
**50x gain**, amplifying ordinary sensor noise into a false "changed" reading across up to 83% of
the frame. That is backwards: the dark corner where the guard actually was should have been the
*clearest* signal, not the noisiest. Replaced with the same chained `photometric_match` fit
`compensate_warmup` already uses (gain+offset against the settled background, walking frame by
frame from the one closest to settled), then diffed against the background with the SAME
threshold every scored frame uses. Before: no track found at all (`warmup_outside_fraction=0.0`
by default-empty-return). After: a real track, `warmup_outside_fraction=1.0`. Verified the
existing reference example (cam06/21377) still tracks correctly (a real, narrowing box, still
reads `inside` throughout) and the full suite stays green.

**Known, honestly-reported limitation of that same fix:** diffing against a fixed background also
flags a STATIC feature that is simply lit differently before the IR gain settles than after — not
noise, a real photometric difference, just not a moving subject. On `cam07/18570` (the user's
"latches onto bright branch, the branch was not moving" clip) the fixed warmup track's largest
contour is the *same* bright branch its scored frames separately lock onto — confirmed by the two
boxes matching almost exactly (183,91,137,87 vs 182,90,138,88). This fix does not solve that case;
telling "lit differently" apart from "moved" needs comparing frames against each other as well as
against the background, which a single frame-vs-background diff does not do.

**Found, NOT fixed — flagged for a decision, since it touches every clip in the corpus:** the
user's most valuable catch was on `cam07/11174` ("has inside flashlight tracking, that should be
enough to drown out the tracked bush"). Frame-level tracing found the actual green flashlight
glow, on the fence, in its own separate contour (`(75,87,52,92)`, area 1785px) — but `detect_clip`
picked a DIFFERENT, larger contour in the same frame (a bush, `(218,88,68,88)`, area 3025px) as
"the" subject, purely because it is bigger. `green_light_ratio` only samples inside whatever
contour won that vote, so it reads exactly 0.0000 despite the flashlight being plainly visible
one frame over. `cam07/11700`'s "weird frozen frame" turned out to be the identical pattern: an
11-frame frozen-bbox run starting 3 frames in (confirmed via `f.largest`'s bounding box staying
byte-identical for 11 consecutive scored frames) — the clip is 200 real frames (~40s at 5fps; its
`fps` metadata reads a bogus 1005, the documented corrupt-fps issue, already handled elsewhere by
`sane_fps`). **This is the same root cause identified for 11174/18570 in session #8, now confirmed
with contour-level evidence rather than inferred from bbox freezing alone: `best_contour`
selection picks by largest area with no regard for provenance (genuine vs. appearance-recovered)
or content (does it contain flashlight-hue pixels), and a large static/environmental blob
routinely outranks a smaller real one in the same frame.** Not fixed this session — changing
`best_contour` selection is corpus-wide in scope (affects `green_light_ratio`, `saturation_ratio`,
`aspect_ratio`, every shape feature on every clip, not just cam07's), and this repo's own history
(`min_reacquire_area` needed two correction rounds after a narrow validation missed a real
regression) is a direct warning against shipping a fix like this without a full labelled-corpus
sweep first. Left as a clearly-scoped, well-evidenced next step.

**Already working correctly, no action needed:** `cam07/19046` ("env that can safely flag as a
maintenance event") already reads `blinding_foreground=True` (`blob_white_fraction=0.483`) while
correctly staying out of the alert channel as `environment_candidate` — exactly the intended,
already-shipped behaviour.

**Ambiguous, not a bug:** `cam07/18318`'s own label note reads "leaf blowing" on a `guard`-labelled
clip, and the rendered still shows only foliage, no visible person — may be a genuinely hard
labelling case rather than a detector failure. Not investigated further.

### Suggested next steps

1. **Decide whether to pursue the `best_contour` selection fix.** The clearest-evidenced, highest
   -leverage open item from this session. A reasonable design direction, not yet measured: when
   multiple blobs exist in a frame, prefer one containing flashlight-hue content (or a genuine,
   non-recovered detection) over a larger one without it, rather than pure largest-area. Must be
   validated against the full labelled corpus and the standing 51-clip debug-render sample before
   shipping, per this repo's own established discipline.
2. Everything in #9's and #8's lists that is still open.

## Handoff for a new agent (2026-09-08, session close #9)

**Read this section first — it supersedes #8 for current state.** 567 tests passing, branch
`feat/phase1-finalisation`, incident regression 5/5, nothing uncommitted. This session sharpened
the flashlight test on the user's prompting, and recorded several camera facts that reframe
earlier "anomalies" as correct behaviour.

### Flashlight detection: saturation, not blob size

The user's ask was to be stricter about what counts as a flashlight so daylight grass stops
registering as one, rather than discarding every daylight clip wholesale. **The proposed
mechanism was a minimum blob size, and measuring it first showed that specific idea does not
work**: daylight grass is not scattered speckle, it forms large contiguous green regions —
median largest-per-frame component **827px, actually bigger than a real flashlight's 580px**.
Size alone is AUC 0.588.

**Saturation is the discriminator, AUC 0.907.** A flashlight's green components run S p50 168
(p90 237); sunlit grass runs S p50 69 with a p99 of only **99**. The old `min_saturation` of 60
sat below the entire grass distribution — which is precisely why this feature had ever needed a
blunt whole-frame daylight veto to be usable.

Shipped (`4492142`): `FLASHLIGHT_MIN_SATURATION` 60 → **130** and `FLASHLIGHT_MIN_BLOB_AREA`
**8**, both named constants in `src/features.py` so every consumer of `green_light_mask` shares
one definition. The blob floor is kept but is the junior partner and only earns its place after
the saturation floor has done the heavy lifting (daylight non-guard clips reading > 0.02:
13.3% → 6.7%).

**The result is a zero-leak feature.** `green_light_ratio`'s maximum across the labelled corpus
is **0.0000** on all 32 animal, all 10 incident, all 8 resident and all 5 neighbour clips, against
guard p90 0.222 and max 0.909. That is what let the daylight veto come off the scored-frame
features entirely — so the payoff is the thing that was asked for:

| 19 labelled daylight guard clips | before | after |
| --- | --- | --- |
| peak `green_light_ratio` | 0.0000 (all vetoed) | **0.4182** |
| daylight animal (21) / environment (58) / resident (3) / neighbour (5) | 0.0000 | **0.0000** |

**The veto is KEPT on `warmup_flashlight_ratio`, and that is not caution** — removing it there
was tried and flips real animals to guard (cam08/7360 the rooikat, cam08/7361, cam10/7631) on the
warmup reading alone. It is a whole-frame measure over the IR-flare frames, where a dusk clip's
colourful foliage genuinely does dominate.

`GREEN_LIGHT_RATIO_MIN` 0.05 → 0.02, because the sharpened mask roughly halves every reading and
leaving 0.05 would have silently made the rule ~2x stricter than the value anyone validated.

End to end: incident 5/5 events and 7/10 clips, animal 13/19 and 15/37 — all unchanged.
Environment leak 6 → 5, alert channel 51 → 50, guard events correct 166 → 164 (the honest cost of
a stricter test).

### cam10: the bush is real, it is outside, and it must NOT be masked

The user corrected #8: cam10's fence line is not too far left, there is a big bush outside it
driving the environment triggers. An aggregate motion heatmap over 40 of its environment clips
(`data/reports/scratch/zone_check_2026-09-08/cam10_env_motion_heatmap.png`) confirms it exactly —
the motion mass fills the right half of frame, x 0.55–1.00, centroid (0.93, 0.41), while the
fence lines sit at x≈0.42 with `outside: right`.

**An ignore polygon over the bush was the obvious next move and it is unsafe. Do not do it.**
Checking where cam10's real detections actually sit:

| clip | label | median centroid x | in the bush region |
| --- | --- | --- | --- |
| cam10/21524 | **incident** ("2 men crawling away") | 0.842 | **100%** |
| cam10/7631, 7632, 18786, 18787 | animal | 0.81–0.94 | **100%** |
| cam10/21523, 17146, 9405, 4308 | incident/animal/guard | 0.28–0.37 | 0% |

The bush region is exactly where this camera sees real intruders and animals. Masking it would
blind cam10 to its own confirmed incident.

**cam10 is in decent shape anyway**, re-scored through the current pipeline (180 clips):
environment_candidate 135 (75%), unclassified 27 (15%), and an **alert channel of only 5 clips
(2.8%), 4 of which are real** (3 animal, 1 incident, 1 environment). Ground truth is 83/94
environment, so the classifier agrees with reality. 98 of its environment calls come from
`blob_count > 10`. Its distinguishing signature is textbook wind-in-a-big-bush:
`outside_pixel_fraction` p50 1.000 (corpus 0.000), `median_fence_distance` 0.444 (0.151),
`blob_count` 11 (4), `row_normalised_area` p90 188,902 (58,676), `path_length` 86.6 (22.4),
`heading_change` 1.55 (0.65) — a big blob far outside the fence, moving constantly and going
nowhere.

Note `scenery_motion_fraction` reads p50 0.000 on cam10 against 0.029 corpus-wide: the
reference-background test does **not** recognise the bush as known scenery. That is the
already-documented NCC brightness-invariance failure from session #4, showing up again.

### Camera facts from the user that reframe earlier findings

- **cam01 was replaced by cam01a**, the pole-mounted one. cam01a reads its own pole as very
  bright and the guard flashes the camera directly — between them that explains its 33%
  overexposure and 37% blinding rate, the worst on site. It is now in `NO_MAINTENANCE_CAMERAS`
  alongside cam04; it needed to be, because `post_flash_red_shift` does not catch it (39 of 43
  flagged clips survive that filter). Obstruction windows 30 → 27.
- **cam15 has never seen a human** — very remote. So #8's "cam15 is flashlight-dead" is not a bug:
  0.0% green-light firing is the correct answer. Recorded in `config/cameras.yaml` so nobody
  tries to fix it. Its most-saturated pixels do sit at hue 33.0, exactly `hue_low`, which would
  matter if a human ever does appear there.
- **cam01/10560-10561 is a worker, i.e. `neighbour`, not `resident`** — relabelled (labels backed
  up to `data/backups/labels_pre_neighbour_fix_20260908.jsonl` first). This also answers #8's open
  cam01 question in part: that person genuinely *is* outside the fence, so `outside: right` read
  him correctly. cam01's 62%-outside rate is not automatically a geometry bug.

### Still open

- cam07's beam-washed-vegetation mode remains the biggest alert-channel source and is not a colour
  problem — the tracker follows the illuminated ground outside the fence rather than the light.
  No colour feature will fix it.
- cam01's remaining overexposure is a real vine against the lens (distinct from cam01a's pole).
- Everything in #7's list, particularly the wait-for-sibling design question.

## Handoff for a new agent (2026-09-08, session close #8)

**Read this section first — it supersedes #7 for current state.** 568 tests passing, branch
`feat/phase1-finalisation`, incident regression 5/5, nothing uncommitted. This session was a
per-camera anomaly hunt prompted by #7's finding that the residual leak is concentrated on a few
cameras. It found one systematic bug affecting a third of the corpus, and corrected two
long-standing assumptions.

### THE bug: the daylight gate silently disabled flashlight detection on a third of all night clips

`scripts/spike.py`'s gate that zeroes `green_light_ratio`/`green_light_flicker` is
`color_fraction > 0.15` — purely an image statistic, and **it is not camera-neutral**. Across all
16,272 genuinely-night clips by the sun table, **29.7% trip it anyway**, concentrated exactly in
the cameras that record colour-cast night footage:

| gate fires on night clips | | |
| --- | --- | --- |
| cam16 95.4% | cam01b 94.8% | cam14 87.0% |
| cam04 55.4% | cam01a 54.8% | cam01 43.3% |
| cam05 40.6% | *vs* cam07 3.8% | cam06 5.3%, cam02 5.6% |

On those cameras every flashlight feature reads 0.0 on nearly every clip. Corpus-wide the
green-light rule fired on **1.1% of cam01b's clips and 2.6% of cam16's, against 40.4% of
cam03's**. On the labelled corpus, gated night guard clips were caught by a flashlight rule 27.2%
of the time vs 46.5% for ungated ones — the gate roughly halved flashlight detection wherever it
fired.

**It was not that those cameras cannot see the light.** Sampling raw pixels with no gate applied:
12/12 cam01b and 12/12 cam16 night clips carry >50 green-mask pixels, up to 36k, at hue 36–40 and
saturation 255. The green is there, correctly coloured, and was being thrown away.

**Fixed** (`147fbd5`, `0a384d8`) the same way the cam03 render confound was fixed: use the clock.
New `src.features.daylight_hint(timestamp)` (True for daylight OR either twilight margin, `None`
without a usable timestamp) feeds a new `extract_clip_features(daylight_hint=...)` parameter that
overrules the image statistic when the sun table says night. Wired into **every** production
caller — backtest, both ranker paths, the storm sweep, the incident regression check, and
render_debug — so they can never disagree about whether a clip was shot at night. `None` is
byte-identical to the old behaviour.

| labelled guard clips caught by a flashlight rule | before | after |
| --- | --- | --- |
| cam01b | 11/79 | **71/79** |
| cam04 | 21/64 | **49/64** |
| cam05 | 18/42 | **29/42** |
| cam16 | 0/8 | **6/8** |
| cam01a | 9/24 | **15/24** |
| all guard clips reading `guard_candidate` | 277/429 | **328/429** |

At event level: guard events correct 136 → **166** of 222, guard leak into the alert channel 26 →
**22**, guard misrouted to environment 41 → **19**, alert channel 55 → **51**. Incident stays 5/5
events and 7/10 clips, animal 13/19 and 15/37, environment leak unchanged at 6. **Of the 52 clips
that change category, all 52 move INTO `guard_candidate` and none leaves it.**

The risk that made this worth measuring rather than assuming: overruling the gate could have made
green foliage on colour-cast night footage read as a flashlight. It did not — every clip that left
the alert channel is a guard-labelled clip now correctly called a guard, and a test covers the
no-green-in-frame case.

### cam07: the flashlight is fine. The tracker is looking at the wrong thing.

The hypothesis was a colour shift on cam07. **Not supported.** cam07's flashlight is green and
detected normally — `green_light_ratio` p90 0.268 (3rd highest of any camera), the green-light rule
fires on 21.1% of its clips, and only 3.8% of its night clips trip the daylight gate (one of the
lowest rates on site).

Rendering its mislabelled guard clips through `scripts/visualize_zone.py` showed the real
mechanism. On **cam07/11174** (labelled guard, note "flashlight") there is an obvious bright green
flashlight sitting on the fence — and the tracked box is 40% of the frame away, on the bushes the
beam is lighting up *outside* the fence. `green_light_ratio` only samples pixels inside the tracked
contour, so it reads exactly 0.000, and the clip then reads `incident_candidate` because the blob
it did track is outside. **cam07/18570** is the harder variant: the beam washes the vegetation with
no green at all (`color_fraction` 0.000), so nothing colour-based can ever catch it.

That is why cam07 produces **14% `incident_candidate` corpus-wide, the highest of any camera and
7x cam06's 2%** — and with 3,768 clips (23% of the whole corpus) it is the single largest source of
incident-channel volume.

`whole_frame_green_ratio` was added (`70cede2`) as the scored-frame counterpart of
`warmup_flashlight_ratio`, and it does see the light the tracked box misses. It is **deliberately
not a rule**: guard p90 0.244 and max 0.881 against every incident under 0.00074 looks strong, but
the worst real *animal* clip sits at 0.01119, leaving only 1.8x margin (the comparable
`warmup_flashlight_ratio` rule carries 4.3x), and against the post-fix rule set it buys one event.

### cam01: it is mostly catching a vine against the lens

Rendered four cam01 clips. Its left half is permanently occupied by an out-of-focus vine/branch
tangle right against the lens, which the IR illuminator lights up brilliantly — visible in
16166, 9553 and 8988, and matching one of its own label notes ("blocked by vine, very difficult
to"). Measured: **20.2% of cam01's clips are overexposed (4x the 5.1% corpus rate) and 22.5% carry
the blinding flag.**

Its other anomaly is geometric: **62.0% of cam01's clips read outside the fence (3.4x the corpus
18.3%)**, while its median fence distance is a perfectly normal 0.145 and its blobs are *smaller*
than average. Small blobs, close to the fence, on the wrong side of it. The daylight frame
(cam01/10560) shows a guard walking a clear path on the right of the fence, labelled OUTSIDE by
the current `outside: right`. **Either that patrol path really is outside the fence there, or
`outside` is flipped for cam01** — deliberately not changed, because this repo's own record says
to trust the user's eyes over a plausible theory. One look at that render settles it.

### Other per-camera anomalies worth knowing

| camera | anomaly | vs corpus |
| --- | --- | --- |
| cam10 | 84.8% of clips read outside; fence line sits far left in frame so nearly everything visible is nominally outside; 63% environment_candidate | 18.3% |
| cam01a | 33.3% overexposed, 36.8% blinding — the worst artifact camera on site | 5.1% / 11.9% |
| cam16 | 13.9% long-IR ramp, 34.3% unclassified | 4.6% / 10.2% |
| cam15 | flashlight-dead for a *different* reason than cam01b/cam16 — its gate rate is only 6.3%, but its most-saturated pixels sit at hue 33.0, exactly the `hue_low` bound | — |
| cam10 | its 22k "green" pixels are the documented fixed light above the fence post, already in an ignore polygon — correctly excluded, not a miss | — |

### Corrections to earlier sections of this file

- **#4's "only cam06 has a `fence_bottom`" is wrong** — all 18 cameras have one, and most have a
  picket trace too.
- **The "DAYLIGHT GATE IS SELF-DEFEATING" finding was right about the mechanism and wrong about
  the remedy.** Four attempts to re-derive daylight *from the pixels* failed, correctly. Nobody
  had tried simply asking the clock, even though `is_daylight` had already been threaded into
  `classify()` for the resident split and into `render_debug` for the overlay. The lesson is
  narrower than "this is unfixable": an image statistic could not separate a flashlight from
  daylight, so the answer had to come from outside the image.

### Suggested next steps

1. **Look at `data/reports/scratch/zone_check_2026-09-08/cam01_10560.png` and say whether that
   patrol path is inside or outside the fence.** One answer settles cam01's 62% outside rate.
2. **cam07's beam-washed-vegetation mode is the biggest remaining alert-channel source** and is
   not a colour problem, so no colour feature will fix it. The tracked blob being the illuminated
   ground rather than the subject is a detector/tracker question.
3. cam10's fence line is worth re-checking for the same reason as cam01 — 84.8% outside is not a
   threshold problem.
4. Everything in #7's list that is still open, particularly the wait-for-sibling design question.

## Handoff for a new agent (2026-09-07, session close #7)

**Read this section first — it supersedes #6 for current state.** 557 tests passing, branch
`feat/phase1-finalisation`, incident regression 5/5 events, nothing uncommitted. This session was
a threshold-mining pass over the full-corpus data from #6, run to the user's stated priority
order: never lose an incident, keep animals, minimise guard and environment, and improve the
maintenance channel.

### Headline: the alert channel is 30% smaller at zero cost to incidents or animals

Measured end to end by re-scoring the whole labelled corpus through the shipped pipeline (not
the analysis copy), at EVENT level over 351 labelled events:

| | before | after |
| --- | --- | --- |
| incident events alerting | 5/5 | **5/5** |
| incident CLIPS alerting | 7/10 | **7/10** |
| animal events alerting | 13/19 | **13/19** |
| animal CLIPS alerting | 15/37 | **15/37** |
| guard leak into the alert channel | 39/222 (17.6%) | **26/222 (11.7%)** |
| environment leak | 16/82 (19.5%) | **6/82 (7.3%)** |
| alert channel size | 79 events | **55 events** |
| environment correctly read as environment | 54/82 | **70/82** |

**Not one protected-class clip changed category.** 46 clips moved, all of them guard/environment/
unlabelled. Alert-channel precision goes 22.8% → 32.7% on labelled events. That last row also
closes the "environment is the weak link now, not guard" finding from #5.

### What shipped (3 commits)

`b8ec6dd` — **`blob_count_median` and `motion_pixel_fraction_median`** in `scripts/spike.py`.
Both existing features are a PEAK over frames, which is what a storm needs but also fires on a
single flare-settle frame at the start of a quiet clip — the bug diagnosed in #5 and left unfixed
because touching the peak thresholds needed a full re-validation. Adding the median alongside
sidesteps that entirely: no existing value moves. The medians separate the protected classes
better than the peaks they de-noise (motion_pixel_fraction AUC 0.236 → 0.271 inverted,
blob_count 0.297 → 0.325), and the gap on the protected side is enormous — the worst real animal
clip PEAKS at 0.907 whole-frame motion but has a MEDIAN of 0.0015.

`79e426c` — **three gates in `scripts/backtest.py::classify()`**:

1. `blob_count_median > 4` → `environment_candidate`, beside the existing peak `blob_count > 10`.
2. `blob_white_fraction >= 0.4` inside the geometry rule → **a blinded lens never alerts**. Same
   threshold `is_blinding_foreground` already uses, deliberately: when a bright obstruction is
   against the lens, the "outside blob" IS the obstruction. The flag stays orthogonal for
   reporting.
3. `motion_pixel_fraction_median > 0.12` → sustained whole-frame motion is wind or an unsettled
   IR ramp, not a compact intruder.

`5cf2b82` — **`obstruction_windows()` in `scripts/rank_candidates.py`**, written as a
`.windows.csv` sibling whenever `--maintenance-out` is given. See "the maintenance channel" below.

### The methodology that matters more than the thresholds

**Every threshold was chosen on PER-CLIP protected-class margin, not event-level survival.**
Event level is the right place to MEASURE (per #6's truncated-preview finding), but a live system
sees one clip at a time and cannot rely on a sibling covering for a suppressed one. Concretely:
`blob_count_median > 3` and `> 2` both keep 5/5 incident EVENTS while silently gating the real
incident clip `cam08/4054` (median exactly 4.0). That is why the shipped threshold is 4 and not
the more aggressive value the event-level Pareto frontier preferred. The aggressive package
(median 2 / mpf_median 0.10 / long-flare veto) reaches guard leak 19 and environment leak 3 —
it is available and measured, and it costs that one incident clip.

### Two things measured and deliberately NOT shipped

- **Peak `motion_pixel_fraction` as the gate.** It is the single strongest discriminator in this
  population (AUC 0.236, i.e. 0.764 inverted — better than anything else tried), and a 0.20 bound
  drives guard leak to 24 and environment to 7 on its own. Rejected because its worst real
  incident clip sits at 0.123 against a guard/environment median of 0.172, and every threshold
  tight enough to be useful also costs an animal event. Worse, a real incident clip that does NOT
  currently alert (`cam09/21521`) reads 0.308 — so a future incident CAN be up there. The median
  is both safer and free.
- **`long_flare_frames >= 18` as a fourth gate.** Buys only 1-2 more events, and `cam06/21520` —
  the crawl incident, the signature threat this system exists to catch — sits at 15. Same
  reasoning that rejected the `metric_aspect` gate. It stays a maintenance signal only.

Also quick-checked and worthless as a `classify()` rule: `post_flash_red_shift` as a guard rule
(the "immediate next step" #5 recommended). Swept 0.05–0.60 through the real rule order: it moves
guard leak by at most 2 events, because the guards it identifies are almost all already caught by
the rules above it. #5's "zero-leak corpus-wide" description is accurate but irrelevant at this
position in the chain — measure a rule where it would actually sit, not corpus-wide.

### The maintenance channel: the per-clip flag is not a work order

**Measured, and it's a clean negative.** On the 16,554-clip corpus `is_blinding_foreground` fires
on 8.9% of clips but shows almost no temporal clustering: P(next clip from the same camera also
flagged | this one flagged) runs 0–19% per camera, and the longest consecutive run anywhere is 5.
A spider web on a lens does not behave like that — it would flag every clip until someone wipes
it off. So the per-clip flag measures transient bright events, and the 1,019-clip queue built on
it is not something anyone can act on.

**The RATE over a window is the signal.** Cameras sit at a 4–23% baseline and specific 14-day
spans hit 25–58%. `obstruction_windows()` produces **30 windows across three years** — roughly one
a month, which is a work list. The clearest case is cam07: 6.4% baseline, 43.1% across 66 clips
over 2026-02-06..03-12, 27 flagged clips spread over 12 separate nights, `blob_white_fraction`
0.41–0.93, `post_flash_red_shift` at or below 0.027 on every one (so not the guard's flashlight).
Several of those same clips were also leaking into the alert channel as `incident_candidate`
before gate 2 above — one physical obstruction showing up in both channels, which is
corroboration rather than coincidence.

`NO_MAINTENANCE_CAMERAS` is honoured for consistency, but **cam04 produces two strongly elevated
windows (47.4% and 43.9% against a 9.1% baseline)** — that human override was made on the
per-clip view and is worth re-checking against the windowed one.

### Where the residual 32 leaking events are

Heavily concentrated, and not evenly by exposure:

| camera | leaking events | labelled events | rate |
| --- | --- | --- | --- |
| cam07 | 7 | 30 | 23% |
| cam01b | 4 | 43 | 9% |
| cam08 | 3 | 19 | 16% |
| cam13 | 2 | 9 | 22% |
| cam03 | 2 | 11 | 18% |

**The likely cause is geometry, not thresholds, and #4's "only cam06 is calibrated" note is now
STALE — all 18 cameras have a `fence_bottom`.** What is anomalous is the fraction of each
camera's own guard clips reading `outside_pixel_fraction > 0.6`: the corpus norm is ~24%, but
**cam01 reads 78.9% (15/19) and cam07 41.7% (15/36)**, with cam16 50% (n=8) and cam01a 41.7%.
Either those two cameras' fence traces are wrong, or their guards genuinely patrol outside the
fence there. That is a question for the user's own eyes on a couple of clips, not another
threshold sweep, and it is the highest-value next step for cutting guard leak further.

### Suggested next steps, in priority order

1. **Check cam01's and cam07's fence geometry.** Highest leverage on the remaining leak, and it
   is a visual question, not a statistical one.
2. **Settle the wait-for-sibling design question** — still open from #6, still unmeasured, and
   the numbers above make it sharper: an OR-across-siblings combiner buys recall the system
   already has (5/5, 13/19) while making the leak worse.
3. **Label more `environment` clips with notes.** Environment recall is now 85% and the leak is
   down to 6 events, but there are still only 82 labelled environment events and just a handful
   carry notes; the low-blob-count single-branch-oscillation population from #6 is still
   unsolved and needs examples before it can be measured.
4. The aggressive gate package is measured and sitting there if the user decides one incident
   CLIP (never an event) is an acceptable price for guard leak 26→19 and environment 6→3.

### Still open, unchanged by this session

- Gate 2 pass/fail — the standing decision, still the user's to make.
- **`resident` has no rule beyond the daylight split** of the inside-only fallback, and zero
  dog-labelled clips exist.
- 3 of 10 incident clips still read fully inside and are carried by their sibling. Not a
  threshold problem — the crawling subject is at the fence base. Documented, not regressed.

## Handoff for a new agent (2026-09-07, session close #6)

**Read this section first — it supersedes #5 for current state.** 541 tests passing, branch
`feat/phase1-finalisation`, incident regression 5/5 events. Nothing uncommitted. This session was
an extended discovery/labelling pass (daylight-outside then night-outside candidate batches) that
surfaced one finding big enough to change how the next agent should measure anything: **many
ground-truth clips are a literal duplicate-prefix of a longer sibling, and the two often produce
different `classify()` categories.** Read that before trusting any per-clip metric, including ones
quoted earlier in this file.

### Fresh full-corpus data, ready to work from

Just re-ran `scripts/backtest.py`'s full logic (not `--labelled-only` — every downloaded clip)
via a parallel wrapper (`data/reports/scratch/full_history_2026-09-07/`, not committed — CSVs
only, regenerate with the `_worker`/`Pool(8)` pattern in that script if it's gone from `/tmp`):

- **`per_clip.csv`** — 16,886 rows, one per downloaded clip, same columns as
  `scripts.backtest.write_csv` (`REPORT_COLUMNS`: identity + `category` + `blinding_foreground` +
  every `FEATURE_COLUMNS` value) plus `caption` (needed to pair siblings).
- **`per_event.csv`** — 8,613 rows, one per physical event (grouped by
  `scripts.label._event_key`, i.e. clips sharing an embedded `@ HH:MM:SS` caption timestamp),
  with `event_category` computed by the new `scripts.backtest.classify_event()` (below) and a
  `message_ids` column listing every clip in that event.

Corpus-wide category counts (`per_clip.csv`, ALL 16,886 clips, mostly unlabelled):
`guard_candidate` 13,009, `unclassified` 1,691, `incident_candidate` 741, `environment_candidate`
699, `no_motion` 332, `animal_candidate` 215, `resident_candidate` 164, `insect_candidate` 35.

On just the 678 labelled clips: label counts are `guard` 429, `environment` 159, `animal` 37,
`unknown` 30, `incident` 10, `resident` 10, `neighbour` 3 (`neighbour` is new this session — see
below). These 678 rows collapse to **only 351 unique events** — nearly half are a duplicate
half of a pair, see next section.

### THE finding: truncated preview clips distort every per-clip metric, `classify_event()` shipped

Many "(Initial*)" alert messages are a literal frame-for-frame prefix of the "(Stopped*)" message
that follows a few minutes later — and the two routinely produce **different** `classify()`
categories, because the short preview catches the scan before a flashlight/track settles.
Confirmed twice on real footage: `cam01a` 18603 (short, reads `incident_candidate`) / 18604 (full,
reads `guard_candidate`, `green_light_ratio=0.459`); `cam08` 10852 (short, `incident_candidate`,
"quick blob outside") / 10853 (full, `guard_candidate`, lingering flashlight, `outside_pixel_
fraction=0.0`).

Since both clips of an event are usually labelled identically once either one is (label
propagation), **`scripts/backtest.py --labelled-only` was double-counting almost every
animal/incident event**: 47 labelled animal/incident clip ROWS collapsed to only 25 unique
EVENTS when this was audited — 88% of them are exactly this short+long pair, both carrying the
same label. Every feature/AUC/threshold number quoted anywhere earlier in this file (`docs/
handoff.md` sessions #1-#5) or in `docs/plan.md` up to today's checkpoint was computed over that
inflated, truncation-mixed population — not a fabricated concern, a confirmed measurement bias.

**New tool shipped, not a `classify()` change:** `scripts.backtest.classify_event(categories) ->
str` reduces a whole event's sibling categories to one verdict; `incident_candidate`/`animal_
candidate` always win (matches the standing "shape may never suppress an outside alert" rule).
Analysis/reporting only — never called from the live per-clip path, since a real system sees one
clip at a time and can't know a sibling's category before it exists. Re-measured the full labelled
corpus grouped by event:

| | per-clip | event-level (any sibling fires) |
| --- | --- | --- |
| incident recall | 70% (7/10) | **100% (5/5)** |
| animal recall | 40.5% (15/37) | **68.4% (13/19)** |
| guard leak into alert channel | 10.7% (46/429) | **18.5% (41/222)** |
| environment leak into alert channel | 10.1% (16/159) | **19.5% (16/82)** |

**The nuance that matters most for whoever picks this up next:** recall improves at event level
(good, no incident ever lost either way) but guard/environment leak gets WORSE, not better —
because "any sibling fires" is exactly the false-alarm exposure a live system reacting to every
message independently already has today. **A naive OR-across-siblings combiner only helps
recall, it does not fix precision.** The open, undecided design question (documented in `docs/
plan.md`'s newest checkpoint, not built): should a future live system wait for the "(Stopped*)"
sibling and specifically trust the LONGER clip's read — not just union every sibling's category —
before alerting? The root cause observed both times so far is specifically that the SHORT clip is
the unreliable one; trusting the fuller clip once it exists is the more promising direction than
plain unioning, but it's unmeasured. This is probably the single highest-value thing to actually
measure next, since it would settle a real design question rather than add another feature.

### Other confirmed findings from this session, all recorded in `/memories/repo/incident-findings.md`

That file has the full narrative with exact numbers for each of these — this is the index, not
the detail:

- **`ANIMAL_ROW_AREA_MAX = 3000.0`** shipped: real animal blobs are small (`row_normalised_area`
  median 666) vs. environment leaking as animal (median 10,471, >15x gap) — a user-sourced
  hypothesis from watching footage, not a feature-fishing result. Environment leak into the alert
  channel 22%→9.8% on the corpus at the time, zero animal cost. Deliberately NOT applied to
  `incident_candidate` (real incident clips get wrongly gated at every threshold tried).
- **`resident_candidate` rule** added, split from the inside-only guard fallback by real
  `is_daylight(timestamp)` — NOT `color_fraction` (rejected first: 64% guard false-fire, the same
  "daylight gate is self-defeating" confound as `green_light_ratio`).
- **`neighbour` label added** (`VALID_LABELS`, migration `scripts/migrate_add_neighbour_label.py`
  already run) — a benign person outside the fence, not resident/guard/threat (a neighbour's
  worker). n=3 now, still too thin for a `classify()` rule of its own.
- **cam09 fence geometry has a dated history now** (like cam01a/cam12) — the user directly traced
  a real remount in `config/cameras.yaml`, effective `2025-01-12T16:22:45Z`. Trust the user's own
  visual trace over a plausible alternative theory — this repo has hit that lesson more than once.
- **Dawn/dusk guard-flashlight miss (n=2, unmeasured further):** at twilight, `green_light_ratio`/
  `green_light_flicker`/`warmup_flashlight_ratio` can all read exactly 0.0 (not just below
  threshold) while `color_fraction` is high from real ambient light, so a guard's flashlight falls
  through every guard rule into `animal_candidate`. Mirror image of the already-rejected
  `color_fraction`-as-daylight-proxy confound.
- **Guard-exits-during-IR-warmup, bottom-left corner (user-observed, unmeasured):** several guard
  clips this session showed very low `persistence`/`longest_detection_run` with a real subject
  visible only briefly before leaving early in the warmup window. No exit-direction feature exists
  yet to confirm the "bottom-left" specifics — would need frame-level track-path data, not just
  the aggregate features already computed.
- **Full night-time-outside discovery is barely started.** 16,582 of 16,886 clips are night (vs.
  only 304 real-daylight clips, which is why the daylight sweep finished this session but night
  didn't) — of 2,609 unlabelled night+outside candidates, only a 37-clip stratified sample was
  reviewed (`data/reports/scratch/night_outside_review_2026-09-07/`), and even that batch isn't
  fully labelled. This is the largest unexplored population in the whole corpus.
- **Ranker (`scripts/rank_candidates.py`) group-drop improvement measured, not implemented:**
  dropping the track/motion AND metric/calibration feature groups together improves leave-one-out
  AUC by +0.064 (animal+incident vs. rest) — lower risk than a `classify()` change since it only
  affects review-queue ORDER, not the alert/suppress decision.

### Suggested next steps for a fresh empirical pass, in priority order

1. **Settle the wait-for-sibling design question** using `per_event.csv` above — specifically
   compare "trust the longest clip in the event" against "union all siblings" against "per-clip,
   no waiting" for guard/environment leak, not just recall (the OR-combiner already measured here
   only tells half the story).
2. **Mine `per_clip.csv`/`per_event.csv` for the low-blob-count single-branch-oscillation
   environment population** — flagged in `/memories/repo/incident-findings.md` as NOT yet solved
   (`heading_change` alone was measured and rejected; `blob_count>10` cannot catch a single
   waving branch).
3. **Use the full corpus counts above to plan a smarter night-time discovery batch** — 741
   `incident_candidate` clips corpus-wide is a lot to leave unreviewed on the highest-priority
   channel; `per_event.csv`'s event-level counts (693 events read `incident_candidate`) are a
   better base for stratified sampling than raw clip counts, since it won't re-sample both halves
   of the same event.
4. Everything in "other confirmed findings" above that says "unmeasured" is a real, evidenced
   lead — none need re-discovering, they need a proper threshold sweep against the fresh data.

## Conventions

- `uv` for everything: `uv run ruff check . --fix && uv run pytest -q` after every module change
  and before every commit.
- Commit logically per change, stay on `feat/phase1-finalisation`, never auto-merge.
- After any detector/feature change, re-render the standing `data/reports/debug_render/` sample
  set and numerically diff old vs new before trusting the change.
- `src/` is flat, no package. `scripts/` are standalone entry points that `sys.path`-insert the
  repo root.
- DB timestamps are UTC; site local time is UTC+2 (SAST).
- Secrets live in `.env` — never grep or cat it, and never scope a search broadly enough to
  match it.

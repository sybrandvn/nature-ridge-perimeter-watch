# Handoff: gate-2 pass/fail is still open; Phase 1 foundation is now built

Written 2026-08-28, updated repeatedly since; last updated 2026-09-07. **If you are a new agent
picking this up, start at "Handoff for a new agent (2026-09-07, session close #7)" at the very
bottom, just above "Conventions"** — it has current state, the current measured classifier
numbers, and the highest-value open question. `docs/plan.md` is the full plan and stays
authoritative; this file is the short version of where things actually stand and what to do next.

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

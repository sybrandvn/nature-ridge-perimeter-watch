# Plan: Nature Ridge Perimeter Watch

Discover the camera layout from message metadata, validate the classical-CV hypothesis on a small
labelled sample, then build the backfill → label → backtest calibration loop. Live service, Docker,
and delivery-reliability machinery are deferred until thresholds are proven. Alert modules ship as
tested, manually-invokable functions only.

## Checkpoint (2026-09-07, cont.): truncated-preview clips distort per-clip metrics; open question on waiting for a sibling before alerting

Full measurement and discovery narrative in `/memories/repo/incident-findings.md` (search
"MAJOR FINDING" and "classify_event"). Summary here for a reader who only checks this file.

**Finding: many "(Initial*)" alert messages are a literal frame-for-frame prefix of the
"(Stopped*)" message that follows a few minutes later, and the two routinely produce DIFFERENT
`classify()` categories** — the short preview catches the scan before a flashlight/track settles,
so it reads as `incident_candidate`/`animal_candidate` while the full clip is plainly
`guard_candidate` (confirmed twice this session: cam01a/18603-18604, cam08/10852-10853). Since
both clips of an event are usually labelled identically once one is, `scripts/backtest.py
--labelled-only` was **double-counting almost every animal/incident event** (47 labelled clip
rows collapsed to only 25 unique physical events when grouped by `scripts.label._event_key`) —
every feature/AUC/threshold number quoted earlier in this file was computed over that inflated,
truncation-mixed population, not the true one.

**New tool, not a `classify()` change:** `scripts.backtest.classify_event(categories)` reduces a
whole event's sibling categories to one verdict, `incident_candidate`/`animal_candidate` always
winning (matches the standing "shape may never suppress an outside alert" rule) — reporting/
analysis only, never called from the live per-clip path. Re-measured the full labelled corpus
grouped by event:

| | per-clip | event-level (any sibling fires) |
| --- | --- | --- |
| incident recall | 70% (7/10) | **100% (5/5)** |
| animal recall | 40.5% (15/37) | **68.4% (13/19)** |
| guard leak into alert channel | 10.7% (46/429) | **18.5% (41/222)** |
| environment leak into alert channel | 10.1% (16/159) | **19.5% (16/82)** |

Recall goes up (good, and no incident event is ever lost either way). But guard/environment leak
also goes up at event level, not down — because "any sibling fires" is exactly the exposure a
live system that reacts to every message independently already has today. Naively combining
siblings with an OR doesn't fix precision, only recall.

**Open decision, not yet made:** should a future live system delay alerting on an "(Initial*)"
message for the few minutes until its "(Stopped*)" sibling arrives, and prefer the FULLER clip's
read specifically (not just OR across all siblings) before firing? The root cause observed twice
so far is specifically that the SHORT clip is the unreliable one — trusting the longer clip once
it exists, rather than merely unioning categories, is the more promising fix hinted at by the
data, but is unmeasured and unbuilt. Deferred along with the rest of the live-alerting pipeline
(see the opening paragraph of this file) — revisit once the live service itself is scoped.

**2026-09-19 evidence update:** a second detailed review confirmed the same mechanism on
cam01b/17500→17501, cam03/5465→5464, cam07/18641→18642, and cam03/20521→20522. In each event the
short clip either loses the person during IR warmup or later locks onto unrelated scenery/noise,
while the fuller sibling carries clear guard evidence. Warmup-only onside motion can now recover a
guard only when the settled detector found no subject; it intentionally cannot override a later
outside subject because confirmed footage can contain both. This strengthens the case for holding
Initial previews for the completed sibling, but the production policy remains unbuilt and should
be measured at event level before the live orchestrator is implemented.

## Checkpoint (2026-09-07): triage router built; classifier state re-measured

Supersedes the stale figures below wherever they conflict. Full narrative in `docs/handoff.md`
section #5; measurements in `/memories/repo/incident-findings.md`.

**Triage now has three outputs, not one queue:** the incident/animal review queue
(`scripts/rank_candidates.py --out`), a separate maintenance "clean this camera" queue
(`--maintenance-out`), and a `blinding_foreground` flag in `scripts/backtest.py` that is
deliberately *independent* of `category` — a clip can be a real guard sighting and an obstructed
lens at the same time.

**Classifier state, measured 2026-09-07 across 534 labelled clips** (this replaces every earlier
recall number in this file):

| true label | outcome |
| --- | --- |
| guard (376) | **71.8% correct** — 34.8% from the inside-only rule, 29.5% from warmup flashlight, 7.4% from green light. 8.8% leak to the alert channel |
| environment (100) | 50% correct; **29 clips leak into the alert channel** |
| incident (10) | 70% correct; 30% suppressed as guard, safe only at event level (5/5) |
| animal (13) | 46.2% correct |
| resident (10) | **no rule exists** |

**The "guard recall is 8.7%" figure repeated throughout this repo is wrong** — it was only the
`green_light` rule's own recall, measured before the two rules that now do the work existed.

**Recall is no longer the bottleneck; leakage is.** The environment leak is the largest single
false-alert source and is 55% concentrated on cam10.

**Decision made and shipped (2026-09-07):** the user chose to add the *upper* bound. `scripts/
backtest.py::classify()`'s animal/incident geometry rule now requires `median_fence_distance <
MEDIAN_FENCE_DISTANCE_MAX (0.40)` as well as `> 0.1`. Re-measured through the real rule order:
environment leak into the alert channel drops 29 -> 14 clips, all 5 incident events still alert
(`scripts/check_incident_regression.py` passes 5/5), and the accepted cost is exactly one animal
event with no sibling clip to cover it, `cam10/7632` (now `unclassified`, was already noted
"visually marginal/hard to confirm, enters during IR flare" at labelling time). This is a
deliberate, explicit break of the standing "never lose an animal" constraint for this one known
clip, not a silent regression.

**CORRECTED (2026-09-08): `cam10/7632`'s event is not actually lost.** "No sibling clip to cover
it" was wrong -- its short/duplicate preview sibling `cam10/7631` independently reads
`animal_candidate` on its own (`median_fence_distance` 0.365-0.370, under the bound), confirmed via
`scripts.backtest.classify_event(['animal_candidate', 'unclassified']) == 'animal_candidate'`. A
live system processing Telegram messages one at a time would alert on `7631` before the fuller
`7632` even arrives, so this sighting is not lost in practice -- only the standalone `7632` clip
fails to independently re-confirm it, which is what "accepted cost" was actually describing. Not
a reason to revisit `MEDIAN_FENCE_DISTANCE_MAX`: `7632`'s own median_fence_distance (0.498-0.504)
was checked against genuine-only centroids (excluding recovered/hallucinated frames) and barely
moves, confirming the animal really was far from the fence, not an artifact of a frozen track. A
persistence+path_length combination was tried as an alternative discriminator to recover it and
similar clips without the bound; rejected, since even the tightest safe-looking combination
re-admits 6-12 environment and 8-15 guard clips for every 2-4 animal clips recovered -- a clearly
bad trade against the leak this bound exists to stop.

**IR flare tracking implemented (2026-09-07):** `src.motion.detect_clip(compensate_warmup=
True)` now does real per-pixel tracking through the dropped IR-flare/warmup window (a
photometric gain/offset fit onto the settled background, then the same diff/track pipeline as
every scored frame), plus brightness/colour-corrected warmup frames for display. Scoped safe by
construction -- it only improves the already render-only `dropped_frame_boxes`, never touches
`FEATURE_COLUMNS` -- and wired into `scripts/render_debug.py`'s debug videos. Full detail,
validation numbers and the one known remaining limitation (`cam06/21520`'s dwelling subject) in
`docs/handoff.md`'s "IR flare tracking" section.

## Checkpoint (2026-08-30): Phase 0 merged to `main`, needs work before Phase 1 starts
Phase 0 (steps 1-14) is merged and tagged as a checkpoint, not a clean sign-off. Open items before
Phase 1 work should build on this without inheriting stale numbers:
- Gate 2 pass/fail is still the user's explicit call, not made (see step 14 and
  `docs/handoff.md`).
- `docs/gate2_separability_finding.md`'s main body predates both a day of extra labelling and the
  2026-08-30 `zones.py` geometry fix — only its "Addendum 2 (2026-08-30)" section is current.
  Rerun the main-body analysis on `data/reports/spike_all_2026-08-30.csv` before finalising gate 2.
- The guard/environment/cam01b-resident `outside_pixel_fraction` inconsistency noted in step 11
  (likely fence line traced too high) is still open.

## Scope decisions (2026-08-28)
- This round: camera-order discovery, CV spike, backtester, stub alert modules. No `main.py`
  service loop, no Dockerfile/compose, no delivery outbox.
- Two gates: camera-order inference, then CV feasibility. Either failing changes the plan.
- **Gate 1 resolved without inference (2026-08-28):** the user decided camera order is just
  numerical/alphabetical by id (cam01, cam01a, cam01b, cam02, ... cam16) — no reason to expect the
  physical fence order differs. `config/cameras.yaml` `order` field set directly from that sort.
  `scripts/infer_camera_order.py` (steps 3-9 below) is no longer on the critical path; kept for
  optional later validation, not required before Phase 1.
- Labeling target: stratified ~300-500 clips, oversampling animal/incident.
- Zone editor: browser UI (no reliable WSL GUI assumed).
- Cameras record night only, ~18:00-06:00 (configurable). All footage is IR/greyscale.
- The source channel also carries camera health notifications. Backfill captures non-video
  messages too.
- A Telegram query bot for trustees/security is a later phase; its data requirements are
  captured now.

## Site facts (from the user)
- ~20 cameras, all on the fence line, mounted along it looking down the line.
- Open line, not a loop. Two distinct ends, plus 2 perimeter entry points (gates).
- Guard patrols out and back along the line nightly, using a flashlight.
- Camera order along the fence is unknown and must be inferred from data.
- Known events: ~1 crawl incident, 2 probes, 3 animal sightings. Multi-year history.
- Probes and animals were not observed as multi-camera sequences; only patrols were.

## Ground truth labels
Schema v5 (2026-08-29) splits this into two independent columns instead of one:
- `label`: what the event WAS -- `guard`, `animal`, `incident`, `resident`, `environment`,
  `neighbour`, or `unknown`. Nullable: a clip can have no class yet if only its `startup_state`
  is known so far. Shared across every clip in the same physical trigger (same embedded alert
  timestamp) -- labeling any one clip in an event fills in `label` for every still-unclassed
  sibling automatically (never overwrites an existing human call on a specific clip).
- `startup_state`: whether *this one clip's own content*, viewed alone (no future clip to
  compare against, matching what a live system would actually have), was usable --
  `clear` (subject visible), `blank` (nothing visible), or `duplicate` (frame-identical prefix
  of the paired later clip's start, confirmed automatically via pixel diff, never hand-picked).
  Says nothing about what the event was.
`environment` covers IR-attracted insects, rain streaks, wind-blown vegetation, and shadow
artifacts. Naming it separately makes false-page burden directly measurable instead of hiding it
inside `unknown`. `resident` (added 2026-08-29) covers an identified resident or other
authorized person moving on the interior side -- not a guard (no patrol signal expected) and not
a threat, so it shouldn't sit in `incident` just because a person is visible. Split out after an
audit found 6 `cam01b` clips explicitly noted as residents sitting in `incident`, which would
have been counted as false negatives (or, worse, trained a classifier to treat ordinary resident
movement as an incident) had they stayed there. `neighbour` (added 2026-09-07) covers a benign
person seen OUTSIDE the fence who isn't a resident or an intruder -- typically a neighbouring
property's own worker, visible near/through the perimeter. Distinct from `resident` (interior
side, identified) and from `incident` (a real security concern); see cam14/3495, the first
confirmed example. Migrated with `scripts/migrate_add_neighbour_label.py` (widened the `labels`
CHECK constraint only, `schema_version` stayed 5, same no-op-on-existing-rows pattern as the
`resident` migration).

Why split: v4 had a single `label` column, and a short pre-alert clip auto-labeled `startup` (or
hand-labeled `startup_clear`/`startup_blank`) silently discarded the event's real class whenever
the paired, decisive clip hadn't been reviewed yet -- found via a live-db audit: 123 startup-family
rows, 107 with their partner still unlabeled, contributing nothing to ground truth. See
`docs/handoff.md` for the migration record.

## Crawl / shape policy
Bounding-box h/w for a crawling person (~0.4-0.7) overlaps large animals (~0.5-1.2). Classical CV
cannot separate them, and a 3-second clip is too short for gait periodicity. Therefore:
- Outside motion alerts on its own merit.
- Shape and trajectory may only escalate to `outside_priority`. Nothing about blob shape may
  downgrade or suppress an outside alert.
- Escalation triggers: upright aspect, low-and-slow deliberate movement, dwell/lingering,
  fence-line crossing, repeated re-triggers on one camera.
- Accepted cost: some animal sightings page as priority. At ~3 animal events per year, preferable
  to missing a crawler.

## Sequence work (downgraded)
Probe-sequence detection is retrospective analysis only, not a live escalation trigger — the
evidence for multi-camera probe signatures is thin (n=2, and only events that were noticed). Its
real value is discovering outside clusters in history that nobody flagged at the time. Patrol
sequences remain load-bearing, because they drive camera ordering and patrol analytics.

## Night-only consequences
- Single threshold profile; no day/night split, no colour features. A derived `time_of_day` tag
  (`src.features.time_of_day`, from the clip timestamp vs the operating window) rides along in the
  spike CSV as a descriptive column only — dusk-lit clips like the 18:04/18:21 SAST animal
  sightings look different, and it's worth being able to see that during analysis without
  branching the thresholds on it.
- IR insect blobs near the lens (large, bright, fast, out of focus, erratic) are expected to be
  the largest false-positive source, ahead of the flashlight. Low edge density plus high centroid
  jitter are the intended discriminators.
- Silence outside the operating window is normal and never a camera fault.

## Geometry model
Per camera: `fence` (ordered normalised polyline), `outside` (`left`/`right`), `depth_cutoff`
(row beyond which sides are unresolvable → forced `ambiguous`), `ignore` (optional polygons for
chronic false triggers). One polyline cannot leave gaps or overlaps the way two hand-drawn
polygons can, and halves the drawing work. Side assignment uses the fraction of blob foreground
pixels past the line, so straddling blobs degrade gracefully.

Camera mounts can shift over time (knocked, re-aimed, re-mounted), which moves the fence line in
frame. A single fixed polyline per camera can't survive that, so `fence`/`outside`/`depth_cutoff`
need to become a dated history (e.g. a list of `{effective_from, fence, outside, depth_cutoff}`
entries per camera) rather than one static value, with lookups picking the entry whose
`effective_from` is the latest one at or before a clip's timestamp. Not needed for the Phase 0c
spike (hand-picked, single snapshot in time), but required before Phase 2's zone editor and
live classification are trusted long-term.

## Perspective consequences
- Aspect ratio is depth-invariant but distinguishes posture, not species — escalation signal only.
- Blob area is depth-dependent — a noise floor scaling with image row, never an absolute
  person/animal cutoff.
- Confidence degrades toward the vanishing point; `depth_cutoff` makes that explicit.
- A flashlight beam aimed down or across the fence lands outside the fence easily. Saturation, low
  solidity, and centroid jitter are the discriminators, and the spike must try to falsify this.

## What the backtester actually optimises
Known positives are tiny (~6 events). Fitting thresholds to that would be noise-fitting. The
fail-safe policy is fixed by rule, and the objective is to minimise false pages across the
thousands of guard and environment clips while keeping that policy intact. Rare-class metrics
ship with raw support counts and are indicative only.

## Steps

### Phase 0: Discovery and feasibility (standalone scripts; gates everything)

**0a — Metadata-only backfill**
1. Telethon pass over full channel history capturing metadata only — no video downloads. Records
   `(message_id, timestamp, camera_id, message_type, raw_caption)` plus health notifications.
   Fast and re-runnable.
2. Derive the camera roster from observed caption formats, with per-camera activity volumes,
   first/last seen, and nightly hour distribution. Flags caption shapes the parser did not
   recognise.

**0b — Camera order inference (gate 1) — [resolved without inference, see Scope decisions]**

Steps 3-9 below describe the originally-planned inference approach. Superseded 2026-08-28: the
user set `order` directly as the numerical/alphabetical sort of camera ids, so none of this ran.
Left in place only in case the assumption ever needs checking against real transition data.

3. Segment clips into patrol passes: consecutive clips with inter-clip gaps under a threshold,
   keeping passes touching ≥3 distinct cameras. No classification needed — a burst across several
   cameras is structurally a patrol.
4. Build a symmetric transition matrix over camera pairs: transition counts plus median and IQR
   of Δt.
5. Detect and collapse co-located cameras (Δt ≈ 0, high co-trigger count) into single nodes
   before ordering.
6. Produce a candidate order by linear spectral seriation — normalised Laplacian, sort by the
   Fiedler vector (`numpy.linalg.eigh`; no new dependency). Linear, not circular, since the fence
   is an open line — but also test the transition matrix for loop structure to confirm the data
   agrees.
7. Refine by minimising direction reversals: under a candidate order each patrol pass should
   decompose into few long monotonic runs. Optimise the ordering with 2-opt against total
   reversal count. Out-and-back passes naturally yield a palindrome with a detectable turn point,
   so clean passes constrain the order directly.
8. Validate before trusting:
   - Split-half stability — infer independently on first and second half of history; compare
     orders.
   - Transit-time consistency — adjacent pairs should show tight, repeatable Δt.
   - Entry-point check — cameras where passes disproportionately start and end should match the
     2 known gates. Independent confirmation.
   - Reference-frame contact sheet — render one frame per camera in inferred order; adjacent
     cameras on a shared line usually share landmarks, so the scene should flow.
   - Cross-check spectral vs reversal-minimised orders; agreement between two independent methods
     is the confidence signal.
9. Emit a proposed order with per-adjacency confidence, plus any low-confidence or unplaced
   cameras, for the user to confirm or edit. Never silently authoritative. Median transit times
   are retained as approximate spacing for later coverage-gap analysis.

**0c — CV feasibility spike (gate 2) — RETIRED as a blocking pass/fail gate, 2026-09-08.**

Gate 2 was designed as a one-time decision on a 150-clip snapshot, made before proceeding to
Phase 1. In practice it never worked that way: Phase 1's foundation (steps 15-19) shipped without
it, and eleven further sessions of empirical classifier work (`docs/handoff.md` sessions #1-#11)
happened anyway, on a corpus that grew from ~150 to 678+ labelled clips, because a single pass/fail
call on stale data kept getting revisited and kept blocking nothing real. The user's own framing:
"gate 2 keeps getting in the way" — the actual question was never "does classical CV separate
these classes at all" (answered yes, repeatedly, in practice) but "is the classifier good enough to
ship," which is not a one-time decision, it's an ongoing measurement. See "Ship readiness" below,
which replaces this gate. Step 14's original finding is kept as historical record, not a live
blocker — do not treat its numbers (150 clips, 0.22 precision) as current; they predate almost
everything in `docs/handoff.md`.
10. [done] Download a clip subset for 2-3 cameras chosen using 0a activity stats and known incident
    locations, ideally including the camera that caught the crawl. Span night, storm, animal,
    guard, and intruder examples. `scripts/download_clips.py` does this against rows already in
    `clips` (from 0a): filter by camera and an optional timestamp window, cap per camera, fetch
    each message's media via Telethon to `data/history/{camera_id}/{message_id}.mp4` through a
    `.part` file and atomic rename, and record the path back into `clips.file_path`. Deliberately
    small-scale and non-resumable — the full-history equivalent is Phase 1 step 21.
    Cameras chosen: `cam06` (crawl incident), `cam08` (probe + dusk animal), `cam05` (dusk
    animal). 142 clips downloaded across them, spanning 2023-2026 so ordinary nights are
    represented alongside the incident windows.
11. [done] Hand-enter fence polylines for those cameras in YAML (2-4 points each). No editor yet.
    Reading coordinates off a zoomed screenshot by eye proved unreliable on low-contrast IR
    frames — three separate attempts tracked a bright diagonal cable rather than the fence. What
    worked: hand the user a clean upscaled reference frame, have them trace the fence in red in
    any paint tool, then colour-threshold the red pixels back out to recover the polyline
    exactly. `outside` is plain screen position (point x vs. the fence's x at the same row) --
    point order doesn't matter, but still verify with `src.zones.side_name` against a
    known-exterior point after any edit as a sanity check.

    2026-08-30 correction: `side_name` originally used a direction-of-travel ("which hand")
    convention, which inverts left/right for any fence traced top-to-bottom relative to naive
    screen reading -- confusing enough that a real config bug was suspected where there wasn't
    one. Replaced with a plain point-x vs. fence-x-at-that-row comparison; verified against real
    footage that this alone (no `cameras.yaml` edits) fixes cam06/cam09/cam10's known crawling
    intruders reading as "inside" while leaving the cam08 control correct. Guard/environment
    clips on those same cameras, and cam01b's residents, still split inconsistently under the
    corrected math -- that's a separate, still-open issue (fence line traced too high, most
    likely -- see docs/handoff.md).
12. [done] Hand-label ~150 clips. 165 labelled across all 18 cameras (target was
    ~150): guard 111, environment 22, startup 11, unknown 9, incident 8, animal 4.
13. [done] Extract candidate features to a flat CSV: outside pixel fraction, aspect ratio, solidity,
    saturation, green-light ratio (site-specific: the guard's flashlight reads as a
    distinct green — added beyond the original feature list), green-light flicker (std
    deviation of the whole-frame green ratio across the clip -- the guard sweeps the beam
    rather than holding it still, catching guard clips the single-frame reading misses),
    row-normalised area, edge density, path length, jitter, persistence. `scripts/spike.py`
    run per-camera into `data/reports/spike_{camera_id}.csv`. The per-clip detector uses
    median-background subtraction after dropping IR-warmup frames, not consecutive-frame
    differencing — the latter locks onto the whole-frame brightness swing these cameras
    produce in the first ~2 seconds and hid a porcupine entirely on cam15/15454.
14. [done, pending user sign-off] Decision gate — does any threshold combination separate guard and
    environment from animal/incident at usable precision? Explicitly count flashlight and IR-insect
    clips landing outside the fence. Written finding: `docs/gate2_separability_finding.md`.
    Headline: the strongest result is `green_light_ratio` combined with `green_light_flicker`
    (temporal variance, catching a swept beam a single frame would miss) as a guard
    identifier — fires on 61% of guard clips, 2% of non-guard, and 0/12 animal+incident.
    Shape/motion features separate
    low-crawling subjects from upright ones in the right direction (animal/incident aspect ~0.75 vs
    guard/environment ~1.07), and excluding green-lit clips cuts the false-positive rate from 40% to
    24% at unchanged recall. Absolute precision is still only 0.22 at 0.75 recall against an 8.3%
    base rate, on just 12 positive clips — usable as an escalation signal on top of outside
    geometry (the design this plan specifies), not as a standalone classifier. Both predicted
    failure modes confirmed: 58% of guard clips on fenced cameras land outside (64% of those
    carrying a light signal), and 55% of environment clips match the IR-insect signature. If
    separation fails, stop and revisit scope (earlier ML, IR/brightness handling, multi-frame
    reference modelling) rather than proceeding — flagged for the user to confirm the
    escalation-only reading counts as "usable", and that more animal/incident labels would move
    confidence more than further feature work.

### Phase 1: Foundation
15. [done] Scaffold the `uv` Python 3.12 project: locked dependencies, ruff, pytest,
    `.env.example`, structured JSON logging, console entry points. Depends on both Phase 0 gates.
    Locked deps/ruff/pytest/`.env.example` existed from the Phase 0 spike already. Structured
    logging added 2026-08-30 as `src/logging_setup.py::configure_logging` (a `JsonFormatter` on
    the root logger) — replaces the old per-call-site `logger.info(json.dumps({...}))` pattern in
    `scripts/meta_backfill.py`/`scripts/download_clips.py` with `logger.info(event, extra={...})`.
    "Console entry points" scoped down to the existing `main()` + `if __name__ == "__main__"`
    pattern per script: this project is `uv init --app --no-package` (flat `src/`, no
    `[build-system]`), so real installed `[project.scripts]` entries aren't meaningful without
    converting to a packaged layout, which would contradict the deliberate flat-layout choice.
16. [done] Validated config loading for env, `config/thresholds.yaml`, `config/cameras.yaml`;
    normalised coordinates; motion-extraction settings split from classification thresholds.
    `cameras.yaml` carries the confirmed fence order and the operating window. Built during the
    Phase 0 spike (`src/config.py`) to support it; carries over as-is.
17. [done] SQLite via `CREATE TABLE IF NOT EXISTS` plus a `schema_version` — no migration
    framework by default (export labels to JSONL, delete db, reimport, is the documented
    fallback). In
    practice, once `clips` holds real backfilled volume, a targeted in-place table rebuild
    (`scripts/migrate_schema_v5.py`: rename, recreate from the current schema, copy+transform,
    bump `schema_version`, keep the old table rather than dropping it) is the proportionate
    approach for a change scoped to one table — done for the v4->v5 label split, 2026-08-29.
    Tables: `clips`, `labels`, `blob_tracks` (feature cache), `system_events`, `backtest_runs`,
    `backtest_results`. WAL, busy timeout, foreign keys, narrow repository functions. Index
    `(timestamp)` and `(camera_id, timestamp)`. Built during the Phase 0 spike (`src/db.py`);
    carries over as-is.
18. [done] Label export/import to JSONL as the durability guarantee, so the database can be
    rebuilt without losing hand-entered ground truth. Built during the Phase 0 spike
    (`src/db.py::export_labels_jsonl`/`import_labels_jsonl`); carries over as-is.
19. [done, 2026-08-30] Promote the Phase 0 scripts into supported modules (`src/backfill.py`,
    `src/sequence.py`) now that their approach is validated. `src/sequence.py` was already
    promoted (built directly there during 0b). `src/backfill.py` is new: `run_backfill`/
    `extract_primitives`/`RawMessage` moved out of `scripts/meta_backfill.py` verbatim, which is
    now a thin wrapper around it (Telethon client wiring + `main()` only), matching the
    established pattern in `scripts/infer_camera_order.py` (algorithm in `src/`, script keeps only
    I/O glue). `tests/test_meta_backfill.py` renamed to `tests/test_backfill.py` to match.


## Ship readiness (replaces Gate 2 pass/fail, 2026-09-08)

Not a phase with its own numbered steps — a standing bar every later phase's work is checked
against, re-measured as the corpus and rules change, instead of a one-time decision made on a
snapshot. "Good enough to ship" means all of the following hold, measured on the *current* labelled
corpus (re-run, don't trust a quoted number — every session in `docs/handoff.md` has learned this
the hard way at least once):

1. **Hard constraint, non-negotiable: no labelled incident event stops alerting.**
   `scripts/check_incident_regression.py` passes (currently 5/5 events). Any change that would
   break this needs the user's explicit, logged sign-off on the specific clip lost (as already
   practiced — see `docs/handoff.md`'s `cam10/7632` correction for what that looks like done
   right, and what it looks like done wrong the first time).
2. **No labelled animal event is silently dropped.** Same standard as incidents, one notch looser
   only because animal sightings are a lower safety stake than a human intruder — still requires
   an explicit, documented exception, not a silent regression buried in a threshold sweep.
3. **Guard and environment leak into the alert channel is small enough to actually review.** This
   is a real number, not a vibe, but it is the user's number to set, not an agent's to assume —
   pick a concrete target (e.g. "under N alert-channel clips per night on average" or a leak
   percentage) and record it here once chosen. Until then, track the trend via
   `scripts/backtest.py` + `scripts/rank_candidates.py` (`docs/handoff.md` has the running series:
   29→14→6 environment-leak clips, 39→26 guard-leak clips, across sessions #5-#7) rather than
   gating on an unset number.
4. **No known, unaddressed corpus-wide detector bug materially distorts the classifier's own
   inputs.** Currently open: the `best_contour`/tracker selection issue (`docs/handoff.md` sessions
   #10-#11) — measured net-positive on the labelled sample but not yet safe as a default; see that
   section for the specific gate before flipping it on.
5. **The full test suite and the incident regression check both stay green** —
   `uv run ruff check . --fix && uv run pytest -q` and `scripts/check_incident_regression.py`, the
   same two commands this repo has run before every commit since session #1.

Phase 2 work (below) is judged against this bar, not against a resurrected Gate 2.

### Phase 2: Corpus and CV
20. Camera-ID parsing hardened against every caption shape observed in 0a, plus
    health-notification formats, with an explicit unknown-camera result.
21. Resumable full backfill capturing every message type: clips to
    `data/history/{camera_id}/{message_id}.mp4` via `.part` then atomic rename; health/status text
    into `system_events`. Dedupe, flood-wait/reconnect handling, deterministic JSONL manifest, one
    structured log line per message. Parallel with step 22.
22. `motion.py` emits zone-independent blob tracks — per-frame contours, centroids, areas, aspect,
    solidity, saturation, edge density, jitter — depending only on video plus MOG2/morphology
    settings. Cached under one hash of the motion config plus an `EXTRACTOR_VERSION` constant.
23. `zones.py` applies polyline, side assignment, depth cutoff, and ignore regions on top of
    cached tracks. Cheap and re-runnable, so polygon iteration never triggers re-extraction.
24. Browser zone editor on localhost: extract a reference frame, draw the polyline, click the far
    side, drag the depth cutoff, add ignore polygons, write `cameras.yaml` atomically. Saves a new
    dated fence version rather than overwriting, so a re-aimed camera keeps its old geometry valid
    for clips predating the change (see Geometry model).
25. `classify.py` rule engine producing `guard_side`, `outside_alert`, `outside_priority`,
    `ambiguous`, each with reason codes and contributing thresholds. Implements fail-safe
    escalation. Unknown camera, missing fence line, above depth cutoff, or undecodable video
    resolve to `ambiguous`. Typed interface for a future ML resolver, unimplemented.

## Phase 2 refactor brief (2026-09-09) — organise what already exists before building what doesn't

The empirical work since Phase 0c built real equivalents of most of steps 20-25 under different
names, ad hoc, while gate 2 sat undecided (now retired — see Ship readiness above). **This is
reorganisation, not new detection logic.** The single rule for all of it: behaviour must not
change. `uv run ruff check . --fix && uv run pytest -q` and `scripts/check_incident_regression.py`
must stay green throughout, and any extracted module's output should be verified byte-identical
against the pre-refactor code on the full labelled corpus before being trusted, same discipline as
every `detect_clip`-level change already in `docs/handoff.md`'s history. Adding new tests during
extraction is expected (a bare function with no dedicated test suite of its own is exactly the kind
of thing this refactor should leave better documented and better covered than it found it) — that
is not "changing behaviour."

**Steps 25 and 28 are DONE (2026-09-09)** — see `docs/phase2_refactor_execution_plan.md` for the
full step-by-step record and `docs/handoff.md`'s session #15 for the short version. Two decisions
worth recording here since they affect anyone touching `src/classify.py` or `src/db.py` next:

- **`backtest_results.predicted_class` was WIDENED, not mapped.** `src.classify.classify()` emits
  eight `*_candidate`/`no_motion`/`unclassified` categories; the CHECK constraint only accepted the
  planned four-class routing vocabulary (`guard_side`/`outside_alert`/`outside_priority`/
  `ambiguous`) this step's own plan text specifies below, which was never built. Rather than map the
  eight measured categories onto those four, schema v6 (`scripts/migrate_schema_v6.py`) widened the
  constraint to accept both. Mapping would have encoded an alerting policy nobody has validated;
  choosing one is gated on Ship readiness criterion #3 above, still unset. If that vocabulary is
  ever actually built, the mapping decision still needs to happen then, not retroactively now.
- **The `classification` thresholds now live in `config/thresholds.yaml`**, wired via
  `src.config.ClassificationThresholds` — user-directed 2026-09-09, beyond this brief's original
  scope but landed alongside it since it's what step 25's "classify.py never hardcodes a number"
  text below always meant. The file's OLD `classification` section was stale and unconsumed before
  this (values like `outside_pixel_fraction.alert_min: 0.5` against the real, measured `0.6`) — the
  16 real values were taken FROM `src.classify`'s code, not reconciled toward the old file. `motion`
  is untouched and still unconsumed (see step 22 below); `motion_fingerprint()` was verified
  unchanged by the classification rewrite, so nothing about step 22's future cache key was disturbed.

**Per-step reality check, most valuable to least:**

| step | plan says | what actually exists | verdict |
| --- | --- | --- | --- |
| 25 `classify.py` | typed rule engine, reason codes per decision | **DONE.** `src/classify.py`: `classify_detailed()` returns `ClassificationResult(category, reason, contributing)`, `classify()` is a one-line wrapper. All 16 thresholds load from `config/thresholds.yaml` via `src.config.ClassificationThresholds`, no hardcoded constants left. | Landed 2026-09-09. Verified byte-identical to the pre-refactor code on the full 678-clip labelled corpus at every intermediate step (extraction, config wiring, reason codes) — see the execution plan doc for the per-step diffs. |
| 28 `backtester.py` | immutable runs: timestamp, config snapshot/hash, git revision | **DONE.** `src/backtester.py::record_run()` wires the existing `create_run`/`finish_run`/`add_backtest_result` functions in; wired into `scripts/backtest.py::main()` by default (`--no-record` to skip). | Landed 2026-09-09, needed schema v6 (see above) since the existing CHECK constraint couldn't accept a real `classify()` output. |
| 23 `zones.py` | polyline/side/depth/ignore on top of cached tracks | `src/zones.py` exists, in daily use | **Already done**, just not literally "on top of cached tracks" (see step 22) since nothing is cached yet. No action needed unless step 22 changes its inputs. |
| 22 `motion.py` | cached zone-independent blob tracks, `EXTRACTOR_VERSION` | **Operational cache shipped 2026-09-13.** `MotionThresholds` strictly loads all 22 real settings; `src/motion.py` owns `EXTRACTOR_VERSION`, complete cache identity, persistence through `blob_tracks`, and the complete detector/tracking implementation; backtests use the cache by default with `--no-cache` available. On 678 labelled clips, uncached/cold/warm CSVs were byte-identical and warm runtime was 3.7s versus ~6m. | **Correctness/performance goal met; optional cache refinement remains.** The cached JSON payload is the final feature map, keyed by video, resolved zone, reference background, daylight and motion config. It is safe but zone-aware: a geometry edit recomputes extraction. Compact JSON-safe geometry and metric observation DTOs now replay those feature groups exactly, but are not persisted. Do not cache `ClipDetection` wholesale; it contains large NumPy imagery. |
| 24 browser zone editor | draw polyline/depth/ignore in a browser, atomic dated write | **Nothing** — every fence trace to date is "hand the user an upscaled frame, they draw in a paint tool, colour-threshold it back out" (`README.md` section 4) | Real, standalone feature work, not a refactor of anything existing. Lowest priority of the five unless retracing cameras becomes frequent enough that the manual process is the bottleneck — it currently isn't (18 cameras, dated-history schema already handles remounts). |
| 20/21 backfill/parsing | hardened camera-ID parsing, resumable full backfill | `scripts/meta_backfill.py`/`scripts/download_clips.py`, 17,000+ clips backfilled without incident | Probably fine in practice but **never verified against the original spec** (every caption shape, health-notification formats, flood-wait/reconnect, dedup). Worth a real audit before assuming done, but not urgent — nothing has broken.

**Suggested next engineering tracks:** the step-22 raw-track architectural refinement above,
step 24's zone editor, or the separate 20/21 backfill audit. The operational caching risk is now
closed: its identity covers every current extraction input and corpus output was verified exactly.

### Phase 3: Labels, backtester, retrospective sequences
26. `scripts/label.py`: resumable, prints the clip path by default, optional configured player,
    single-key labelling, skip and correction support.
27. `src/sequence.py` extended for analysis: patrol detection (inside progression along the
    confirmed order, cadence, coverage gaps) as a supported feature, plus a retrospective probe
    report listing outside clusters across adjacent cameras in history for manual review. The
    latter is exploratory output, not a live trigger.
28. `src/backtester.py`: immutable runs recording timestamp, config snapshot/hash, and Git
    revision when available. Reuses cached blob tracks, re-derives geometry and classification,
    never contacts Telegram.
29. Reports to `data/reports/{run_id}/` as console text plus JSON: cross-tab; outside alert
    precision/recall/F1 (animal+incident positive, guard+environment negative); priority metrics;
    projected false pages per night as the headline number; ambiguous/review-workload rate;
    per-camera breakdown; misclassified IDs with paths, reasons, and full feature vectors.
30. Raw support counts beside every metric.
31. Global thresholds only. Per-camera overrides only where a report demonstrably requires one —
    sparse per-camera labels would otherwise overfit.

### Phase 4: Alert modules (no service)
32. `telegram_alert.py` and `ntfy_alert.py` as pure tested functions with mocked transports, plus
    a manual send-test command verifying credentials, channel ID, ntfy reachability and optional
    bearer token.
33. README covering Telethon first-auth, camera-order discovery, spike findings, backfill,
    polyline drawing, labelling, backtesting, threshold iteration, and the manual alert test.
34. Commit per passing phase on a feature branch; no auto-merge.

### Phase 5: Trustee query bot (after calibration)
35. [done] Role-based access from a Telegram user-ID allowlist: `trustee` (full), `security`
    (restricted). Enforced server-side in every handler — hiding a menu button is not access
    control. Unknown IDs refused and logged.
36. [done] Commands: `/about`, `/tonight`, `/health`, `/animals`, `/map` for both roles; `/patrols`
    trustee only (passes per night, times, cameras covered, coverage gaps).
37. [done] Camera health surfaces parsed `system_events` primarily, with schedule-aware silence detection
    as a backstop for a camera that dies without notifying. Silence outside the operating window
    is never a fault.
38. [done] The map is an approximate schematic — numbered positions along a line, no accurate
    coordinates or coverage arcs — because anything retrievable in Telegram can be forwarded.
39. [done] Analytics carry confidence caveats; patrol counts lean on sequence detection rather than
    single-clip classification.

## Sensitivity note
Patrol-compliance analytics profile identifiable individuals' movements and working patterns.
Restricting that view from the security company is a sound control, but guards remain data
subjects (POPIA applies in South Africa). Worth a brief word with trustees on retention and
access before the bot ships.

## Candidate refinements (not yet built, revisit with evidence)

> **Status note (2026-09-07):** several items in this section are now partly or wholly built —
> the dual fence lines shipped as `fence_bottom`, the fence-height/metric calibration shipped as
> `src/ground_calibration.py`, and the flashlight-vs-subject idea produced a working feature
> (`post_flash_red_shift`). Read the 2026-09-07 checkpoint at the top of this file and
> `docs/handoff.md` section #5 before treating anything below as unbuilt.

- **Dual fence lines (inside/outside band, not just one side-assignment line)**: on some cameras the
  guard passes close beside/under the fence, close enough that a single polyline's side test could
  misclassify that proximity as outside. Idea (2026-08-28, user): draw two polylines bounding the
  fence structure itself (its inside and outside edge) instead of one, so a blob has to be genuinely
  beyond the outside line — not just past the single line — to score as a crossing. Deliberately not
  building this now: doubles the per-camera drawing/config work for a problem we haven't observed
  yet. Revisit only if backtesting turns up false positives that trace back to near-fence guard
  proximity rather than an actual crossing.
- **Per-camera normal-footprint model as a classification input**: idea (2026-08-28, user) —
  build a per-camera positional heatmap from historical `guard`-labelled clips (where the blob
  centroid/track usually falls), then flag detections that fall well outside that usual footprint
  as an anomaly signal, even on the inside. Distinct from the `activity heatmap` already listed
  under Explicitly deferred (Phase 5), which is a trustee-bot analytics *display* feature — this
  one would feed `classify.py`/escalation directly, as a spatial prior rather than a dashboard.
  Caveats the user flagged: flashlight glare/lighting changes could look like positional shift
  without being one, so the footprint would need to be built from track geometry (centroid path),
  not brightness; and it needs enough labelled `guard` clips per camera before a "usual" shape
  means anything, which the labelling pass (Phase 3, step 26) hasn't produced yet. Revisit once
  labelling gives a real per-camera sample size, not before.
- **Annotated-video Telegram delivery option**: attach a copy of the alert clip with the motion
  blob/bounding box and the fence polyline burned in per frame, as an optional alternative or
  addition to the plain-text alert (2026-08-28, user request). Depends on `motion.py`/`classify.py`
  (Phase 2/3, not yet built) to produce the per-frame contours to draw, and a `send_video`-capable
  path in `telegram_alert.py` (currently text-only, `send_message` only). Natural fit as a Phase 4
  addition once the base classifier and plain-text alerts are proven — no upstream pipeline to
  draw from yet, so there's nothing to wire it into today.
- **Fence-base reference line for real-world scale, height, and speed (2026-09-04, user).**
  **Phases 1-3 IMPLEMENTED 2026-09-05** (schema/parsing, dual-line render, geometry helpers); see
  `docs/handoff.md`'s 2026-09-05 session for the full writeup. Summary of what shipped:
  Idea: draw a SECOND polyline tracing the fence's own base/ground line (in addition to the
  existing top-of-fence `fence` polyline), alongside the known real-world fence height
  (~2m). Together these two lines calibrate pixels-to-metres AT EVERY ROW of the frame (the
  fence's own apparent height in pixels at a given row is a built-in perspective ruler for that
  camera, no separate camera calibration needed). This converts several already-pixel-relative
  features into real-world units for the first time:
  - **Real height/size**: a tracked blob's height in pixels, divided by the local pixel-per-metre
    ratio at its own row, gives an actual height estimate — a ~1.7m subject reads very differently
    from a ~0.3m one, regardless of how close to the camera either is. `row_normalised_area`
    already partially addresses depth-dependence (see Perspective consequences above) but stays in
    pixel units; this would be the first true metric-scale feature.
  - **Real speed**: combined with timestamps between frames, calibrated real-world displacement
    per second — distinct from the existing `normalised_speed` (already in `src/features.py`,
    scaled by the blob's OWN width in body-lengths, not metres). A real speed in m/s is
    comparable ACROSS cameras and across distance-from-camera in a way `normalised_speed` isn't.
  - **Dwell time / lingering vs. moving-at-pace vs. small-and-fast**: the user's own framing --
    lingering in one spot may mean one thing (a guard posted/waiting, or an animal browsing),
    moving steadily at a walking pace may indicate a guard patrol, small-and-fast may indicate an
    animal. `persistence`/`longest_detection_run` (dwell proxies) and calibrated real speed/size
    together would let a rule distinguish these for the first time with an actual physical basis,
    not just relative pixel statistics.
  - Not built: needs a second hand-traced polyline per camera (same drawing-effort cost as the
    "dual fence lines" idea above, though this one calibrates depth rather than gating inside/
    outside), a `src/zones.py`/`CameraZone` schema change, and a validated pixels-per-metre
    formula checked against known real subject sizes (e.g. the confirmed animal sightings in
    `/memories/repo/incident-findings.md`) before trusting any derived speed/height number.
  - **Shipped 2026-09-05**: `CameraZone.fence_bottom`/`fence_height_m` (defaults `None`/2.0m, all
    25 existing test-suite `CameraZone(...)` constructions untouched). Inside/outside
    classification (`classify_zone`/`track_crosses_fence`/`median_fence_distance`, via a new
    `effective_fence()`) now prefers `fence_bottom` when a camera has one, else falls back to
    `fence` unchanged -- cam06 is the only camera with a bottom line so far (traced over
    `data/history/cam06/8467.mp4`), every other camera's behaviour is byte-identical to before.
    New pure geometry helpers in `src/zones.py`: `fence_separation_at_y`, `pixels_per_metre_at_y`,
    `estimated_height_m`/`estimated_width_m`/`estimated_speed_mps`, `subject_base_y` (feet-row,
    not centroid), `is_grounded_at_fence` (returns `None`/uncalibrated above the base line's own
    traced range -- the bird-sitting-on-the-fence confound). `render_debug.py`/`visualize_zone.py`
    draw both lines (top rail unchanged colour, base line in orange) when both exist. **Sanity
    check (phase 5) against all 41 cam06 guard clips**: per-clip median `estimated_height_m`
    clusters at 1.66m median (genuine bg-diff frames only), right in the 1.6-1.9m target band;
    clips with >=10 genuine frames cluster even tighter (1.5-1.7m on 6 of 10 such clips) --
    calibration passes, ruler is trustworthy for cam06. **Discovery (phase 4) for
    `in_fence_band`/`entered_band_from_outside`**: checked against the only calibrated camera's
    incident/animal ground truth (cam06 has 1 incident clip, 21520, and 0 animal clips) --
    `entered_band_from_outside` is `False` (the crawl incident's whole track reads "outside" and
    never enters the band region at his row; the confirmed real-world crawl-along-the-fence
    subject never crossed into it). n=1 is far too small to certify anything either way. **Neither
    the calibration functions nor the fence-band functions are wired into
    `scripts/backtest.py::classify`** -- this was deliberately left as additive infrastructure,
    same as `scenery_motion_fraction`'s own history; a real rule needs a labelled sample size and
    threshold sweep this session didn't have time for (only 1 camera is calibrated at all).
  - **All 18 cameras now have `fence_bottom` traced (2026-09-05, later same day) -- phases 4/5
    re-run on the full corpus, both with material findings:**
    - **Phase 4, full incident+animal set (20/21 detectable)**: only one nominal
      `entered_band_from_outside` hit (`cam09/21522`), confirmed via `render_debug.py` to be the
      already-documented wire-lock tracking artifact on that exact clip, not a real subject.
      Real count of genuine band-crossings: **0/20**. Still nowhere near enough to build a rule.
    - **Phase 5, 15 cameras with guard clips (up to 20/camera)**: does **NOT generalise beyond
      cam06**. Per-camera median height ranges 0.89m-6.93m, with a single-frame outlier as high
      as 18.73m. Root cause diagnosed: `pixels_per_metre_at_y` is correctly direction-monotonic
      (more px/m near the camera, fewer far away) but the far end is very coarse (as low as
      ~2.5 px/m) -- ordinary bbox-height measurement noise on a far subject gets amplified into
      an absurd real-world height once divided by that tiny scale. cam06's own 41-clip
      validation happened to work because enough of its guard sightings are close-range;
      that isn't true corpus-wide. **Verdict: do not wire the metric (height/width/speed)
      helpers into scoring as-is.** Credible fix if revisited: a minimum px-per-metre floor
      (same shape as `MIN_FENCE_SEPARATION_PX`) so far-field estimates return `None`
      ("uncalibrated") instead of a wrong number -- not attempted, needs its own
      threshold-on-labelled-data pass. The inside/outside classification improvement from
      `fence_bottom` itself is unaffected by this finding -- only the metric-scale features are.
    - **`fence_pickets` angle-correction shipped 2026-09-05 (user insight).** A second likely
      contributor to the phase-5 blowup, distinct from the far-field-noise root cause above:
      pairing `fence`/`fence_bottom` at the SAME image row (as `fence_separation_at_y` always
      did) silently assumes a picket renders perfectly vertical in frame -- false whenever the
      camera looks down the fence at an angle. New `CameraZone.fence_pickets` (plural -- see
      below) lets `fence_separation_at_y` project along the interpolated picket direction to
      find where it actually crosses the top rail, instead of assuming the crossing is directly
      above. Falls back to the old same-row method when no picket is traced -- every camera is
      byte-identical until one is added. The upright-taper cap prefers `fence_vanishing_point()`
      (where `fence`/`fence_bottom` actually converge if extended) over `depth_cutoff` when a
      valid convergence exists.
    - **cam06 validated with 3 real pickets, and the model upgraded as a result (2026-09-05).**
      Traced 3 pickets at different depths on the same camera (over
      `data/history/cam06/8468.mp4`, replacing an earlier mistrace on a bad/outlier clip,
      8467.mp4). Measured tilt ratio: 0.28 (closest) -> 0.20 -> 0.196 (farthest) -- **does NOT
      taper linearly with depth** (drops sharply near the camera, nearly flattens further out).
      `CameraZone.fence_picket` (singular) was therefore generalised to `fence_pickets` (plural):
      each traced picket contributes one (row, tilt ratio) sample, piecewise-linearly
      interpolated between neighbours, only falling back to clamping/linear-taper-to-cap beyond
      the nearest/farthest real sample. Not yet re-run against the phase-5 height-clustering
      check with this corrected model -- next step if this is picked up again.
- **Flashlight-vs-subject side divergence as a guard-specific signal (2026-09-04, user + this
  session's re-derivation)**: the user's insight — "guards are on the inside, they can cross the
  line since they are visible through the fence if they walk close, they shine their flashlight
  across the fence" — is not a hypothesis, it's confirmed in this session's re-derivation of the
  animal/incident rule (see the commit re-deriving `scripts.backtest.classify`): `outside_pixel_
  fraction` alone scores AUC 0.700 for animal+incident vs guard+environment, but is NOT safe as a
  standalone rule (guard false-fire 40%+ at every threshold tried) precisely because guards
  routinely register as "outside" too. The idea, not yet built: track the fence-side of the
  FLASHLIGHT'S OWN light mask separately from the subject's own tracked-box side (both already
  computable per frame from existing zone geometry + `green_light_mask`). A guard's own body
  staying inside while their flashlight beam crosses outside would read as "subject-inside,
  flashlight-outside" — a divergence a real intruder essentially never produces (per the same
  insight: "incidents are on the wrong side of the fence with mostly no flashlight"). Two new
  candidate features: `flashlight_outside_fraction` (time flashlight-mask pixels read outside,
  independent of the tracked subject) and `subject_flashlight_side_divergence` (how often the two
  disagree). Not measured yet — would need per-frame side-classification of the light mask wired
  into `extract_clip_features`, distinct from the existing whole-box `flashlight_bbox_overlap`.
- **"What else" — further candidate features raised or implied 2026-09-04, none built or
  measured yet:**
  - Time-of-day/schedule pattern: guards patrol on a roughly regular schedule; a track's local
    time relative to known typical guard-activity hours could be its own weak prior (distinct
    from `is_daylight`/`is_twilight`, which are about ambient light, not patrol scheduling).
  - Sequential cross-camera appearance at walking pace: `src/storm_events.py`'s neighbor/order-
    adjacency machinery (built this session for the unrelated storm-corroboration problem) is
    structurally the same shape needed to detect "a subject reaches an ADJACENT camera within a
    plausible walking-speed time gap" as a POSITIVE guard-patrol signal — the mirror image of
    today's finding that such correlation is a guard signature, not a storm one. Currently only
    used as a environment/storm filter; repurposing it as a guard-identification feature is
    unexplored.
  - Dog/pet detection for `resident`: user asked "can we look for a dog also". No dog-labelled
    clips exist in the corpus yet (checked: `labels.notes` has no confirmed dog sighting on
    record) — needs real examples before any shape/speed-based rule could be derived, same
    discipline as everywhere else in this repo. The already-planned real-height/speed calibration
    above would help here too (a dog is both smaller and faster than a person at the same
    distance from camera).
  - `resident`: `is_daylight`/`is_twilight` (already built, 2026-09-04) plus the fence-side
    geometry (staying inside, never crossing) are the two existing ingredients closest to a
    `resident_candidate` rule — not yet assembled into one, and `resident` currently has NO rule
    of its own in `scripts.backtest.classify` at all (falls through to `unclassified`).
  - **Daytime de-escalation for a known-benign-but-unidentified person (2026-09-05, user;
    label added 2026-09-07).** cam14/3495 is a neighbour's worker: not a resident (not
    identified/authorized on this property), not a guard, borderline "almost an intruder" by
    appearance alone, but genuinely not a threat. `VALID_LABELS` gained `neighbour` for exactly
    this case (a benign person outside the fence, not resident/guard/threat) -- cam14/3495
    relabelled from `unknown` accordingly. Still NO `classify()` rule of its own: only n=1
    confirmed example exists corpus-wide as of this write (a discovery sweep for more candidates
    found several real `animal`/`environment` clips but zero further `neighbour` ones -- see
    repo memory). A real rule needs more examples first, same discipline as everywhere else in
    this file; the actual alert-priority/de-escalation concept itself (Phase 4 territory) is a
    separate, still-undesigned question even once a detection rule exists.



## Explicitly deferred
`main.py` live loop, listener queueing, delivery outbox and crash recovery, Dockerfile/compose and
ARM64 build, live clip retention policy, run-comparison CLI, migration framework, per-camera
threshold sets, YOLO/ONNX, activity heatmap, trend analytics, probe-sequence live escalation.

## Relevant files
- `scripts/meta_backfill.py`, `scripts/infer_camera_order.py`, `scripts/download_clips.py`,
  `scripts/spike.py`, `scripts/label.py`
- `src/config.py`, `src/db.py`, `src/motion.py`, `src/zones.py`, `src/classify.py`,
  `src/backfill.py`, `src/sequence.py`, `src/backtester.py`
- `src/telegram_alert.py`, `src/ntfy_alert.py` — functions only this round
- `src/bot.py` — Phase 5
- `config/cameras.yaml` (dated fence polyline history, outside side, depth cutoff, ignore regions,
  confirmed order, operating window), `config/thresholds.yaml`
- `tests/`, `README.md`, `pyproject.toml`, `uv.lock`

## Verification
1. `uv lock --check`, `uv run ruff check .`, `uv run pytest`.
2. Gate 1: inferred camera order passes split-half stability, transit-time consistency, and the
   entry-point check; spectral and reversal-minimised orders agree; user confirms against their
   mental map.
3. Order-inference unit tests on synthetic patrol timelines with known ground-truth order,
   including injected noise clips, a missing camera, and a co-located pair.
4. Gate 2: written separability finding on ~150 labelled clips, including flashlight and
   IR-insect false-positive counts.
5. Unit tests: polyline side assignment with straddling blobs, depth cutoff, ignore regions,
   config rejection, caption and health-notification fixtures, classifier boundaries,
   zero-denominator metrics.
6. Fail-safe test: a synthetic wide, low, slow outside blob must never classify as `guard_side`
   or drop below `outside_alert`.
7. Synthetic-video test — blob tracks byte-identical across runs.
8. Cache test — polyline or threshold edits trigger zero OpenCV extractions; MOG2 or
   `EXTRACTOR_VERSION` changes invalidate.
9. Backfill twice on a small range — no duplicate rows or manifest entries, interrupted `.part`
   recovers, health notifications land in `system_events`.
10. Polyline round-trips at a different playback resolution.
11. Label export/import survives a database delete and rebuild.
12. Manual alert test — one Telegram message, one ntfy notification.
13. Bot authorisation — `security` refused `/patrols` at handler level, unlisted IDs refused every
    command.

## Standing caveats
- Calibration numbers are evidence, not a security guarantee; this supplements the existing guard
  arrangement rather than replacing it.
- Crawling humans and large animals are not reliably separable with classical CV. Mitigated by
  policy, not thresholds.
- Inferred camera order is a hypothesis until validated and user-confirmed; it must never be
  silently authoritative.
- Storage: full multi-year history is plausibly 10-40 GB; metadata-first backfill defers that
  decision.

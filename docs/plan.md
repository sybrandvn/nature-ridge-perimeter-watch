# Plan: Nature Ridge Perimeter Watch

Discover the camera layout from message metadata, validate the classical-CV hypothesis on a small
labelled sample, then build the backfill → label → backtest calibration loop. Live service, Docker,
and delivery-reliability machinery are deferred until thresholds are proven. Alert modules ship as
tested, manually-invokable functions only.

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
- `label`: what the event WAS -- `guard`, `animal`, `incident`, `resident`, `environment`, or
  `unknown`. Nullable: a clip can have no class yet if only its `startup_state` is known so far.
  Shared across every clip in the same physical trigger (same embedded alert timestamp) --
  labeling any one clip in an event fills in `label` for every still-unclassed sibling
  automatically (never overwrites an existing human call on a specific clip).
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
movement as an incident) had they stayed there.

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

**0c — CV feasibility spike (gate 2)**
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
35. Role-based access from a Telegram user-ID allowlist: `trustee` (full), `security`
    (restricted). Enforced server-side in every handler — hiding a menu button is not access
    control. Unknown IDs refused and logged.
36. Commands: `/about`, `/tonight`, `/health`, `/animals`, `/map` for both roles; `/patrols`
    trustee only (passes per night, times, cameras covered, coverage gaps).
37. Camera health surfaces parsed `system_events` primarily, with schedule-aware silence detection
    as a backstop for a camera that dies without notifying. Silence outside the operating window
    is never a fault.
38. The map is an approximate schematic — numbered positions along a line, no accurate
    coordinates or coverage arcs — because anything retrievable in Telegram can be forwarded.
39. Analytics carry confidence caveats; patrol counts lean on sequence detection rather than
    single-clip classification.

## Sensitivity note
Patrol-compliance analytics profile identifiable individuals' movements and working patterns.
Restricting that view from the security company is a sound control, but guards remain data
subjects (POPIA applies in South Africa). Worth a brief word with trustees on retention and
access before the bot ships.

## Candidate refinements (not yet built, revisit with evidence)
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

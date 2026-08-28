# Plan: Nature Ridge Perimeter Watch

Discover the camera layout from message metadata, validate the classical-CV hypothesis on a small
labelled sample, then build the backfill → label → backtest calibration loop. Live service, Docker,
and delivery-reliability machinery are deferred until thresholds are proven. Alert modules ship as
tested, manually-invokable functions only.

## Scope decisions (2026-08-28)
- This round: camera-order discovery, CV spike, backtester, stub alert modules. No `main.py`
  service loop, no Dockerfile/compose, no delivery outbox.
- Two gates: camera-order inference, then CV feasibility. Either failing changes the plan.
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
`guard`, `animal`, `incident`, `environment`, `unknown`.
`environment` covers IR-attracted insects, rain streaks, wind-blown vegetation, and shadow
artifacts. Naming it separately makes false-page burden directly measurable instead of hiding it
inside `unknown`.

## Crawl / shape policy
Bounding-box h/w for a crawling person (~0.4-0.7) overlaps large animals (~0.5-1.2). Classical CV
cannot separate them, and a 3-second clip is too short for gait periodicity. Therefore:
- Far-side motion alerts on its own merit.
- Shape and trajectory may only escalate to `far_side_priority`. Nothing about blob shape may
  downgrade or suppress a far-side alert.
- Escalation triggers: upright aspect, low-and-slow deliberate movement, dwell/lingering,
  fence-line crossing, repeated re-triggers on one camera.
- Accepted cost: some animal sightings page as priority. At ~3 animal events per year, preferable
  to missing a crawler.

## Sequence work (downgraded)
Probe-sequence detection is retrospective analysis only, not a live escalation trigger — the
evidence for multi-camera probe signatures is thin (n=2, and only events that were noticed). Its
real value is discovering far-side clusters in history that nobody flagged at the time. Patrol
sequences remain load-bearing, because they drive camera ordering and patrol analytics.

## Night-only consequences
- Single threshold profile; no day/night split, no colour features.
- IR insect blobs near the lens (large, bright, fast, out of focus, erratic) are expected to be
  the largest false-positive source, ahead of the flashlight. Low edge density plus high centroid
  jitter are the intended discriminators.
- Silence outside the operating window is normal and never a camera fault.

## Geometry model
Per camera: `fence` (ordered normalised polyline), `far_side` (`left`/`right`), `depth_cutoff`
(row beyond which sides are unresolvable → forced `ambiguous`), `ignore` (optional polygons for
chronic false triggers). One polyline cannot leave gaps or overlaps the way two hand-drawn
polygons can, and halves the drawing work. Side assignment uses the fraction of blob foreground
pixels past the line, so straddling blobs degrade gracefully.

## Perspective consequences
- Aspect ratio is depth-invariant but distinguishes posture, not species — escalation signal only.
- Blob area is depth-dependent — a noise floor scaling with image row, never an absolute
  person/animal cutoff.
- Confidence degrades toward the vanishing point; `depth_cutoff` makes that explicit.
- A flashlight beam aimed down or across the fence lands on the far side easily. Saturation, low
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

**0b — Camera order inference (gate 1)**
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
10. Download a clip subset for 2-3 cameras chosen using 0a activity stats and known incident
    locations, ideally including the camera that caught the crawl. Span night, storm, animal,
    guard, and intruder examples.
11. Hand-enter fence polylines for those cameras in YAML (2-4 points each). No editor yet.
12. Hand-label ~150 clips.
13. Extract candidate features to a flat CSV: far-side pixel fraction, aspect ratio, solidity,
    saturation, row-normalised area, edge density, path length, jitter, persistence.
14. Decision gate — does any threshold combination separate guard and environment from
    animal/incident at usable precision? Explicitly count flashlight and IR-insect clips landing
    on the far side. If separation fails, stop and revisit scope (earlier ML, IR/brightness
    handling, multi-frame reference modelling) rather than proceeding.

### Phase 1: Foundation
15. Scaffold the `uv` Python 3.12 project: locked dependencies, ruff, pytest, `.env.example`,
    structured JSON logging, console entry points. Depends on both Phase 0 gates.
16. Validated config loading for env, `config/thresholds.yaml`, `config/cameras.yaml`; normalised
    coordinates; motion-extraction settings split from classification thresholds. `cameras.yaml`
    carries the confirmed fence order and the operating window.
17. SQLite via `CREATE TABLE IF NOT EXISTS` plus a `schema_version` — no migration framework.
    Tables: `clips`, `labels`, `blob_tracks` (feature cache), `system_events`, `backtest_runs`,
    `backtest_results`. WAL, busy timeout, foreign keys, narrow repository functions. Index
    `(timestamp)` and `(camera_id, timestamp)`.
18. Label export/import to JSONL as the durability guarantee, so the database can be rebuilt
    without losing hand-entered ground truth.
19. Promote the Phase 0 scripts into supported modules (`src/backfill.py`, `src/sequence.py`) now
    that their approach is validated.

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
    side, drag the depth cutoff, add ignore polygons, write `cameras.yaml` atomically.
25. `classify.py` rule engine producing `guard_side`, `far_side_alert`, `far_side_priority`,
    `ambiguous`, each with reason codes and contributing thresholds. Implements fail-safe
    escalation. Unknown camera, missing fence line, above depth cutoff, or undecodable video
    resolve to `ambiguous`. Typed interface for a future ML resolver, unimplemented.

### Phase 3: Labels, backtester, retrospective sequences
26. `scripts/label.py`: resumable, prints the clip path by default, optional configured player,
    single-key labelling, skip and correction support.
27. `src/sequence.py` extended for analysis: patrol detection (near-side progression along the
    confirmed order, cadence, coverage gaps) as a supported feature, plus a retrospective probe
    report listing far-side clusters across adjacent cameras in history for manual review. The
    latter is exploratory output, not a live trigger.
28. `src/backtester.py`: immutable runs recording timestamp, config snapshot/hash, and Git
    revision when available. Reuses cached blob tracks, re-derives geometry and classification,
    never contacts Telegram.
29. Reports to `data/reports/{run_id}/` as console text plus JSON: cross-tab; far-side alert
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

## Explicitly deferred
`main.py` live loop, listener queueing, delivery outbox and crash recovery, Dockerfile/compose and
ARM64 build, live clip retention policy, run-comparison CLI, migration framework, per-camera
threshold sets, YOLO/ONNX, activity heatmap, trend analytics, probe-sequence live escalation.

## Relevant files
- `scripts/meta_backfill.py`, `scripts/infer_camera_order.py`, `scripts/spike.py`,
  `scripts/label.py`
- `src/config.py`, `src/db.py`, `src/motion.py`, `src/zones.py`, `src/classify.py`,
  `src/backfill.py`, `src/sequence.py`, `src/backtester.py`
- `src/telegram_alert.py`, `src/ntfy_alert.py` — functions only this round
- `src/bot.py` — Phase 5
- `config/cameras.yaml` (fence polyline, far side, depth cutoff, ignore regions, confirmed order,
  operating window), `config/thresholds.yaml`
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
6. Fail-safe test: a synthetic wide, low, slow far-side blob must never classify as `guard_side`
   or drop below `far_side_alert`.
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

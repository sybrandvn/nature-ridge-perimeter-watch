# Detection & measurement capability map

Written 2026-09-09, prompted by the user's own framing while investigating the residual guard
leak: *"isolate all methods available and see what each of them's purpose is and what they can
measure and how they work for the whole clip, per frame, per object, and see if we can piece
together the full story of each clip."* This is that inventory. It is a map of what already
exists, not a proposal — nothing here is new code. Cross-reference `src/classify.py`'s module
docstring for which of these are actually wired into a decision today (most are not).

**How to read the scope column**: every measurement in this codebase operates at one of five
distinct levels, and conflating them is the single most common source of a surprising number.

| scope | means | example |
| --- | --- | --- |
| **whole-clip** | one number for the entire clip | `outside_pixel_fraction` (from the single best frame) |
| **per-frame** | computed for every frame, usually then reduced (median/peak/fraction) | `blob_count`, `motion_pixel_fraction` |
| **single-track** | about "the" one tracked target only (`track_contour`'s pick) | `path_length`, `jitter`, `green_light_ratio` |
| **per-object** | about every persistently-identified blob (`multi_tracks`), not just the one tracked target | `multi_object_*` (2026-09-09) |
| **warmup** | the dropped/pre-scored IR-flare frames, not the scored ones | `warmup_flashlight_ratio`, `warmup_outside_fraction`, `warmup_dynamic_*` |

---

## 0. The cross-cutting trap: "genuine" vs "recovered" evidence

Before anything else, the one thing that explains a large fraction of surprising readings across
*many* different features: a frame's detection can come from four different mechanisms, and most
per-track features only trust the first.

| provenance | what it means | `FrameDetection` flag |
| --- | --- | --- |
| genuine | a real background-diff blob this frame | none set |
| recovered | no bg-diff blob, but a template match against the last known appearance found one | `recovered=True` |
| filled by reverse | the forward pass found nothing here; a backward scan did | `filled_by_reverse=True` |
| filled by anchor | matched against one fixed best-frame exemplar, not a refreshed template | `filled_by_anchor=True` |

`metric_features_from_observations` (ground-plane/metric features), `path_length`/`jitter`/`persistence`
(single-track motion features), and the new `multi_object_*` features (2026-09-09) all **only
count genuine, bg-diffed frames** — a track that's mostly recovered/appearance-matched (typically a
small, fast, intermittently-detected subject — an animal is the common case) reads as having little
or no evidence for these, even though the single "best contour" pipeline (which DOES use
recovered/filled boxes) correctly tracked it the whole time. This is not a bug in any one feature —
it's a deliberate, repeated design choice ("a hallucinated continuation box says nothing about the
real subject") — but it means comparing a recovery-blind feature against a recovery-tolerant one
(e.g. `multi_object_dominant_outside_fraction` vs `outside_pixel_fraction`) needs this kept in mind,
found and fixed once already this session (`_multi_object_outside_features`'s
`fallback_outside_fraction` parameter) and worth checking again before trusting any new
per-track/per-object comparison.

---

## 1. How a "blob" becomes "the subject" (single-track pipeline)

The core tracking mechanism `outside_pixel_fraction`, `path_length`, `green_light_ratio`, and
almost every other single-track feature ultimately depends on.

| function | file | what it does |
| --- | --- | --- |
| `detect_clip` | `scripts/spike.py` | Orchestrates everything below into one `ClipDetection`. Background subtraction (MOG2-style diff against a per-clip median), per-frame contour finding, warmup/flare drop, then the tracking passes. |
| `track_contour` | `scripts/spike.py` | Given this frame's candidate contours and the last known box, picks the one that *continues the existing track* (IoU first, then nearest-centroid within a size-scaled margin, rejecting an implausible area jump) rather than always re-picking the largest blob. This is what keeps two co-occurring subjects' identities from swapping. |
| `reacquire_by_template` | `scripts/spike.py` | For a frame where background-diff found *nothing* (subject blends into background brightness-wise): template-matches the last known crop against a search window, velocity-shifted toward the direction of travel. Produces a `recovered` frame. |
| `_run_track_pass` | `scripts/spike.py` | Runs the forward and backward `track_contour` + `reacquire_by_template` combination over a clip, refreshing the match template on every real detection (lets it follow a subject that changes appearance — but also lets a partly-wrong box teach the next match to look for a partly-wrong thing). |
| `_anchor_trace` | `scripts/spike.py` | A *different* recovery mechanism: sweeps outward from one fixed best-evidenced frame, matching every other frame against that single never-updated exemplar. Can't drift the way a refreshed template can, but will lock onto an unrelated static background patch forever once the real subject has left frame — capped by `max_streak`. |
| `track_multiple_objects` | `scripts/spike.py` | The **per-object** tracker (`multi_tracks`): every distinct blob gets its own persistent id across frames, with `merged_ids` for the "two subjects sharing one blob" case. **Only sees genuine per-frame blobs — no recovery/reacquire/anchor mechanism feeds it.** Runs unconditionally for every clip, purely for diagnostic overlays until 2026-09-09; still not read by `classify()`. |

**What this means practically**: "the" tracked subject for most features is a single lineage that
can legitimately be genuine detection → recovered → anchor-traced within one clip — a mix of real
evidence and inference. `multi_tracks` sees only the genuine slice of that same clip. Neither is
"more correct" in general; they answer different questions (`outside_pixel_fraction` — where was the
subject, best guess, across the whole tracking apparatus; `multi_object_*` — where did something
*provably* move, with real identity, ignoring anything inferred).

---

## 2. Fence-side / geometry (`src/zones.py`)

Everything here answers "which side of the fence" or "how far from it," given a point or a track.
All of it depends on `effective_fence` — `fence_bottom` (base line) when a camera has one traced,
else `fence` (top rail) — this is the fence_bottom-era switch discussed with the user 2026-09-09.

| function | scope | what it measures | wired into `classify()`? |
| --- | --- | --- | --- |
| `classify_zone` | per-point | inside / outside / ignored / ambiguous for one point | indirectly (everything below calls it) |
| `outside_pixel_fraction` | whole-clip, single-track | fraction of the single best frame's contour points reading outside | **yes** — the main outside-geometry gate |
| `zone_classifiable_fraction` | whole-clip, single-track | fraction of points that got a real verdict at all (disambiguates a 0.0 "all inside" from "nothing classifiable") | **yes** — guards the inside-only fallback rule |
| `outside_frame_fraction` | per-frame, single-track | fraction of scored frames where the tracked contour read majority-outside | no — reporting only |
| `track_crosses_fence` | whole-clip, single-track | did the centroid track appear on both sides at any point | no — computed as `fence_crossed`, reported only, not read by `classify()` |
| `median_fence_distance` | whole-clip, single-track | **median** absolute distance from the fence across the whole centroid track — robust to a single frame's noise by construction | **yes** — the second half of the outside-geometry gate |
| `in_fence_band` / `entered_band_from_outside` | per-point / whole-clip | whether a point sits between the two fence lines (top+base), and whether a track entered that band from outside | no — discovery-stage only, explicitly not wired (`src/zones.py` comment) |
| `fence_separation_at_y`, `pixels_per_metre_at_y`, `estimated_height_m`/`width_m`/`speed_mps` | per-row / per-object | the older picket-interpolation metric ruler | superseded by `src.ground_calibration` for cameras that have opted in (see §8) — the interpolation isn't projectively consistent (caps at ~4.7m with a discontinuity, measured on cam06) |

**The multi-object gap, precisely**: none of these take a `multi_tracks` object as input except via
the new `_multi_object_outside_features` (§ below) — every other geometry feature here is
single-track scoped.

---

## 3. Colour / flashlight (`src/features.py`) — four distinct signals, none per-object

This is worth its own section because the user's own question ("do we always know when an object
is a flashlight") exposed that these four are easy to conflate and none answer "is object N a
flashlight."

| function / feature | scope | what it actually checks | wired? | known blind spot |
| --- | --- | --- | --- | --- |
| `green_light_ratio` | single-track, single frame | hue/saturation inside the **tracked contour's shape** on its best frame | **yes** — first rule in `classify()` | reads 0.0 if the tracker is following something *other* than the light (confirmed on cam07/11174 — beam on the fence, tracked box 40% of frame away on illuminated bushes) |
| `whole_frame_green_ratio` | whole-frame, per-frame peak | hue/saturation across the **entire frame**, not contour-restricted | no — diagnostic only, thin margin against real animal clips (1.8x) | a small beam patch in a large frame can still read low even when visually obvious |
| `warmup_flashlight_ratio` | warmup, per dropped-frame | same hue check, scored on the frames dropped for IR flare | **yes** — second guard rule | only covers frames *before* the tracked target ever appears |
| `flashlight_bbox_overlap` → `flashlight_subject_fraction` | single-track, per tracked **bbox** (not contour shape) | fraction of the tracked box's own rectangular pixels reading flashlight-hue, aggregated across tracked frames | no — reporting only; also drives `scripts/render_debug.py`'s "FLASHLIGHT" overlay label | bbox-shaped (coarser than contour-shaped), and again single-track only |

**Answering the literal question**: no signal here, or anywhere else in this codebase, checks
whether an arbitrary `multi_tracks` object is a flashlight. All four are scoped to either the single
tracked target or the whole frame. Checking a specific `multi_object_*` dominant object for
flashlight-ness is unbuilt — `green_light_ratio`'s own signature (`frame_bgr, contour`) already
accepts *a* contour, so scoring each per-object candidate is a mechanically small extension, not new
detection work — same category as `_multi_object_outside_features` itself.

Also present, not flashlight-specific: `color_fraction`, `saturation_ratio`, `color_saturation_fraction`
— general colour-presence checks (used e.g. to split `animal_candidate` from `incident_candidate` by
daylight colour, not to detect a light source).

---

## 4. Shape / motion character (`src/features.py`, single-track unless noted)

| function | what it measures | wired? |
| --- | --- | --- |
| `aspect_ratio` | height/width of the best contour | reported; the old aspect-based rule was retired 2026-09-04 (premise inverted) |
| `solidity` | contour area / convex-hull area — how "filled-in" the shape is | **yes** — insect rule (`jitter>50 and solidity<0.85`, largely dead against the current tracker) |
| `edge_density` | Canny edge pixels inside the contour / area | reported only |
| `blob_white_fraction` | fraction of the contour reading blown-out white | **yes** — blinding-lens gate, shared with `is_blinding_foreground` |
| `path_length` / `jitter` | total centroid movement / how erratically it moves, over genuine frames | reported; `jitter` feeds the insect rule |
| `persistence` | genuinely-detected frames / total frames | reported |
| `longest_detection_run` | longest unbroken run of genuine detections | reported |
| `area_stability` | how much the contour's area varies frame to frame (lower = more subject-like, same convention as `height_consistency` below) | reported only |
| `normalised_speed` | centroid displacement in **body-widths** per frame (scale-invariant across distance, not across real units) | reported |
| `heading_change` | how much the direction of travel changes | reported |

---

## 5. Whole-frame / whole-clip aggregate motion

| feature | scope | what it measures | wired? |
| --- | --- | --- | --- |
| `motion_pixel_fraction` / `_median` | per-frame → peak/median | fraction of the whole frame that's "motion" this frame, no blob/side awareness at all | `_median` **yes** (sustained whole-frame motion gate); peak version tried and rejected (worst incident 0.123 vs guard/environment median 0.172 — not enough margin) |
| `blob_count` / `_median` | per-frame → peak/median | how many separate blobs passed the area gates this frame | **yes**, both — the environment/storm gate |
| `scenery_motion_fraction` | whole-clip | fraction of clip's motion area matching a **per-camera reference background** (§7) at the same coordinates — "this moves, and it's always there" | reported only (feeds the reference-background veto elsewhere, not a `classify()` feature directly) |
| `rectangular_black_white_balance` | per-frame contours → peak | scans filled rectangular contours for a clipped black+white sensor/decode signature, even if the contour did not become the main track | debug/reporting evidence; not safe alone, while the narrower terminal reverse-seed conjunction is wired |
| `global_camera_shift_score` | adjacent frames → peak | phase-correlation evidence for broad image translation after edge normalisation; detects wind-driven camera/fence movement but can also rise when a large nearby subject dominates the image | debug/reporting only; never an alert veto |
| `multi_object_outside_fraction_weighted` / `_dominant` / `_count` | per-object, whole-clip | area-weighted outside fraction across every persistent object; the largest-total-area object's own fraction; how many distinct objects | **no** — added 2026-09-09, reporting only, see this session's measurement notes in the commit log for why it isn't ready |

---

## 6. Warmup / IR-flare handling

The IR illuminator's gain step means the first several frames of most clips are undifferenceable
("flare") — dropped from the background model and from every per-frame feature above, but not
thrown away entirely.

| function | what it does |
| --- | --- |
| `flare_frames` / `flare_settle_index` | Per-clip, measures when the gain step actually settles (varies frame 1 to 21+), so the drop window is measured, not a fixed guess. |
| `photometric_match` / `apply_photometric_match` / `photometric_match_color` | Brightness+colour-cast correction, so a dropped warmup frame can be fairly diffed against the *settled* background instead of comparing pre-gain-step pixels to post-gain-step ones. |
| `_warmup_motion_features` → `warmup_outside_fraction` | Re-diffs corrected warmup frames against the settled background, tracks the largest changed region, and reads its fence side. The location signal remains reporting/ranker evidence because inside warmup presence overlaps positives. |
| `_warmup_motion_features` → `warmup_dynamic_frame_fraction`, `warmup_dynamic_outside_fraction` | Diffs adjacent photometrically corrected warmup frames, so a static branch merely lit differently from the settled background does not count as moving. When **no settled/scored motion exists**, strong persistent onside dynamics recover four labelled guards with no animal/incident category change; otherwise these remain evidence only and cannot override a later real subject. |
| `post_flash_red_shift` | Whole-clip colour-shift signal spanning warmup **and** scored frames — the flash itself is often inside the flare window. |
| `warmup_flashlight_ratio` | See §3 — hue check on the same corrected warmup frames. |

---

## 7. Reference background / scenery matching (`src/reference_bg.py`)

Answers "is this permanent scenery or a real, one-time subject" by comparing against OTHER clips
from the same camera, not this clip's own background (comparing against your own clip is circular —
measured zero separation, real subjects and known scenery both scored 0.87-1.00 against their own
clip's median).

| function | what it does |
| --- | --- |
| `clip_median_background` / `combine` | Per-clip median frame, then median-of-medians across many clips → one stable reference. |
| `bucket_clips` / `sample_evenly` | Buckets by camera, geometry era (a remount invalidates a reference, same as a fence retrace), lighting (day/night), and rough time of year. |
| `reference_for` | Picks the bucket matching a clip's camera/era/lighting, nearest in time; returns `None` (not a guess) when nothing matches — e.g. cam11 has only 2 clips total, never gets one. |
| `align` | Registers a clip's own background against the reference (handles small camera shifts within one era). |

---

## 8. Metric ground-plane calibration (`src/ground_calibration.py`, opt-in per camera)

Real-world units instead of pixel-space proxies — single-view metrology from the fence's own
geometry (top rail + base line give a horizon vanishing point; the traced pickets, vertical in the
world, give a second vanishing point; the two together fix focal length and, combined with
`fence_height_m`, absolute scale). Superseded the older per-row picket-interpolation ruler in
`src/zones.py`, which measurably caps around 4.7m with an unphysical discontinuity.

Feeds `metric_features_from_observations` (`src/scoring.py`), which is where the actual per-clip features
come from — **excludes recovered/reverse-filled frames**, same trap as §0:

| feature | what it measures | wired? |
| --- | --- | --- |
| `implausible_height_fraction` | how often the blob implies a physically-impossible real height | **yes** — environment physics gate |
| `off_plane_fraction` | how often the base point isn't on the ground plane at all | reported |
| `height_consistency` | coefficient of variation of raw implied height frame to frame — **lower = more subject-like** (a rigid body keeps its real height; rain/branch/flare doesn't) | reported only — a real, already-computed static-vs-artifact discriminator that isn't gating anything yet |
| `depth_progression` | net ground-plane distance travelled ÷ total distance travelled — near 1 for a subject changing range steadily, near 0 for something that doesn't translate in depth at all. **Deliberately independent of the flashlight being visible.** | reported only |
| `depth_range_m` | max − min ground-plane distance across the clip | reported |
| `subject_height_m` / `_width_m` / `_area_m2` / `metric_aspect` | median real-world size, scale-invariant (a human is ~1.6-1.9m regardless of range; `row_normalised_area`/`aspect_ratio` conflate near/far) | reported |
| `distance_median_m` / `speed_mps` | median range; real walking-speed comparison (~1.4 m/s) across subjects of different sizes | reported |

`uncalibrated=1.0` and every other key reads a bare 0.0 when the camera hasn't opted in or the
traced geometry is unusable — the `implausible_height_fraction` classify() rule explicitly checks
`uncalibrated==0.0` first for exactly this reason.

**This section directly matters for the residual guard leak investigated 2026-09-09**:
`height_consistency` and `depth_progression` are *already-built, already-measured* static-vs-moving
discriminators — closer to what the earlier "a subject translates, wind oscillates in place" idea
was reaching for than anything proposed fresh this session. Neither is currently checked against
the `multi_object_*` dominant object; both are single-track scoped and calibration-gated (works for
whichever cameras have opted in — not all 18).

---

## 9. Time of day (`src/features.py`)

`time_of_day`, `is_daylight`, `is_twilight`, `minutes_from_daylight_boundary`, `daylight_hint` — the
real sun-time signal (not derived from the image) threaded into every `classify()` call as
`features["is_daylight"]`. Used by the resident/guard daylight split — checked against
`color_fraction` and rejected, since a guard's own flashlight raises whole-frame saturation the same
way ambient daylight does.

---

## 10. Cross-camera / retrospective (not per-clip)

- **`src/storm_events.py`**`find_corroborated_events` — did a *neighbouring* camera also read
  `environment_candidate` within a short window. Validated 100% precision at the default settings,
  but explicitly a discovery/proposal tool, never a label — real wind affects more than one camera;
  a guard walking a stretch of cameras in sequence looks identical in shape.
- **`src/sequence.py`** — patrol-order inference across a whole night's clips (Phase 3 territory,
  not a per-clip signal at all — listed for completeness, not relevant to "the story of one clip").

---

## What this suggests, without proposing any specific change

Three real, already-existing gaps this map surfaces, beyond the one already being worked
(`multi_object_*`'s fallback fix):

1. **No feature checks flashlight-ness per `multi_tracks` object.** `green_light_ratio`'s own
   signature already takes an arbitrary contour — scoring each per-object candidate is a small
   extension of an existing primitive, not new detection work.
2. **`height_consistency`/`depth_progression` (§8) are unused static-vs-moving discriminators**
   that already exist and are already measured, for whichever cameras have metric calibration
   opted in. Cross-referencing them against a `multi_object_*` dominant object, on calibrated
   cameras, could test the "is this a real subject or a swaying/static artifact" question directly
   rather than inferring it from geometry alone.
3. **Every per-track feature (§0) is either genuine-only or best-guess-including-recovery, and
   nothing currently reports which regime a given clip's numbers came from.** A `genuine_frame_fraction`-
   style feature (persistence already reports something adjacent, but per-provenance breakdown
   doesn't exist) would make it possible to tell, from the CSV alone, whether a clip's `multi_object_*`
   reading rests on real evidence or the zero-evidence fallback — right now that requires re-running
   `detect_clip` by hand, as this session did for the visual-inspection pass.

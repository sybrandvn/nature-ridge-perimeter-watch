# Detection & classification improvement review

Written 2026-09-09, from a full read of `docs/plan.md`, `docs/handoff.md` (sessions #1-#15),
`docs/gate2_separability_finding.md`, `docs/detection_capability_map.md`, `src/classify.py`,
`src/features.py`, `scripts/spike.py`, `scripts/rank_candidates.py` and the live database.

## Implementation status (updated same day, later session)

Section 7's "do first" and most of "do next" items are DONE, on `feat/phase2-refactor`, in this
order (each committed and verified separately -- full test suite green throughout, 665 tests,
`scripts/check_incident_regression.py` unaffected):

1. **The `_multi_object_outside_features` bbox bug (1.1) -- FIXED.** Was reading corner-form
   `(x0,y0,x1,y1)` as `(x,y,w,h)`. Measured impact confirmed as described (105/141 sampled clips
   changed, 49/141 dominant-object flips). classify()'s output is byte-identical before/after on
   the full labelled corpus (not yet wired into any rule).
2. **`check_incident_regression.py` now asserts on animal events too (1.2) -- FIXED.** Surfaced
   three previously-invisible animal-event misses beyond the already-documented cam10/7632: real
   detection gaps on cam10/9405 (fence-distance lower bound), cam15's porcupine (never clears the
   outside-fraction majority), and cam10/17146 (a bird on the fence rail confuses the metric
   height gate). None fixed yet -- the script fails on them by design until each is fixed or
   individually signed off, per Ship readiness criterion #2. **This is new, real information --
   not a regression this session caused.**
3. **Real metrics in `scripts/backtest.py` (section 0) -- SHIPPED.** Confusion matrix,
   alert-channel precision/recall/F1, per-camera leak, written as a `.summary.json` alongside
   every backtest CSV. Also fixed `is_daylight` being silently dropped from recorded runs.
4. **The neighbour rule (2.2) -- SHIPPED as `neighbour_candidate`.** Measured for real (not
   estimated) at its actual position in the rule chain: 4/5 labelled neighbour clips (3/3 events),
   zero cost to incident/animal/resident/guard, 3/159 environment clips added to this low-priority
   channel. New DB schema v7 (`neighbour_candidate` needed a widened `predicted_class` CHECK
   constraint), migrated on the live db.
5. **Per-object flashlight scoring (section 3, stage 2) -- SHIPPED as
   `_multi_object_flashlight_features`, reporting-only.** Confirmed directly on cam07/11174 (the
   motivating clip): finds the real flashlight at ratio 0.43 even though the single-track
   `green_light_ratio` reads 0.0 there. Zero leakage into any protected class on the full labelled
   corpus, 63.5% guard hit rate on this signal alone. Not yet wired into `classify()` -- needs its
   own threshold/rule-position measurement, same discipline as everything else here.
6. **Ranker fixes (1.3) -- SHIPPED.** Fit-set sibling leakage in leave-one-out CV fixed (now fits
   on one row per physical event); `has_reference_background` restored as a model input.
   Per-camera standardisation (5.1) shipped as an **opt-in flag, off by default** --
   `--per-camera-standardize` -- after a real measurement came back negative on the only test
   affordable this session (labelled-subset-only per-camera stats; see the flag's own docstring
   for exactly what was and wasn't measured, and why that doesn't contradict the original
   full-corpus finding).
7. **Optical-flow direction coherence (4.1) -- BUILT AND MEASURED, hypothesis NOT confirmed.**
   `src.features.optical_flow_direction_coherence` + `flow_direction_coherence` in
   `extract_clip_features`, reporting-only. The specific mechanism proposed (single-pair internal
   coherence) reads BACKWARDS on the full labelled corpus (environment median 0.964, highest of
   any class) -- likely because 5fps is too coarse to see an oscillation within one frame pair.
   See section 4.1's own update and the feature's docstring for the honest writeup and the
   credible next variant (unmeasured). Kept as infrastructure, not a validated discriminator.

**Not done, deliberately, per this doc's own "needs a decision first" list:** the fence-distance
band change (§2.3), the what/where/when classifier restructure (§2.1, §3 stage 4), and
reference-background-as-primary-model (§4.4) -- all still waiting on the open questions at the
end of this document.

---

Every number below was measured in this session against the current DB, not quoted from an
earlier doc. Where a claim came from reading code, it was verified by running the code.

The user's stated priorities, which everything here is ranked against:

> Missing an incident is a critical failure. We want to be notified of animals and neighbours,
> and to a lesser extent residents. Noise is environment movement.

---

## 0. Headline: the bottleneck has moved back to recall, and nobody has re-measured since

The last recorded backtest run (`backtest_results`, 678 labelled clips, 2026-09-09) gives this
per-clip confusion. **Leakage — the thing sessions #5-#7 were built around — is now small. Recall
on three of the four classes the user actually wants is the problem.**

| true label | n | reaches the alert channel | correct | dominant failure |
| --- | --- | --- | --- | --- |
| incident | 10 | **7** | 7 | 2 suppressed as guard by the inside-only rule |
| animal | 37 | **15 (41%)** | 13 | 13 `unclassified`, 5 `no_motion` |
| neighbour | 5 | **0** | 0 | **no rule exists**; 4 `unclassified`, 1 `environment` |
| resident | 8 | n/a | 2 | 5 read `guard` |
| guard | 429 | 23 (5.4% leak) | 315 (73%) | — |
| environment | 159 | 5 (3.1% leak) | 117 (74%) | — |

Guard leak is 5.4% and environment leak 3.1%. Both are an order of magnitude better than the
19.5%/18.5% event-level figures session #6 was reacting to. Meanwhile animal recall is 41% and
neighbour recall is 0%.

**Nothing in the repo reports this table.** `scripts/backtest.py::main` writes a CSV and prints a
`Counter` of categories; it never compares `category` against the `label` column it faithfully
carries. Every precision/recall/AUC figure in this repo was computed by hand in a session and
survives only as prose in a docstring. That is the single cheapest thing to fix and it gates
honest judgement of everything else — see §2.4.

---

## 1. Three verified defects, in order of blast radius

### 1.1 CONFIRMED BUG: `_multi_object_outside_features` reads the bbox in the wrong convention

`scripts/spike.py:1882`.

```python
x, y, w, h = obj.bbox
point = ((x + w / 2.0) / frame_width, (y + h) / frame_height)
...
areas[obj.track_id] = areas.get(obj.track_id, 0.0) + float(w * h)
```

`TrackedObject.bbox` is **corner form** `(x0, y0, x1, y1)`. It is built by `_contour_bbox`
(`scripts/spike.py:174`), which returns `(x, y, x + w, y + h)`. The dead-reckoning path inside
`track_multiple_objects` (`scripts/spike.py:801`) unpacks it as `x0, y0, x1, y1`, and so does the
renderer (`scripts/render_debug.py:430`). This one function is the only place that reads it as
width/height.

Consequences, all silent:

- The sample x becomes `(x0 + x1/2)/W` instead of the midpoint.
- The sample y becomes `(y0 + y1)/H`, which exceeds 1.0 — a point below the frame — for any blob
  in the lower half. `classify_zone` does not bounds-check and `side_name` extrapolates the fence
  polyline, so an out-of-frame point still returns a confident `inside`/`outside`.
- `areas` becomes `x1 · y1`, a function of *position*, not size. `dominant_id` therefore selects
  the bottom-right-most object rather than the largest — the exact opposite of what the feature's
  own unit test says it is for.

**Measured impact** (141 randomly sampled labelled guard/environment clips, buggy vs corrected):

| | |
| --- | --- |
| clips whose weighted outside-fraction changes | **105 / 141 (74%)** |
| mean absolute change in the weighted fraction | 0.329 |
| clips where the dominant object's outside-fraction flips by more than 0.5 | **49 / 141 (35%)** |

The unit tests do not catch this because they construct `TrackedObject(bbox=(x, y, w, h))` by hand
(`tests/test_spike.py:1467,1484,1497,1514,1545`), with comments that only make sense under the
width/height reading (`# area 900 (30x30)`). The tests encode the bug.

**This invalidates commit `c1e55c4`'s measurement.** The `multi_object_*` features were measured,
found to make guard leak worse, and shelved — with this bug present. The whole per-object
direction, which is the foundation of everything the user is asking for, was assessed on
corrupted numbers.

Fix: one line, plus re-derive the two tests to corner form. Then re-run the measurement.

### 1.2 The incident regression check validates a configuration nobody runs, and ignores animals

`scripts/check_incident_regression.py` is the only automated guardrail in the repo, so both of
these matter.

**It does not pass a reference background.** Line 78 calls `extract_clip_features(path, zone,
daylight_hint=...)` with no `reference_background`. `scripts/backtest.py` and
`scripts/rank_candidates.py` both pass one. The guard therefore scores clips through a different
detector than either analysis tool and than what would ship.

**Its 21 non-incident fixture rows are decorative.** The fixture holds 10 incident rows (5 events),
11 animal rows (8 events) and 10 resident rows (5 events). The assertion at line 99 is
`event_label[event] == "incident" and all(suppressed_flags)`. The animal and resident events are
loaded, classified, printed — and never checked. **An animal regression passes silently today.**

Given "we want to be notified of animals", that is exactly the wrong asymmetry. Ship readiness
criterion #2 ("no labelled animal event is silently dropped") is not actually enforced by anything.

### 1.3 The ranker's cross-validation leaks siblings, and its feature count has drifted 2.5x

`scripts/rank_candidates.py`:

- **`NUMERIC_FEATURES` is 48, not the 19 the module docstring claims** (line 5). It is derived from
  `FEATURE_COLUMNS`, which has grown to 55.
- **The fit set is per clip** (line 369). `leave_one_out_auc` (line 381) holds out one clip while
  its near-identical `(Initial…)`/`(Stopped…)` sibling — carrying the identical event-level label —
  stays in training. The LOO AUC is optimistically biased by construction. The fix already exists
  in the same file: `prefer_longest_per_event` is called at line 435, on the *queue*, after the
  model is already fitted. Moving it above line 369 is a one-line change.
- **`blank`/`duplicate` startup clips are excluded from the queue but not from the fit.** Because
  labels propagate across an event, a blank precursor to an incident trains as a full-weight
  positive with near-degenerate features.
- **`has_reference_background` is excluded from the feature set** (line 70), so the ranker consumes
  `scenery_motion_fraction` — which reads 0.0 both for "no scenery found" and for "no reference
  existed" — with the disambiguating flag deliberately removed. `docs/handoff.md` line 594 contains
  the explicit instruction not to do this.
- 48 features against ~15-42 positives is past the overfitting knee, and session #6 already
  measured that dropping two feature groups improves LOO AUC by **+0.064**. Never implemented.
- `docs/handoff.md` line 596 states that backtest and rank_candidates "still pass no reference, so
  every recorded AUC remains comparable". **That is now stale — both pass one.** Every AUC recorded
  before that change is not comparable to one measured today.

### 1.4 Minor, but it bit this review

`scripts/backtest.py:56` puts `is_daylight` in `_NON_NUMERIC`, so it is dropped from
`REPORT_COLUMNS`, from the CSV, and from the `features_json` that `record_run` stores. Runtime
classification is correct (line 128 sets it immediately before line 129 classifies), so this is a
reporting gap, not a rule bug. But it means **no recorded backtest row can be replayed faithfully**
— any offline re-simulation silently treats every clip as night. I hit this and got a wrong answer
before catching it.

---

## 2. The structural recommendation: stop forcing one clip into one of eight boxes

This is the largest single change available and it makes three of the user's asks fall out for free.

### 2.1 The classes are three orthogonal axes, not one enum

`classify()` is a first-match-wins decision list whose order is explicitly load-bearing. Every new
rule interacts with every rule above it, which is why the docs repeatedly warn "measure a rule where
it would actually sit, not corpus-wide". With 55 features and 12 rules that has stopped scaling.

The seven labels are actually cells in a small grid:

| | inside the fence | outside the fence |
| --- | --- | --- |
| person **with a flashlight** | guard | guard (beam across the line) |
| person, night, no light | resident | **incident** |
| person, daylight, no light | resident | **neighbour** |
| non-person animal | animal | animal |
| vegetation / insect / lens artifact | environment | environment |

Three axes: **what** (person / animal / vegetation / light), **where** (inside / outside / at the
fence), **when** (day / night / twilight). Guard adds one attribute: a flashlight is present.

Two things follow immediately:

1. **`neighbour` and `resident` need no new detection work.** They are existing axes recombined.
   §2.2 shows a neighbour rule that works today on all 5 examples.
2. **The alerting policy becomes a separate layer from the evidence.** Right now "guard is tested
   first" is simultaneously a statement about evidence and a statement about priority, and it is
   why 2 of 10 incidents are suppressed as guard. Under a factored model the classifier says
   "person, outside, no flashlight" and a policy table decides that pages someone.

This is also what unblocks Ship readiness criterion #3 (the unset leak budget) — a factored score
lets you set an alert threshold without re-deriving every rule's position in a chain.

### 2.2 A neighbour rule that works right now

Measured this session on the labelled corpus, applied only to clips the current chain does *not*
already route to the alert channel:

> `outside_pixel_fraction > 0.6` **and** metric `subject_height_m` in **[0.9, 2.2] m** (calibrated
> cameras only) **and** `is_daylight`

| class | fires on |
| --- | --- |
| neighbour | **5 / 5** |
| incident | 0 / 10 |
| animal | 0 / 37 |
| resident | 0 / 8 |
| guard | 1 / 429 |
| environment | 15 / 159 |

Zero cost to any protected class. The 15 environment false-fires land in a new low-priority
`neighbour_candidate` channel, which is exactly the notification tier the user described.

**The reason this is cheap is that daylight is rare here**: 304 of 16,887 clips (1.8%) are real
daylight, about two a week across the corpus span. A daylight-gated channel is small by
construction. Twilight adds 1,319 clips (7.8%) and widening to daylight-or-twilight costs 4 more
guard and 3 more environment fires while still catching 5/5 — a reasonable dial.

`subject_height_m` is the load-bearing part and it is currently **reported only, never used by any
rule**. On the population that reaches the outside-geometry branch it separates well:

| class | subject_height_m p10 | p50 | p90 |
| --- | --- | --- | --- |
| animal | 0.10 | 0.30 | **0.69** |
| neighbour | **1.02** | 1.26 | 1.46 |
| guard | 0.22 | 1.09 | 2.09 |

Important caveat, measured: height does **not** separate real subjects from vegetation
(environment p50 0.93, spanning the same range as guard). It is a *species/size* axis, good for
routing animal-vs-person. It is not a real-vs-noise axis and must not be used as one.

### 2.3 The fence-distance band is now the dominant animal-recall killer

`median_fence_distance` must lie in (0.10, 0.40). Both bounds were fitted when environment leak was
29 clips. It is now 5. The trade has moved and nobody has revisited it.

Swept through the real rule chain, per clip:

| band | animal alerts | incident | guard leak | environment leak |
| --- | --- | --- | --- | --- |
| **0.10 – 0.40 (current)** | **15 / 37** | 7/10 | 23 | 5 |
| 0.05 – 0.40 | 19 / 37 | 7/10 | 27 | 8 |
| 0.00 – 0.40 | 20 / 37 | 7/10 | 33 | 8 |
| 0.10 – 0.50 | 19 / 37 | 7/10 | 29 | 12 |
| 0.00 – 0.50 | 24 / 37 | 7/10 | 39 | 15 |
| 0.00 – 1.00 | 26 / 37 | 7/10 | 40 | 21 |

Removing the band entirely recovers 11 animal clips for 17 more guard and 16 more environment
false alerts. Incident recall is unaffected at every setting.

Reading the individual misses: **11 of the 16** non-alerting animal clips are excluded by this band
alone, and by nothing else — five below the lower bound (an animal standing *at* the fence:
cam05/9690 at 0.036, cam06/18274 at 0.061, cam05/9691 at 0.067, cam06/18273 at 0.079, cam10/9405 at
0.097) and six above the upper (cam05/18791 at 0.405, which misses by 0.005; cam12/9721, 10081,
10082, 10147 at 0.48-0.52; cam10/7632 at 0.504). That per-clip count matches the sweep exactly:
lifting the band moves animal alerts 15 → 26.

The lower bound is also what kills 4 of the 5 neighbours (they stand at the fence, 0.045-0.097).

**The band is a proxy** for "is this a compact real subject rather than a big bush out in the
field". That job should be done directly. Until it is, the band is buying 5 environment
suppressions at the price of 11 animals — which is backwards against the stated priorities. This is
a decision for the user, not an agent: see the open questions at the end.

---

## 3. The object-linking design, made concrete

The user's own framing was right and is the correct target architecture:

> link up objects and tag them as the same in post, flashlight should always be separately tracked
> as flashlight and leaves room for tracking the main object, env noise should be able to be
> identified

Today the pipeline is: many blobs → **one** track → whole-clip scalars → one category. The tracker
picks by largest area, and `green_light_ratio` samples inside whatever won that vote. That single
choice causes several separately-documented failures (cam07/11174's flashlight reading 0.000
because the tracker was on an illuminated bush 40% of the frame away; cam07/13842's bird invisible
because a track was already active elsewhere).

`prefer_flashlight_candidate` tries to fix this by making the tracker *follow the light*. That is
the wrong direction — it makes the box the beam rather than the person, and its four documented
regressions are all "the override jumped onto flashlight-illuminated vegetation". The user's
framing avoids that entirely: **the flashlight is its own object, and the person is what else is
moving near it.**

Proposed four stages. Everything in stage 1 already exists.

**Stage 1 — links.** `track_multiple_objects` already runs on every clip and already assigns
persistent ids with `merged_ids`. Weaknesses to fix first, all visible in the code:

- Assignment is greedy and per-track, not globally optimal, so two genuinely distinct subjects with
  overlapping boxes both claim the same candidate and get reported as "merged" rather than
  separated (`scripts/spike.py:760-806`).
- A candidate claimed by any track can never spawn a new id, so a second subject entering inside an
  existing track's box is invisible until it separates.
- Merged tracks dead-reckon indefinitely with no cap.
- It sees only `blobs` (min-area-filtered), never `candidates`, so it cannot see small subjects at
  all — which is precisely the animal population.
- There is no appearance model. Standard SORT/ByteTrack-style association (Kalman motion
  prediction + an appearance histogram, Hungarian assignment) would close gaps without the
  template-lock pathologies that `max_recovered_streak`, `min_reacquire_area`, `max_anchor_streak`
  and the scenery veto have each been added to paper over. **Four separate patches now exist for
  one missing motion model.**

**Stage 2 — per-track type.** Score each track independently:

| type | evidence already available |
| --- | --- |
| `flashlight` | `green_light_ratio(frame, contour)` already takes an arbitrary contour. Plus: small, highly saturated, and it *moves independently of the body it belongs to*. |
| `person` | `subject_height_m` 0.9-2.2 m on the ground plane, `metric_aspect`, translates in depth |
| `animal` | `subject_height_m` < 0.8 m, low, wide |
| `vegetation` | returns to the same place (§4.2), incoherent optical flow (§4.1), matches the per-camera reference background |
| `artifact` | `blob_white_fraction`, no ground-plane solution, near-lens scale |

The capability map already notes that scoring flashlight-ness per object is "a mechanically small
extension of an existing primitive, not new detection work". That is the single highest-value item
in this whole document relative to its cost.

**Stage 3 — beam vs torch.** Once the flashlight is its own object, the vegetation-hijack failure
becomes expressible for the first time. A torch head is *small, saturated, compact and moving*; a
beam-lit bush is *large, diffuse, lower-saturation and stationary*. `FLASHLIGHT_MIN_SATURATION` at
130 already encodes half of this (session #9 measured flashlight S p50 168 vs sunlit grass p99 99).
Per-object, you can additionally require compactness and motion, which no whole-clip scalar can.

**Stage 4 — clip verdict as a set, not a winner.** The clip carries *every* track type it found.
This is what directly serves "missing an incident is a critical failure": today a guard's flashlight
short-circuits the whole chain at rule 1 and nothing else in the clip is ever evaluated. Under
stage 4 the guard track is explained and *removed*, and whatever is left is still assessed. A guard
and an intruder in the same frame is currently inexpressible; it is the exact scenario this system
exists for.

It also cleans up a known accepted risk: `classify()`'s docstring notes that "a genuine incident
that a guard responds to within the same clip would be routed to guard", mitigated only by a
threshold margin. Stage 4 removes the risk rather than bounding it.

---

## 4. Feature directions worth trying

Filtered against everything `docs/handoff.md` records as already tried and rejected (see §6). These
are the ones that are genuinely new *and* target a documented open failure.

### 4.1 Optical flow — built and measured; the single-pair version does NOT work as hypothesised

**Update, same day, later session:** implemented and measured against the full labelled corpus.
The specific mechanism proposed below (single-pair directional coherence within a blob) does
**not** separate environment from a real subject — it reads backwards (environment median 0.964,
higher than every other class). Full measurement and the likely mechanism (5 fps is too coarse to
see an oscillation within one step; it only shows up over many steps) are in
`src/features.py::optical_flow_direction_coherence`'s docstring. Kept as reporting-only
infrastructure, a genuinely new axis, but **do not re-attempt this exact hypothesis expecting a
different result on more data** — the credible next variant (tracking the per-pair mean flow
ANGLE's sign-change rate across a whole clip, not single-pair internal coherence) is unmeasured
and is where anyone picking this back up should start. Original proposal kept below for context.

**Nothing in this codebase computed optical flow before this session.** Every motion feature is
frame-differencing against a whole-clip median. That leaves an entire, standard axis unused:

- **Directional coherence within a blob.** A rigid body's flow vectors all point the same way. Wind-
  shaken foliage has vectors pointing in many directions at once. This is the textbook discriminator
  for exactly the "single waving branch" population that sessions #6 and #7 both flag as unsolved
  and that `blob_count > 10` cannot catch by design (one branch is one blob). **Measured: does not
  hold at single-pair granularity on this corpus, see the update above.**
- **Curl and divergence.** Oscillation has high curl; translation has none. Not attempted.
- It is per-object, so it drops straight into §3 stage 2. Not attempted per-object either — only
  the single-track version was built.
- Farneback dense flow on 320×240 at 5 fps is cheap. `cv2.calcOpticalFlowFarneback` needs no new
  dependency. **Sparse Lucas-Kanade was used instead** (see below), matching the caution already
  in this section.

The one caution: 5 fps is slow for flow, and these subjects can move a long way between frames.
Sparse Lucas-Kanade on corner features inside each blob is the more robust variant at this frame
rate and is worth trying first. **This is what was built and measured.**

### 4.2 Temporal recurrence, not net displacement

Session #2 measured "net displacement / path length" at AUC 0.665 and correctly rejected it —
it is confounded by short tracks (exactly two centroids always scores 1.0 trivially).

A different question that is not the same measurement: **does the object come back?** A branch
returns to where it started; a subject does not. Concretely, two cheap forms:

- **Per-track box revisit:** does the track's box at frame *N* overlap its box at frame 0, having
  moved away in between? Robust to short tracks in a way the displacement ratio is not.
- **Per-pixel duty cycle:** count how many times each pixel *crosses* the foreground threshold
  across the clip, rather than accumulating magnitude. A vegetation pixel alternates repeatedly; a
  subject's pixel fires once as it passes. `scripts/motion_heatmap.py` already accumulates
  per-pixel change but takes a peak over time, deliberately (its docstring records why). The
  *crossing count* is a different statistic on the same data and is not currently computed
  anywhere.

### 4.3 Spatial dispersion of motion mass

`blob_count` counts components but says nothing about how they are spread. Two candidates that are
cheap and orthogonal to it:

- **Radius of gyration of motion mass** about its centroid.
- **Fraction of total motion area in the largest connected component.** A subject is one compact
  region; wind is many.

Both would help the single-branch case that `blob_count` structurally cannot see.

### 4.4 Fix the background model (repeatedly deferred, twice-evidenced)

The whole-clip median background is the root cause named in at least three separate investigations:

- `cam06/21520`'s crawler dwells through most of its own clip, so it biases its own median and
  nothing diffs against it (session #5's IR-flare work says explicitly this is "a background-MODEL
  problem, not a warmup-compensation problem").
- cam07's short clips have 3-5 considered frames — far too few for a stable median — so the bush's
  own texture wobble reads as spurious motion. Session #13 confirmed these detections are
  `recovered=False`, i.e. genuine diff hits, which is why the scenery veto cannot catch them.

`src/reference_bg.py` already builds and buckets 133 per-camera references. The unbuilt step is
**using the reference as the background model** when a clip has too few frames or the subject
dwells, rather than only as a downstream veto. Flagged as needing explicit sign-off in sessions #3,
#4 and #13, and never attempted. It now has two independent, reproducible triggers.

### 4.5 A learned person/animal detector — an honest option, not a recommendation

`docs/plan.md` commits to classical CV, and I am not proposing to break that unilaterally. But it
should be said plainly: the biggest single lever for "person or animal" is a detector trained on
this site's own footage. 429 guard clips supply thousands of person crops. A generic COCO-trained
model would do badly on 320×240 night IR with 20-pixel subjects; a small model fine-tuned on this
corpus is a different proposition. The user's own note that "the LLMs so far cannot reliably
distinguish features from these clips" is about zero-shot vision-language models and does not
transfer to a supervised detector fine-tuned on the actual data.

The honest cost: a labelled box set (not just clip-level labels), a training loop, and a new class
of failure mode this repo has no discipline for yet. Worth a scope conversation, not a slipped-in
change.

---

## 5. Post-analysis gating

The user asked specifically whether post-processing can gate the classifier better. Five options,
strongest first.

### 5.1 Per-camera percentile normalisation — highest leverage, needs no labels

This directly attacks the problem `docs/gate2_separability_finding.md` identifies as fatal:

> The model therefore learned "not guard-like" as a proxy for positive — and cam07, which has 3738
> clips and almost no labels, is uniformly not-guard-like. cam07 takes 151 of the top 250.

Label coverage is wildly uneven and cannot be fixed quickly: cam07 has 3,799 clips and 55 labels
(1.4%); cam10 has 180 clips and 94 labels (52%). The fix is not more labels, it is **expressing
each feature as a percentile within its own camera's distribution** rather than as a raw value. A
clip in cam10's 99th percentile for `blob_count` is anomalous *for cam10* even though cam10's
baseline is high. Every threshold becomes "how unusual is this for this camera", which is the
question anomaly detection actually asks.

The 16,887-clip corpus supplies those distributions for free, with no labels at all.

That same doc measured this and reported it works (top-500 recall 5/15 → 9/15). **It was never
implemented** — I grepped; there is no per-camera standardisation anywhere in the repo. Either
build it or delete the claim.

### 5.2 Sibling-clip fusion — fully measured, decided, never built

Sessions #6, #7 and #13 measured this thoroughly. Median sibling gap 190s, p95 285s; a 30s wait
catches 6.1% of pairs, 60s catches 10.2%, you need ~300s for 99.9%. Session #13's own
recommendation is right and should just be built:

> Fire on the short clip immediately (never suppress a real detection) and send a **correction** if
> a fuller sibling later disagrees, rather than holding every alert for minutes.

`classify_event()` already exists for the reduction and is currently **dead code outside tests** —
verified, its only non-test caller is its own definition. Note the measured nuance: a naive
OR-across-siblings buys recall the system already has while making leak worse. Trusting the
*longer* clip specifically is the promising variant and is still unmeasured.

### 5.3 Patrol context as a guard explanation — free, and pointing the right way

Measured this session from timestamps alone, no video decoding. "Did a camera within 3 fence
positions fire in the preceding 6 minutes?"

| class | past-only, 6 min |
| --- | --- |
| guard | 21.7% (93/429) |
| environment | 12.6% |
| **animal** | **5.4% (2/37)** |
| incident | 30.0% (n=10) |

A 4x ratio between guard and animal, on a signal that costs nothing and is completely orthogonal to
every image feature. Animals are solitary single-camera events; guards sweep the line.

`docs/plan.md` downgrades sequence work to retrospective-only on the grounds that probe-sequence
evidence is thin (n=2). That is right about the *positive* direction and wrong about this one: the
well-evidenced signal here is the **negative** — hundreds of patrols say what a guard sweep looks
like. Absence of a sweep raises animal likelihood. Both the forward-looking and past-only versions
were measured; past-only is what a live system can actually use.

Use it as a ranking/priority input, not a hard gate — 21.7% is far too weak to suppress on.

### 5.4 Group the review queue by corroborated weather event

`src/storm_events.py::find_corroborated_events` is built, tested and validated (100% precision at
default settings, 11/11 — though note the Wilson 95% lower bound on 11/11 is about 74%, and the
parameters were chosen on the same two review batches the precision was measured on).

It is explicitly not a label and should not become one. But as a **presentation** layer over an
existing alert queue it is free and safe: if three neighbouring cameras fired within 30 minutes,
show the reviewer one weather event instead of three separate alerts. That is a review-burden win
with zero classification change.

Note `exclude_twilight_minutes` defaults to `None` despite the measured 38% `blob_count` guard
false-fire spike 15-60 minutes after sunset that it exists to address. The mitigation ships off.

### 5.5 Make the recorded backtest runs readable

`src/backtester.py` records immutable runs with a config hash and git revision. **Nothing reads them
back** — `iter_backtest_runs`/`iter_backtest_results` are called only from tests. The module's own
docstring says it exists so "two runs could never be reliably compared or reproduced"; half of that
is delivered. A `backtest-diff` tool comparing two `run_id`s, plus real metrics in
`scripts/backtest.py` (§0), would turn every future change from a hand-measured session narrative
into a reproducible number. Given how much of this repo's history is "an earlier figure went stale
and nobody noticed", this pays for itself quickly.

---

## 6. Do not retry these

Recorded across `docs/handoff.md` with measurements. Listed so this document does not accidentally
send someone back around a loop.

| idea | why it failed |
| --- | --- |
| `aspect_ratio < 0.95` for animal/incident | premise fully inverted; AUC 0.518 |
| jitter + solidity insect rule | 0/22 environment clips reach jitter>50 under the current tracker |
| motion heatmap as a detection prior | lights 60-68% of frame on vegetation clips |
| per-blob NCC vs reference background | AUC 0.42-0.45, *worse than random* — NCC is brightness-invariant, so a lit rail still matches |
| net displacement ÷ path length | AUC 0.665, confounded by short tracks |
| whole-frame motion fraction as a storm metric | did not separate the 4 known storm clips |
| colour/saturation as a daylight proxy | failed four separate times; a flashlight moves every such statistic the same way ambient light does. Use the clock |
| lowering the detector threshold globally | 18→12 drops LOO AUC 0.968→0.832 |
| multi-scale template matching | box flickers between per-frame best scales |
| high-pass illumination filtering | destroyed small-animal detection (rooikat −80% hit pixels) |
| frame-local background similarity for static locks | structurally circular — a frame is recovered *because* it resembles the background |
| `metric_aspect` gate on the incident channel | crawlers sit at the very bottom of the range, 0.08 above the threshold |
| peak `motion_pixel_fraction` as a gate | worst real incident 0.123 vs guard/environment median 0.172 |
| `long_flare_frames >= 18` as a gate | the crawl incident sits at 15 |
| `post_flash_red_shift` as a guard rule | zero-leak corpus-wide but redundant at its position in the chain |
| `multi_object_dominant_outside_fraction` as an OR-alternative geometry gate | measured 2026-09-09 after the bbox fix (§1.1) — swept evidence floors from `count>=5, mdof>0.6` to `count>=15, mdof>0.8`; environment leak never drops below ~7-10 clips while recovering at most 1 real animal clip (cam15/15454). The bbox fix did NOT resolve this — a storm/wind clip's scattered motion often reads as "one dominant object, mostly outside" just as convincingly as a real intruder does, for the same reason `blob_count` exists as a separate gate. Confirms the capability map's own prior caution ("not usable as a classify() gate until the guard leak's actual cause is understood") independent of the bug. |

**New structural finding from this measurement, not previously documented: an animal seen entirely INSIDE the fence has no path to `animal_candidate` at all.** `classify()`'s inside-only fallback only ever produces `guard_candidate`/`resident_candidate` (split by daylight) — there is no inside-only animal category. Checked against the full labelled corpus: of 6 labelled animal/incident clips reading "inside" by both `outside_pixel_fraction` and the (bug-fixed) multi-object reading, 4 already have an alerting sibling in the same event (cam06/21519, cam09/21521, cam15/17949, cam05/18788) — real ground truth, but not a live gap, matching this repo's own "check per event" discipline. The remaining 2 are genuinely uncovered: cam15/15453 (the porcupine's own event IS recovered by its sibling 15454's real, if fragile, outside-geometry reading — see the table above) and cam10/17146 (bird on the fence rail — no sibling, no calibrated-height signal available since the ground-plane assumption is exactly what's violated here). **Population is currently 1 clip** (cam10/17146) once event-level coverage is accounted for — too thin to engineer a threshold on, same call this repo already made for the `resident` rule at a similarly small sample. Worth tracking as a real, named gap and revisiting once more labelled examples of an animal seen inside the fence exist; not worth guessing a rule against n=1.

Two meta-lessons from that history are worth restating, because both cost real time:

1. **Check the protected classes' own distribution before believing a threshold derived from
   confirmed examples.** Three signals looked clean on hand-picked clips and died on base rate.
2. **A bounding box sitting on something that looks like foliage in a dark IR still is not evidence.**
   This repo has two logged retractions where an AI misread grainy frames and the user's own
   knowledge of the footage overturned it. Do not conclude from stills.

---

## 7. Suggested order

Ranked by value per unit of risk.

**Do first — cheap, verified, low risk**

1. **Fix the `_multi_object_outside_features` bbox bug** (§1.1) and re-derive its two unit tests.
   One line. Then re-run the measurement that shelved the per-object features — it was taken on
   corrupt numbers.
2. **Make `check_incident_regression.py` assert on animal events too, and pass a reference
   background** (§1.2). Ship readiness criterion #2 is currently unenforced.
3. **Add real metrics to `scripts/backtest.py`** (§0) — a confusion matrix against the `label`
   column it already carries, plus per-camera breakdown. Everything else in this list is easier to
   judge once this exists.
4. **Ship the neighbour rule** (§2.2). 5/5 with zero protected-class cost, and daylight's 1.8%
   corpus share caps the channel size by construction.

**Do next — real work, well-evidenced**

5. **Per-object flashlight scoring** (§3 stage 2). Small extension of an existing primitive, and it
   is the foundation of everything the user asked for.
6. **Per-camera percentile normalisation** (§5.1). Already measured to work, never built, needs no
   labels, and attacks the corpus-coverage problem nothing else can.
7. **Fix the ranker's sibling leakage and prune its feature vector** (§1.3). Both are one-line
   changes with a measured +0.064 AUC waiting.
8. **Optical-flow coherence per object** (§4.1) — the genuinely missing measurement axis, aimed at
   the single-branch environment population that is explicitly unsolved.

**Bigger, needs a decision first**

9. **Factor the classifier into what/where/when** (§2.1) and make a clip carry a *set* of findings
   (§3 stage 4). This is the change that stops a guard's flashlight from short-circuiting the
   evaluation of everything else in the frame.
10. **Reference background as the primary background model for short/dwelling clips** (§4.4).
    Corpus-wide blast radius; has been correctly deferred three times pending sign-off.

---

## Open questions for the user

These change what gets built and cannot be answered from the data.

1. **The fence-distance band (§2.3).** Removing it recovers 11 of 22 missed animal clips at a cost
   of 17 guard and 16 environment false alerts on the labelled corpus. Given "we want to be
   notified of animals" and "noise is environment movement", which way does that trade go? A middle
   setting (0.05-0.50) recovers most of the animals for about half the cost.

2. **Ship readiness criterion #3 is still unset** and now blocks more than it did. How many
   false alerts per night is actually reviewable? Without a number, every threshold decision in
   this document is an agent guessing at your tolerance.

3. **Should a clip be able to carry more than one finding** (§3 stage 4) — "guard present AND
   animal present"? It is the right model, but it changes what an alert *is*, and downstream
   notification has not been designed yet.

4. **Is a supervised person/animal detector in scope** (§4.5), or does the classical-CV commitment
   in `docs/plan.md` stand? This is the largest available lever on the animal/person axis and the
   answer determines whether §2.2's metric-height approach is the ceiling or a stepping stone.

5. **More `neighbour` and `resident` examples.** Both classes have 5 and 8 labelled clips. The
   neighbour rule above is fitted on all 5 of them, which means it is a strong lead, not a
   certified rule. A targeted search for daylight-outside clips would be cheap — there are only 304
   daylight clips in the entire corpus, so that population can be reviewed exhaustively.

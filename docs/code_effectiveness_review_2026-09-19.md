# Detection and tagging code review — 2026-09-19

Reviewed baseline: `df536f2`, branch `feat/phase2-refactor`, extractor
`motion-features-v6`. The baseline findings below are retained so the evidence
and rejected assumptions remain reviewable. The implementation follow-up later
the same day moved the extractor to `motion-features-v7`.

**Assessment:** the pipeline detects useful motion and retains all five labelled
incident camera-trigger events, but neither its object identities nor its semantic
tags are reliably correct in general. There are reproducible correctness defects,
two missed animal events outside the regression fixture, and a debug renderer that
can disagree with the classifier it is supposed to explain. Fix the validation and
renderer first, then measure detector repairs before implementing suppression in
the live event resolver.

## Evidence and limits

Read the README and all seven existing `docs/*.md` documents, including historical
corrections and rejected experiments. Reviewed the detector, tracker, geometry,
feature scoring, classification, calibration, references, cache, backtester,
labelling, ranking, ingestion/persistence and debug-rendering paths. The deepest
checks below concern detection and tagging; the absent live service was not tested.

Verification performed:

- Ruff clean; all **759 existing tests pass**.
- Re-ran the real incident regression: 31/31 fixture clips processed; its current
  assertions pass, including its one animal exception. This assertion is weaker
  than the alert policy; see finding 1.
- Re-ran all **708 labelled clips** with the normal reference backgrounds, both
  cached and uncached. Categories, reasons, maintenance flags and all other feature
  columns agree except `global_camera_shift_score`: 20 rows differ by at most
  0.030. That feature is reporting-only. Do not claim byte-identical extraction.
- Replayed the CSV's feature mappings through `classify_detailed`: zero category
  mismatches when absent values and `no_features` are reconstructed correctly.
- Grouped by source channel, camera and embedded caption timestamp: **366 events**.
  Extracted ten additional, unlabelled sibling clips so the policy comparison uses
  all available siblings, not just labelled rows. No labels were propagated.
- Ran 14 synthetic counterexamples through current functions. These test specific
  invariants; they are not estimates of real-world failure frequency.

The existing labels describe events, not frame-by-frame boxes or track identities.
Consequently, classification recall cannot establish localization accuracy, identity
consistency, or whether the selected blob is the actual subject. No new visual
ground truth or species labels were invented in this review.

## Implementation follow-up — 2026-09-19

The concrete correctness recommendations are now implemented and tested:

- metadata-only clip upserts preserve downloaded media paths;
- the protected-event checker uses an explicit urgent-category allowlist, fails
  on missing media/config coverage, contains all 49 known protected-event sibling
  clips, and passes 5/5 incident plus 20/20 animal events with no exception;
- the debug renderer applies ignore polygons and computes features, daylight and
  classification from the same `ClipDetection` it draws;
- warmup inside suppression requires classifiable geometry;
- `outside_area_fraction` and `zone_classifiable_area_fraction` measure the filled
  silhouette independently of contour vertex density and remain reporting-only;
- bbox conversion round-trips, template matching rejects evidence-free flat
  windows/non-finite scores, missed-frame prediction accumulates velocity,
  confirmation is consecutive, and unrelated merges no longer disqualify the
  primary subject;
- exposure settling distinguishes a real late light change from startup while
  retaining delayed near-black IR startup behavior;
- parallel ranking reports extraction exceptions instead of silently converting
  them to zero-valued no-motion rows.

All 14 updated synthetic invariants pass. The cold v7 run over all 708 labelled
clips, replayed through the final classifier, gives TP 32 / FP 24 / FN 15 / TN
637 when unknowns count as negatives: precision **57.1%**, recall **68.1%**, F1
**62.1%**. Excluding unknowns gives TP 32 / FP 18 / FN 15 / TN 607: precision
**64.0%**, recall **68.1%**, F1 **66.0%**. At event level, the current any-sibling
policy has TP 24 / FP 16 / FN 0 / TN 307 over consistently labelled events.

The initial coexistence proposal—letting an outside non-flashlight track override
a flashlight—was rejected after the cold run: it created eight new guard alerts
across seven cameras. The coexistence measurements remain available, but production
classification keeps the flashlight decision until jointly occurring subjects are
labelled. The filled-area feature likewise remains reporting-only; replacing the
legacy vertex statistic would require a separate threshold derivation.

The five incident groups span **two incident nights**; several are related camera
triggers. They are not five independent trials of general security effectiveness.
The corpus has 16,887 metadata rows; the 708 labelled clips are a selected sample.
These results do not establish unattended false alerts per night or unseen-incident
recall.

## Current measured effectiveness

| Label | Labelled clips | Urgent-alert clips | Consistently labelled events | Events with any urgent-alert sibling |
| --- | ---: | ---: | ---: | ---: |
| incident | 10 | 7 | 5 | 5 |
| animal | 37 | 22 | 19 | 17 |
| guard | 444 | 8 | 229 | 7 |
| environment | 168 | 12 | 87 | 11 |
| neighbour | 5 | 0 | 3 | 0 |
| resident | 8 | 0 | 4 | 0 |
| unknown | 36 | 6 | 18 | 5 |

Urgent means `animal_candidate` or `incident_candidate`. Neighbour tagging succeeds
on 4/5 clips; those are intentionally outside this urgent channel. Resident tagging
succeeds on only 2/8 clips; five residents are called guards.

One additional event, **cam10/9404–9405**, has conflicting `unknown`/`animal` labels.
It alerts, but is excluded from the consistently labelled event denominators above.
It remains included in the per-clip counts under its individual row labels. Report
this conflict; do not silently choose the first or last sibling's label.

The shipped summary gives TP 29 / FP 26 / FN 18 / TN 635, precision **52.7%** and
recall **61.7%**, because it treats unknowns as negatives. Excluding unknown clips
gives TP 29 / FP 20 / FN 18 / TN 605, precision **59.2%**, recall **61.7%**.
Neither convention should turn user-approved ambiguous sightings into confirmed
false detections.

Two consistently animal-labelled events have no alerting sibling:

| Event | Current result | Direct exclusion |
| --- | --- | --- |
| cam12/9720–9721 | `no_motion`, `unclassified` | 9721: outside=1.0, median fence distance=0.4783, above 0.40 |
| cam12/10081–10082 | both `unclassified` | outside=1.0; distances=0.4848 and 0.4865, above 0.40 |

Both are absent from the regression fixture. Their labels merit preservation and
review; the note on 9721 is itself uncertain. This audit establishes a mismatch with
the recorded animal labels, not a new visual confirmation of their contents.

## Prioritized correctness findings

### 1. P1 — The protected-event regression can pass without an alert, or without footage

Locations: `scripts/check_incident_regression.py:97`, `:158`, `:187`;
`tests/fixtures/incident_regression.jsonl`.

The checker tests absence from a suppression denylist. That list omits
`insect_candidate` and `neighbour_candidate`, so either counts as a successful
incident detection even though neither is an urgent alert. A synthetic incident
classified as each category prints PASS and exits 0. Skipping its only fixture clip
also prints PASS, with **0/1 clips checked and zero incident events**.

Separately, only eight animal fixture events are checked, versus nineteen
consistently labelled animal events plus the mixed-label event now in the DB.
The cam10/7632 exception also omits the alerting 7631 sibling; the docs already
corrected the assertion that this event is lost. Fixture rows 10560/10561 still say
resident although the live labels and documented correction say neighbour.

**Next agent:** use an explicit shared urgent-category allowlist; require every
expected protected event and its required media to be evaluated; treat extraction
failure distinctly; compare fixture coverage and labels with a frozen labelled
event manifest. Include every sibling of 7632 and the two Cam12 misses. Preserve
known misses as visible baseline failures/explicit reviewed dispositions, not new
silent exceptions. Tests must cover missing all footage, partial sibling loss,
insect/neighbour outcomes and newly added protected labels.

### 2. P1 — Metadata replay deletes the database's links to downloaded footage

Locations: `src/backfill.py:56`; `src/db.py:211`.

Metadata backfill supplies `file_path=None`; the conflict update unconditionally
assigns `file_path = excluded.file_path`. Replaying a previously downloaded clip
therefore changes its path from a valid filename to NULL. The files remain on disk,
but labelling/backtest selection skips them and the downloader selects them again.
This contradicts the README's safe-rerun promise and affects the planned duplicate
live-message handling too. Reproduced against an isolated in-memory DB.

**Next agent:** separate metadata updates from media-state updates, or preserve an
existing path when the metadata-only operation has no replacement. Use an explicit
operation to invalidate media deliberately. Test downloaded clip → repeated
metadata ingestion → same path, labels and evaluation eligibility. Audit any
existing NULL paths against local files without re-downloading blindly.

### 3. P1 — The debug video does not show the same analysis as the backtest

Locations: `scripts/render_debug.py:527`, `:541`, `:553`, `:581`.

Three mismatches coexist:

- The overlay's direct `detect_clip` call omits `ignore_polygons=zone.ignore`.
  A second extraction used for the HUD applies the ignore regions. The displayed
  box can therefore belong to a different object from the one being classified.
- The renderer passes `daylight_hint` into extraction but never adds
  `features['is_daylight']` before classification. Those are different inputs.
  On real footage, 17146 displays `environment_candidate` instead of the
  backtest's `animal_candidate`; 9405 displays `unclassified` instead of animal;
  3495 displays `unclassified` instead of neighbour. Removing this feature from
  all current CSV rows changes 31 categories, including six animal clips.
- The overlay still gates flashlight labels on whole-frame colour/daylight,
  while scored-frame flashlight classification deliberately removed that gate.

**Next agent:** use one clip-analysis result for the boxes, feature vector,
classification, clock interpretation and explanatory tags. Reuse
`features_from_detection`, with ignore regions applied at detection, instead of
decoding independently twice. Test the six daylight animal cases, neighbour 3495,
an ignored-light camera and a daytime flashlight. Fix this before asking a human
to adjudicate apparent detector failures from new renders.

### 4. P1 — Missing warmup geometry is interpreted as confirmed inside motion

Locations: `src/scoring.py:247–258`; `src/classify.py:537–555` and `:446–473`.

`dynamic_hits` includes motions with ambiguous side classification. If every
verdict is ambiguous, `warmup_dynamic_outside_fraction` becomes 0.0. The classifier
then treats that as evidence of inside motion. Two textured synthetic warmup frames
with a moving grey object and no settled track produce
`guard_candidate/warmup_dynamic_inside` with dynamic fraction=1.0, outside=0.0,
both when there is **no fence geometry** and when all motion is **above the depth
cutoff**. No flashlight is present.

The terminal-artifact guard rule consumes the same ambiguous zero. This is the
absence-versus-inside distinction already handled correctly for settled geometry.

**Next agent:** add warmup classifiable-evidence count/fraction and require real
inside evidence at both suppression sites. Preserve no-evidence as unknown.
Test missing fence, missing side, all-above-cutoff, mixed ambiguous/outside motion,
and the reviewed 5464/17501/5465 guard recoveries. Bump the extractor version.

### 5. P1 — Outside “pixel fraction” depends on polygon encoding, not foreground area

Locations: `src/scoring.py:82`, `:444–540`; `src/zones.py:127–140`.

The scorer passes simplified contour **vertices** to `outside_pixel_fraction`.
Every vertex receives one vote. A rectangle covering x=40…90 with a vertical fence
at x=50 scores **0.5** although **78.4% of its filled pixels** are outside.
Adding two collinear points on the rectangle's right edge changes the score to
**0.667**, with no change in its physical shape. With otherwise identical outside
features, the category changes from `unclassified` to `incident_candidate`.

Rectangular recovery boxes are particularly exposed: four vertices frequently
produce only 0, 0.5 or 1 regardless of the true area split. The low-level function's
docstring permits vertex samples, but that makes it a different measurement from
the pixel-area rule described in the plan. It is not invariant to contour
simplification, so it is unreliable as a purported physical majority.

**Next agent:** first add an independent rasterized area feature with a zone mask
and explicit classifiable pixel count. Compare it with the old vertex feature on
every labelled clip and the standing render sample, particularly 15454 and crawling
incidents. Do not silently swap definitions under thresholds fitted to vertex votes;
measure and document new thresholds, protected-event deltas and a cache-version
bump. Test shape resampling, slanted fences, holes, depth exclusion and image edges.

### 6. P2 — Multi-object identity prediction stops advancing during missed frames

Locations: `src/motion.py:1084–1096`, `:1160–1168`, `:1204–1209`.

Unmatched tracks increment `miss` but keep the last observed box. Every following
prediction is only one velocity step beyond that old box; it never accounts for
the accumulated gap. Reacquisition also records the entire gap displacement as a
one-frame velocity. A four-pixel-wide object at x=10,16,missing,missing,34 follows
exactly constant velocity and is inside the five-frame miss budget, yet its IDs
are **0,0,missing,missing,1**.

`pending` also survives misses: `confirm_frames=2` confirms hit/miss/hit as two
“consecutive” observations. This second issue is dormant at the current default
of one, but prevents the optional gate from doing what it documents.

**Next agent:** track last-observation frame and predicted state separately; use
elapsed frames for prediction and velocity, reset consecutive confirmation on
misses, and define confirmation behavior through merges. Test zero/one/multiple
misses, crossing subjects and merge/split. Measure fragmentation, not only category
counts. Do not raise the default confirmation threshold; that previously lost real
short-lived flashlight evidence.

### 7. P2 — Flat templates produce confident recovery without location evidence

Location: `src/motion.py:762–802`.

`reacquire_by_template` uses normalized coefficient matching without rejecting
zero-variance templates. A constant-grey template of value 30 matched against a
constant image of value 200 returns a box even at threshold 0.99. It chooses the
search-window corner despite there being no subject or distinguishing texture.
This can populate inferred geometry and false trajectories. `_patch_similarity`
already handles flat patches separately; template reacquisition does not.

**Next agent:** return unavailable/uncertain for degenerate localization evidence;
test flat and near-flat templates, empty/nonfinite scores and textured positive
controls. Measure effects on the known faint animals before choosing a variance
floor. Raising the correlation threshold cannot fix this example.

### 8. P2 — Bounding-box conversion enlarges inferred objects

Locations: `src/motion.py:703`, `:756–759`; consumers in `src/scoring.py`.

Boxes use exclusive upper corners from `cv2.boundingRect`. Conversion back to a
contour writes those exclusive coordinates as included vertices. A round trip
`(10,20,20,30)` becomes **(10,20,21,31)**. This adds a row and column to recovered
boxes and per-object colour regions; it is material for very small subjects.

**Next agent:** define one pixel-coordinate convention and test box→contour→box,
mask pixel count, one-pixel boxes and right/bottom borders. Expect feature changes;
do not treat this as a cosmetic drawing fix.

### 9. P2 — A merge elsewhere in the frame disqualifies the main subject's best frame

Locations: `src/scoring.py:95–101`, `:474`, `:1110`.

`_frame_is_merged` returns true if **any** multi-object track is merged. Both the
appearance and geometry reducers then skip the main contour for best-frame
selection, even if it is a separate object elsewhere. Reproduced with an isolated
main subject at bottom-right and two tiny merged tracks at top-left.

**Next agent:** associate the selected contour with its own object/merge provenance.
Test that an unrelated merge cannot change its selected frame or geometry, while
an actual merged main silhouette is handled explicitly. Real-corpus impact remains
unmeasured; this is a demonstrated scope error, not an estimated error rate.

### 10. P2 — A late flash can delete valid early motion as “opening warmup”

Locations: `src/features.py:716–738`; `src/motion.py:339–365`, `:516`.

`flare_settle_index` scans the entire clip for the last brightness jump. Eighteen
stable medians followed by a late two-frame flash cause the first **eight stable
frames** to be discarded under the 40% cap. That is not the opening gain ramp.
The later `is_flare` flags are displayed but never exclude frames from scoring.

**Next agent:** distinguish opening exposure settling from later exposure events,
and retain per-frame validity rather than deleting an unrelated prefix. Test an
early subject followed by a late flashlight, recurring flare, and a genuine opening
ramp. Do not simply discard every flare-marked frame: the docs already identify
real subjects that cause these brightness changes.

## Architectural and measurement limitations

**Flashlight presence still suppresses every other subject.** This is an existing
documented limitation, not a newly discovered object-association fix. Supplying
strong independent outside evidence and then adding a separate flashlight changes
`incident_candidate` to `guard_candidate/multi_object_flashlight`. Even
`multi_object_dominant_excl_flashlight_has_evidence=1` and outside fraction=1 are
ignored by the classifier. The same issue exists in the earlier flashlight rules.
Per-object flashlight detection is useful; a guard explaining the entire clip is a
separate assumption that is false for coexistence scenes.

The next object-level design should preserve each track's evidence and findings
before applying event routing. A guard finding should not explain away an unrelated
outside track. Add coexistence fixtures before wiring any such policy. Do not
enable the already-rejected person/animal track-count rules as the solution.

**Person/animal tags are size hypotheses, not recognized species.** Current
`multi_object_animal_track_count > 0` occurs in **138/168 environment clips** versus
**14/37 animal clips**; person-height tracks occur in **125/168 environment clips**.
These diagnostics correctly remain outside `classify()`. The general animal versus
incident branch mainly splits on colour, not anatomy. A semantic model would need a
separate labelled-box evaluation on these actual IR cameras before any claim of
improvement; no model migration is proposed by this review.

**A stationary subject can be absorbed into its own median background.** The
existing reference-as-primary experiment exposed foreground but did not recover
animal classification. Keep that measured negative. A next experiment should
compare candidate-level evidence from both backgrounds, with registration quality,
reference age and temporal consistency recorded, instead of globally replacing the
background or lowering the difference threshold.

**Evaluation still has blind spots.** Unknowns are counted as negatives in the
summary and ranker; ranker scaling is fitted before leave-one-out splitting;
longest-per-event fitting can still retain blank/duplicate rows when the informative
sibling is absent or undetected. Neighbour is omitted from `NEGATIVE_LABELS` and
silently excluded from that model's fit. Parallel extraction catches all exceptions
and changes them to undetected clips (`scripts/rank_candidates.py:245`). Report
errors explicitly, freeze cohort membership, fit preprocessing inside each split,
and split by camera/night/incident episode as appropriate. Existing in-sample
threshold margins are not held-out validation.

**Video failure is conflated with no motion.** Zero decoded frames become `None`,
and `classify(None)` returns `no_motion`. Introduce explicit decode/analysis status
before building the watcher. A missing/corrupt recording is not evidence that no
subject was present. Bound clip memory/decoding and validate media type as part of
that API; metadata ingestion currently accepts any Telegram document as a clip.

**Time inputs need one definition.** Daylight and geometry currently use Telegram
delivery time, while grouping uses the embedded event time. A several-minute sibling
delay can change daylight classification for the same recorded scene (already
documented for 19025/19026). Preserve event/capture and delivery times separately;
validate camera-clock reliability before changing the timestamps used by existing
rules or dated geometry.

## Sibling-policy experiment

All 366 labelled events and their available siblings were included. For confusion
matrices, exclude the 18 entirely unknown events and the one conflicting-label event:
**347 consistently known events**, with 24 protected and 323 other events.

| Policy | TP | FP | FN | TN | Precision | Recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Any urgent-alert sibling wins | 22 | 18 | 2 | 305 | 55.0% | 91.7% |
| Longest clip alone | 19 | 6 | 5 | 317 | 76.0% | 79.2% |
| Longest guard overrides; otherwise retain any alert | 22 | 13 | 2 | 310 | 62.9% | 91.7% |

The third row is a diagnostic upper-level category comparison, **not a validated
completed-sibling resolver**: it does not prove completion, prefix relationship,
latency, evidence independence or correctness on unseen coexistence events.

Longest-only newly loses these animal events:

- cam10/7631–7632;
- cam01/16027–16028;
- cam05/18790–18791.

Both longest-only and the guard-override candidate also suppress explicitly
approved ambiguous/alert-worthy **cam07/4487** and **cam13/7778**. Longest-only
additionally loses approved **cam07/19288**. The apparently improved confusion
matrix therefore violates documented review decisions. Do not ship either policy
on these numbers alone. The approved-alert list needs an explicit regression layer
separate from the event's semantic label.

## Recommended next-agent sequence

1. **Repair the measurement boundary:** findings 1 and 3; full sibling coverage;
   known/unknown/conflicting labels separated; approved alerts represented explicitly.
   Save baseline category/reason/maintenance results and the reviewed renders.
2. **Fix metadata replay and warmup ambiguity:** findings 2 and 4, each with focused
   regression tests. Verify the five recent warmup guard recoveries and all protected
   and approved events after the latter change.
3. **Repair tracking contracts individually:** gap prediction, flat-template
   localization, box convention and merge ownership. Run each against synthetic
   tracks plus the complete labelled set and standing video sample, retaining
   per-frame provenance. Do not combine several changes into an uninterpretable diff.
4. **Measure area-based side assignment and exposure handling as detector changes.**
   Add evidence alongside existing fields before replacing tuned inputs. Bump
   `EXTRACTOR_VERSION` for every changed feature meaning; revalidate cached/uncached
   agreement and inspect every protected-class change.
5. **Build object truth suitable for validation:** annotate visible-object boxes and
   identities on the known crawl, tiny animals, two subjects, guard-plus-outside
   subject, branches, lens artifacts, ignore-boundary crossings and warmup exits.
   Track localization recall, missed visible frames, identity switches, fragmentation
   and false box duration. Report genuine and inferred frames separately. A smooth
   recovered box is not proof that the object is still present.
6. **Then revisit event routing and deployment.** Preserve clip findings and unresolved
   evidence; measure latency as well as noise. The planned shared `src` analysis API
   should serve renderer, backtest, ranker and watcher with identical inputs. Deliver
   the live/Docker work from `next_agent_handoff.md` after these prerequisites; do not
   substitute the offline scripts for a service.

Preserve the documented negative results: global threshold lowering, single-pair
optical-flow coherence, person/animal height-count gates, higher global track
confirmation, bush ignore polygons, and blanket artifact/camera-shift vetoes are not
validated improvements. In particular, do not mask cam10's bush: known intruders
occupy that region.

## Reproduction and artifacts

Run from the repository root:

```bash
uv run ruff check .
uv run pytest -q
uv run python scripts/backtest.py --labelled-only --no-record --out /tmp/review.csv
uv run python scripts/backtest.py --labelled-only --no-cache --no-record --out /tmp/review-cold.csv
uv run python scripts/check_incident_regression.py
uv run python scripts/audit_detection_assumptions.py
uv run python scripts/audit_detection_assumptions.py --backtest-csv /tmp/review.csv > /tmp/audit.json
```

The audit script reads the real DB only when `--backtest-csv` is supplied, then
opens it read-only. Synthetic DBs and fixtures are isolated. It prints expected
invariants, observed results, and `holds`; false means a reproduced counterexample,
not that this diagnostic command itself failed. It is not yet a CI acceptance gate.
The normal backtest can populate the feature cache even with `--no-record`; use
`--no-cache` when that write is undesirable.

Local evidence is retained under `data/reports/code_review_2026-09-19/`:
`audit.json`, the cached and cold CSVs, summary JSON, test output and regression
output. These are gitignored and reflect the labels/reference images present during
this review. The diagnostic script and this report are the durable handoff.

Documentation reconciliation: the newest handoff supersedes older implementation
plans. README still calls Gate 2 blocking and parts of the capability map still
point into `scripts/spike.py` or call wired features reporting-only. The older
refactor plan is a historical execution record, not a restriction against the
review requested here. Update the current capability map after each repair rather
than appending another contradictory current-state claim to historical prose.

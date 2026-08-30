# Gate 2 — CV feasibility spike: separability finding

Generated from `scripts/spike.py` across every camera with labelled clips
(`data/reports/spike_cam*.csv`). 160 feature rows from 165 labels — 5 clips
produce no detectable motion at all and are dropped, which is correct
behaviour for genuinely blank frames.

Per-camera fence geometry originally existed only for cam05/cam06/cam08; other
cameras got `outside_pixel_fraction=0.0` (ambiguous by design, see
`src/zones.py::classify_zone`). **Updated 2026-08-28: all 18 labelled cameras
now have real fence geometry** (see `config/cameras.yaml` and the "Does wider
geometry bias the result?" section below) — the spike was rerun and every
number in this document reflects the full 18-camera geometry, not the original
3-camera one.

## Detector correction (matters for reading everything below)

The first pass of this spike used consecutive-frame differencing and produced
misleading features. These cameras spend roughly the first two seconds of every
clip settling their IR gain, and that whole-frame brightness swing is a much
larger inter-frame delta than any actual subject. On cam15/15454 it hid a
porcupine completely: the detector reported `persistence=0.22`, `jitter=13.5`
and a 3-pixel representative blob, while the animal is plainly visible crawling
from mid-frame to bottom-left once the image stabilises.

`extract_clip_features` now drops warmup frames, models the background as the
per-pixel median of what remains, rejects blobs larger than 25% of the frame as
residual illumination change, and takes shape features from the *largest*
detected blob rather than whichever frame happened to be last. On that same
clip it now reports `persistence=0.61`, `jitter=6.8`, `aspect_ratio=0.81` — a
smooth, deliberate, low-and-wide track, which is what a porcupine actually is.

All numbers below come from the corrected detector. The earlier version of this
document understated separability substantially.

## Label counts

| label | n |
|---|---|
| guard | 110 |
| environment | 22 |
| incident | 8 |
| startup | 8 |
| unknown | 8 |
| animal | 4 |

Positive class for the gate-2 question (`animal` + `incident`) is **12 clips**
against 132 `guard`/`environment` negatives — an 8.3% base rate. Still far too
few positives to certify precision/recall; treat rates below as indicative.

## The strongest result: green flashlight identifies the guard

`green_light_ratio > 0.05` (green hue on the clip's clearest motion frame)
fires on:

- **60/110 guard clips (55%)**
- **1/50 non-guard clips (2%)**
- **0/12 animal or incident clips**

The guard's flashlight reads as a distinct green in this footage, and after the
detector fix it is a clean positive identifier for "this is our guard" with
essentially no leakage into the classes we care about. It doesn't fire on all
guard clips (the light isn't always aimed into frame), so absence proves
nothing — but presence is highly informative.

### Flicker adds recall: the guard sweeps the beam, doesn't hold it still

The single-frame reading above only looks at whichever frame has the largest
motion blob. It misses guard clips where the beam sweeps into frame at a
different moment. `green_light_flicker` (std deviation of the whole-frame green
ratio across every frame of the clip, after dropping IR-warmup frames) picks up
exactly that: a beam swinging in and out of shot spikes the ratio up and down,
where ambient/reflected green sits flat near zero.

`green_light_flicker > 0.02` alone:

- **27/110 guard clips (25%)**
- **0/50 non-guard clips (0%)**
- **0/12 animal or incident clips**

Combined as `green_light_ratio > 0.05 OR green_light_flicker > 0.02`:

- **67/110 guard clips (61%)**, up from 55% with the single-frame reading alone
- **1/50 non-guard clips (2%)** — unchanged
- **0/12 animal or incident clips** — unchanged

27 guard clips clear the flicker threshold; only 20 of those also clear the
single-frame ratio threshold, so flicker genuinely catches guard clips the
single-frame reading misses, at no added false-positive cost on this sample.

Note this is a *confirmation* signal, not a suppression one. `docs/plan.md`'s
fail-safe rule stands: nothing here may downgrade an outside alert.

## Shape and motion separation

Animal/incident clips are low and wide with smooth tracks; guard/environment
clips are upright and erratic:

| feature | animal | incident | guard | environment |
|---|---|---|---|---|
| aspect_ratio (median) | 0.74 | 0.76 | 1.07 | 1.09 |
| jitter (median) | 2.1 | 38.1 | 33.9 | 52.0 |
| path_length (median) | 25 | 281 | 249 | 868 |

Best rules found:

| rule | recall | precision | FPR |
|---|---|---|---|
| `aspect_ratio < 0.95` | 0.75 | 0.15 | 0.40 |
| `aspect_ratio < 0.95 AND green_light_ratio < 0.05` | 0.75 | 0.22 | 0.24 |
| above `AND jitter < 45` | 0.50 | 0.25 | 0.14 |

Adding the green-light exclusion cuts the false-positive rate from 40% to 24%
at no cost to recall — the flashlight feature is doing real work.

Precision stays low in absolute terms (0.22 at 75% recall, ~2.7x the 8.3% base
rate). As a standalone classifier that is not usable. As an escalation signal
layered on outside geometry — which is what `docs/plan.md` actually specifies —
it is.

## Flashlight false positives (plan.md requires this count)

Across all 18 cameras (110 labelled `guard` clips):

- **58/110 (53%)** have an outside-majority pixel fraction — more than half the
  blob lands outside the fence line purely from geometry, despite being a guard
  patrol on the inside.
- **38/58 (66%)** of those carry a light signal (`green_light_ratio > 0.05` or
  `saturation_ratio > 0.15`).

This confirms the plan's predicted failure mode ("a flashlight beam aimed down
or across the fence lands outside the fence easily") as real and frequent. The
green-light feature is what makes those recoverable rather than pure noise.

## Does wider geometry bias the result?

Expanding fence geometry from 3 cameras to all 18 was checked directly against
this question (2026-08-28) rather than assumed either way:

| metric | 3-camera geometry | 18-camera geometry |
|---|---|---|
| guard outside-majority rate | 58% (36/62) | 53% (58/110) |
| ...recoverable via light signal | 64% (23/36) | 66% (38/58) |
| environment insect-signature rate | 55% (12/22) | 55% (12/22, unchanged — doesn't use geometry) |
| green-light guard identifier (recall / false-fire / animal+incident leak) | 61% / 2% / 0 | 61% / 2% / 0 (unchanged) |

No meaningful bias: the headline rates hold within a few points across 3x more
geometry and roughly double the guard sample. `outside_pixel_fraction` alone
is still not a clean separator on its own — on the 15 newly-geometried cameras,
`environment` clips average a *higher* outside fraction (0.85) than `guard`
clips do (0.51), which is the opposite of naive intuition (insects near the
lens can sit anywhere relative to the fence line, including past it, without
ever crossing anything). This is consistent with, not a change to, the
existing recommendation: outside geometry is an escalation input combined
with the green-light/shape signals, never a standalone classifier.

## IR-insect false positives (plan.md requires this count)

**12/22 (55%)** of `environment` clips match the insect signature
(`jitter > 50`, `solidity < 0.85`) — erratic, ragged, low-solidity blobs. Higher
than the 41% the previous detector reported, because background subtraction
picks up the insect tracks the frame-differencing version was missing under the
IR flare. Consistent with the plan's expectation that IR insects are the largest
false-positive source.

## Recommendation

Materially stronger than the first pass, but still one open decision:

1. The green flashlight feature is a genuine result — combining single-frame
   presence with the flicker/variance signal gets a 61%-hit, 2%-false-alarm
   guard identifier, with zero leakage into animal/incident. Worth keeping
   regardless of what happens at this gate.
2. Both predicted failure modes are confirmed and quantified (58% flashlight
   outside rate; 55% of environment clips insect-like).
3. Shape/motion features separate low-crawling subjects from upright ones in the
   right direction and with sensible magnitudes, but 12 positive clips cannot
   support a precision claim. More `animal`/`incident` labels would change
   confidence here more than any further feature work.
4. Whether this passes depends on the bar: usable as an **escalation** signal on
   top of outside geometry (the design in `docs/plan.md`), not usable as a
   standalone classifier. Confirming that reading is the remaining call.

## Addendum (2026-08-29): straddling, and two new blob-count/motion-area features

Re-run on 250 feature rows (up from 160) after schema v5's class propagation
filled in more clips. Two ideas checked, both directly against real data:

**"Only outside" vs "straddles inside and outside" as a guard signal.** A
single blob whose contour points are only partly outside the fence line
(`0 < outside_pixel_fraction < 1`, i.e. the same blob straddles the line —
this is the flashlight beam or body crossing it, not a track moving over
time) is directional for guard but weaker than it first looked on the smaller
sample:

| label | n | fully outside (==1) | straddles (mixed) |
|---|---|---|---|
| guard | 194 | 23% | 50% |
| incident | 16 | 31% | 31% |
| environment | 22 | 5% | 18% |
| animal | 5 | 20% | 20% |

"Fully outside" alone does **not** separate animal/incident from guard — guard
reads fully-outside about as often as incident does (23% vs 31%), even after
excluding clips with a flashlight signal. "Straddles" is still guard-leaning
but incident's rate rose from 0% to 31% once more incident clips were
labelled, so treat it as a mild prior, not a rule.

**Storm/wind hypothesis (large scattered motion, not one compact blob).**
Added two new spike features (`scripts/spike.py`, `min_blob_area_fraction`
default `0.0005` matching `config/thresholds.yaml`'s `blob.min_area_fraction`):
`motion_pixel_fraction` (peak fraction of the frame that's foreground motion
in any single frame) and `blob_count` (peak number of simultaneous blobs above
the area cutoff). The existing single-largest-blob detector explicitly
discards this — it picks one contour and rejects anything over 25% of the
frame as an illumination artifact, so wind-blown-vegetation clips were
previously invisible to every feature in this document.

| label | n | mean motion_pixel_fraction | mean blob_count |
|---|---|---|---|
| environment | 22 | 0.28 | 24.0 |
| incident | 16 | 0.21 | 14.2 |
| guard | 194 | 0.20 | 6.7 |
| unknown | 13 | 0.10 | 4.4 |
| animal | 5 | 0.07 | 3.4 |

Directionally supports the idea — `environment` clips do show the widest,
most fragmented motion. Not clean yet: `incident`'s blob_count is also
elevated, most likely an artifact of only 16 labelled clips rather than a real
incident characteristic. Needs more labelled `incident`/`animal` volume before
either feature earns a threshold rule.

Both are additive: existing conclusions/recommendation above are unchanged,
this only adds two more candidate columns to `data/reports/spike_*.csv` for
whenever gate 2 (or a future revisit) evaluates thresholds.

## Addendum 2 (2026-08-30): outside/inside geometry fix, re-run

`src/zones.py::side_name` was direction-of-travel-relative (depended on which
way the polyline's points were ordered), which inverted left/right for any
fence traced top-to-bottom -- confusing enough it looked like a config bug.
Fixed to plain screen position (point x vs. the fence's x at the same row);
see the commit for the fence-truth verification. This directly changes every
row's `outside_pixel_fraction`, so both tables above are re-run here on the
regenerated `data/reports/spike_all_2026-08-30.csv` (also reflects the 6
cam01b clips moved from `incident` to the new `resident` class).

**Straddling, re-run:**

| label | n | fully outside (==1) | straddles (mixed) | fully inside (==0) |
|---|---|---|---|---|
| guard | 194 | 31% | 50% | 19% |
| incident | 10 | 80% | 20% | 0% |
| environment | 22 | 77% | 18% | 5% |
| animal | 5 | 80% | 20% | 0% |
| unknown | 13 | 77% | 0% | 23% |

Notably better separation than the pre-fix numbers: guard is now the only
label that reads "fully inside" at all (19%) and straddles half the time,
while incident/environment/animal all read "fully outside" 77-80% of the
time. Still not clean enough alone for a threshold (guard's own 31%
fully-outside rate overlaps the others, and this is measured on a detector
that still has the separate, open fence-height issue affecting some clips
per-camera), but a real improvement over the pre-fix numbers, not just noise.

**Storm/wind, re-run** (motion_pixel_fraction/blob_count don't depend on the
geometry fix, but `incident`'s n dropped 16->10 once the resident clips were
reclassified):

| label | n | mean motion_pixel_fraction | mean blob_count |
|---|---|---|---|
| environment | 22 | 0.28 | 24.0 |
| resident | 6 | 0.41 | 25.5 |
| guard | 194 | 0.20 | 6.7 |
| unknown | 13 | 0.10 | 4.4 |
| incident | 10 | 0.09 | 7.4 |
| animal | 5 | 0.06 | 3.4 |

With the residents removed, `incident` no longer looks storm-like (motion
dropped 0.21->0.09, blob_count 14.2->7.4) -- the earlier "small-n artifact"
caveat was actually those 6 resident clips (bakkie/vehicle, multiple people)
contaminating the incident sample. `resident` itself now reads as the
highest-motion class of all, plausibly real (vehicle + multiple people) but
n=6 is too small to treat as anything but a first look.

## Addendum 3 (2026-08-30): six new features, and why the corpus ranking still fails

Re-measured every rule and feature in this document against the *live* label
set (250 labels, 237 with detections) now that the full clip download has
finished (16886 clips, 16558 with detections).

### The documented rules had silently degraded

Guard labels grew 110 -> 200 since the main body of this document was written,
and the `scripts/backtest.py` rules got worse as they did:

| rule | then | now |
|---|---|---|
| guard: `green_light_ratio>0.05 or green_light_flicker>0.02` | recall 0.61 | recall **0.42** |
| animal_or_incident: `aspect_ratio<0.95 and green_light_ratio<0.05` | recall 0.75, precision 0.22 | recall **0.60**, precision **0.09** |

Guard specificity held up (1 non-guard fire in 56, and it leaked only into
`resident`), so the flashlight signal is still real -- it just no longer covers
most of the guard population.

### Six new features (implemented, tested, committed)

Sixteen candidates were prototyped and scored by AUC against the labels. Six
survived and are now in `src/features.py` / `src/zones.py` and wired into
`scripts/spike.py` (`FEATURE_COLUMNS` is now 24 wide):

| feature | what it captures | why it helps |
|---|---|---|
| `longest_detection_run` | longest run of *consecutive* detected frames, as a clip fraction | separates an unbroken track from scattered flicker; `persistence` alone cannot |
| `area_stability` | coefficient of variation of the tracked blob's area | real bodies hold a silhouette; flashlight pools, foliage and insects swell and collapse |
| `normalised_speed` | mean per-frame centroid displacement in blob-widths | scale-invariant "body lengths per frame", unlike raw `path_length` |
| `heading_change` | mean absolute turn angle between motion steps (sub-pixel steps dropped) | purposeful travel vs erratic wandering |
| `fence_crossed` | does the track appear on both sides of the fence polyline | **strongest new signal** (AUC 0.794 guard-vs-positive): the guard walks the fence line, animals and intruders stay on one side |
| `median_fence_distance` | median abs. horizontal distance from the fence, frame-width fraction | fence-corridor proximity |

Measured through the real pipeline (not the prototype), hand-rolled weighted
logistic regression with leave-one-out CV, 15 positives (animal+incident) in
237 rows:

| model | LOO AUC | P@10 | P@20 | P@30 | P@50 |
|---|---|---|---|---|---|
| before (13 features) | 0.909 | 4/10 | 6/20 | 9/30 | 13/50 |
| **after (19 features)** | **0.968** | 5/10 | **11/20** | **14/30** | **15/50** |

All 15 positives now fall inside the top 50 of 237. Only miss above rank 30 is
`cam05/18270 animal` at rank 31.

`blob_count` -- which already existed but was used by no rule -- turns out to be
the single best noise discriminator on its own (AUC 0.903 for `environment`
vs real subject). `saturation_ratio` is now a marginally better guard
identifier than `green_light_ratio` (0.765 vs 0.752).

### Improved rule candidates (measured, not yet applied to backtest.py)

`LIGHT = green_light_ratio>0.05 or green_light_flicker>0.02 or saturation_ratio>0.05`

| rule | recall | precision | FPR | lift over 6.3% base rate |
|---|---|---|---|---|
| current `aspect<0.95 and grn<0.05` | 0.60 | 0.09 | 0.40 | 1.5x |
| `not LIGHT and outside_pixel_fraction>0.9 and blob_count<=13 and motion_pixel_fraction<0.25` | **0.67** | **0.31** | 0.10 | **4.9x** |

Guard rule: adding `or saturation_ratio > 0.15` lifts recall 0.43 -> 0.58 while
still leaking into zero animal/incident clips.

### Negative result: lowering the detector threshold makes accuracy worse

40% of `animal` clips have a largest blob under 0.0005 of the frame (~38 px on
320x240) -- `cam08/7360` is about 4 px, effectively undetected. The obvious fix
is a lower threshold. It backfires:

| threshold | LOO AUC | P@20 | P@30 | P@50 |
|---|---|---|---|---|
| 18 (current) | **0.968** | 11/20 | 14/30 | 15/50 |
| 12 | 0.832 | 8/20 | 10/30 | 10/50 |
| adaptive 6*MAD clipped to [6,18] | 0.759 | 4/20 | 6/30 | 9/50 |

Keep `threshold=18`. Small-subject recovery has to be a **second gated pass**
restricted to the fence corridor / depth band, never a global threshold change.
Separately, `largest_contour()` has no minimum-area filter, so shape features
get computed on 4-pixel specks and report `aspect_ratio=1.0, solidity=1.0`.
That should become an explicit "motion trigger, no resolvable subject" state.

### Can we find similar events in the corpus? Not yet -- and the reason matters

The 19-feature model was fitted on the 237 labelled rows and used to rank all
16558 detected clips. Every one of the 15 known positives lands between rank
178 and 971. The top 100 contains **zero** known positives.

This is not a feature problem, it is a **sampling problem**. Only 1.51% of the
corpus is labelled, and the coverage is wildly uneven:

| camera | clips | labelled | coverage |
|---|---|---|---|
| cam07 | 3738 | 13 | 0.35% |
| cam04 | 2325 | 10 | 0.43% |
| cam03 | 1159 | 2 | 0.17% |
| cam05 | 2496 | 30 | 1.20% |
| cam13 | 611 | 6 | 0.98% |
| cam08 | 81 | 23 | 28.4% |
| cam10 | 177 | 20 | 11.3% |

Nearly all negatives are `guard` clips from the small, well-labelled cameras.
The model therefore learned "not guard-like" as a proxy for positive -- and
cam07, which has 3738 clips and almost no labels, is uniformly not-guard-like.
cam07 takes 151 of the top 250. Standardising features per camera helps
(top-500 recall 5/15 -> 9/15) but cannot manufacture the missing negatives.

**The bottleneck is now label coverage, not feature quality.** The LOO AUC of
0.968 is honest *within* the labelled pool and meaningless outside it.

`data/reports/candidates_2026-08-30.csv` holds a per-camera stratified review
queue (top 25 unlabelled per camera, 417 rows) with a matching
`.message_ids` file for `scripts/label.py --message-ids-file`. Labelling that
queue -- especially the cam07 / cam04 / cam03 / cam13 rows -- is the highest
value next action, because it supplies the negatives those cameras lack.

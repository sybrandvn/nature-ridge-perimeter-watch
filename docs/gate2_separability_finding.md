# Gate 2 — CV feasibility spike: separability finding

Generated from `scripts/spike.py` across every camera with labelled clips
(`data/reports/spike_cam*.csv`). 160 feature rows from 165 labels — 5 clips
produce no detectable motion at all and are dropped, which is correct
behaviour for genuinely blank frames.

Per-camera fence geometry exists only for cam05/cam06/cam08; other cameras get
`far_side_pixel_fraction=0.0` (ambiguous by design, see
`src/zones.py::classify_zone`). Their remaining features are still valid.

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

`green_light_ratio > 0.05` fires on:

- **60/110 guard clips (55%)**
- **1/50 non-guard clips (2%)**
- **0/12 animal or incident clips**

The guard's flashlight reads as a distinct green in this footage, and after the
detector fix it is a clean positive identifier for "this is our guard" with
essentially no leakage into the classes we care about. It doesn't fire on all
guard clips (the light isn't always aimed into frame), so absence proves
nothing — but presence is highly informative.

Note this is a *confirmation* signal, not a suppression one. `docs/plan.md`'s
fail-safe rule stands: nothing here may downgrade a far-side alert.

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
layered on far-side geometry — which is what `docs/plan.md` actually specifies —
it is.

## Flashlight false positives (plan.md requires this count)

On the 3 cameras with fence geometry, 62 labelled `guard` clips:

- **36/62 (58%)** have a far-side-majority pixel fraction — more than half the
  blob lands outside the fence line purely from geometry, despite being a guard
  patrol on the near side.
- **23/36 (64%)** of those carry a light signal (`green_light_ratio > 0.05` or
  `saturation_ratio > 0.15`).

This confirms the plan's predicted failure mode ("a flashlight beam aimed down
or across the fence lands on the far side easily") as real and frequent. The
green-light feature is what makes those recoverable rather than pure noise.

## IR-insect false positives (plan.md requires this count)

**12/22 (55%)** of `environment` clips match the insect signature
(`jitter > 50`, `solidity < 0.85`) — erratic, ragged, low-solidity blobs. Higher
than the 41% the previous detector reported, because background subtraction
picks up the insect tracks the frame-differencing version was missing under the
IR flare. Consistent with the plan's expectation that IR insects are the largest
false-positive source.

## Recommendation

Materially stronger than the first pass, but still one open decision:

1. The green flashlight feature is a genuine result — a 55%-hit, 2%-false-alarm
   guard identifier, with zero leakage into animal/incident. Worth keeping
   regardless of what happens at this gate.
2. Both predicted failure modes are confirmed and quantified (58% flashlight
   far-side rate; 55% of environment clips insect-like).
3. Shape/motion features separate low-crawling subjects from upright ones in the
   right direction and with sensible magnitudes, but 12 positive clips cannot
   support a precision claim. More `animal`/`incident` labels would change
   confidence here more than any further feature work.
4. Whether this passes depends on the bar: usable as an **escalation** signal on
   top of far-side geometry (the design in `docs/plan.md`), not usable as a
   standalone classifier. Confirming that reading is the remaining call.

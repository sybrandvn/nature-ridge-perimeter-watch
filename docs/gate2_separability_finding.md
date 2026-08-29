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

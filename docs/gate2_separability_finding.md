# Gate 2 — CV feasibility spike: separability finding

Generated from `scripts/spike.py` run across every camera with labelled clips
(`data/reports/spike_cam*.csv`, 165 rows = full label set, all 18 cameras).
Per-camera fence geometry is only defined for cam05/cam06/cam08; other cameras
get an uninformative `far_side_pixel_fraction=0.0` (ambiguous, by design —
see `src/zones.py::classify_zone`) but their other features are still valid.

## Label counts

| label | n |
|---|---|
| guard | 111 |
| environment | 22 |
| startup | 11 |
| unknown | 9 |
| incident | 8 |
| animal | 4 |

Positive class for the gate-2 question (`animal` + `incident`) is **12 clips**
against 133 `guard`/`environment` negatives. That is well below what's needed
for a confident precision/recall estimate — treat everything below as
indicative, not confirmatory.

## Headline answer to the gate-2 question

*"Does any threshold combination separate guard and environment from
animal/incident at usable precision?"* — **partially, not cleanly.**

- Single features overlap heavily between positive and negative on this
  sample: `jitter` (pos mean 40.3 vs neg mean 44.6), `path_length` (pos 687 vs
  neg 1024), `row_normalised_area` (huge variance both sides, not usable
  without a log transform and more data).
- The best two-feature rule found (`edge_density <= 0.10 AND solidity >=
  0.80`) gets **recall 58% (7/12), precision 26% (7/27 flagged), false-positive
  rate 15% (20/133)** — roughly 3x the 8% base rate, but still 3 in 4 flagged
  clips would be guard/environment.
- `animal` and `incident` don't resemble each other much on this sample
  (`aspect_ratio` pos values span 0.65-2.67; `persistence` spans 0.05-0.98) —
  treating them as one "positive" cluster may itself be the wrong framing;
  n=4 and n=8 are both too small to tell.

Per the plan's fail-safe design, this is meant to be an **escalation-only**
signal layered on top of far-side geometry (nothing may downgrade or suppress
a far-side alert), not a standalone classifier. Read as escalation lift (26%
precision vs 8% base rate, catching most of the rare positives), it's usable;
read as a general classifier, it is not separable at this label count.

## Flashlight false-positive count (plan.md explicitly requires this)

On the 3 cameras with fence geometry (cam05/cam06/cam08), 62 labelled `guard`
clips:
- **36/62 (58%)** have a far-side-majority pixel fraction — i.e. more than
  half the detected blob's points land on the far side purely from geometry,
  even though the clip is a guard patrol (near side).
- **19/36 (53%) of those** also show a colour/green-light signal
  (`saturation_ratio > 0.15` or the new `green_light_ratio > 0.1`) — directly
  matching the plan's predicted failure mode: "a flashlight beam aimed down or
  across the fence lands on the far side easily." This confirms it's a real,
  frequent effect on this dataset, not a hypothetical.
- The new `green_light_ratio` feature (site-specific: the guard's flashlight
  reads as a distinct green) adds signal beyond generic `saturation_ratio`:
  34% of guard clips clear a combined "sustained green" bar
  (`green_light_ratio > 0.1` and `persistence > 0.2`, to reject single-frame
  flicker) vs 0% of animal/incident and 5%/11% of environment/unknown. Zero
  false positives against the two priority classes on this sample is the
  strongest single result in this spike — worth keeping as a guard-side
  confirmation signal, though it only fires on ~1/3 of guard clips (light not
  always aimed into frame) so it can't carry classification alone.
- Caveat: 3/11 `startup`-labelled clips also show real green-light + motion
  (persistence 0.67-0.83, path_length 80-225px) despite `startup` being
  defined as "short blank clip, not ambiguous, just empty." Worth a manual
  recheck — these may be guard clips that were mislabeled `startup`, or
  genuine edge cases where the guard walks past just as the camera wakes.

## IR-insect false-positive count (plan.md explicitly requires this)

Of the 22 `environment`-labelled clips, **9/22 (41%)** match the expected
insect signature (`jitter > 50`, `solidity < 0.85`, `edge_density < 0.35`) —
consistent with the plan's prediction that IR insects near the lens (erratic,
low-solidity, low-texture blobs) are the largest false-positive source, ahead
of the flashlight.

## Recommendation

Not a clean pass or fail — flagging for a decision rather than declaring one:

1. The flashlight and IR-insect effects the plan worried about are both real
   and now quantified (58% far-side false-positive rate from flashlight
   geometry alone on guard clips; 41% of environment clips look insect-like).
2. `green_light_ratio` is a genuinely useful new discriminator (0% false
   positive rate against animal/incident) but partial coverage (34% of guard
   clips) means it helps rather than solves.
3. The animal/incident positive class (n=12) is too small on this spike
   subset to certify precision/recall numbers — before trusting any threshold
   combination in the real backtester (Phase 3), more animal/incident labels
   would materially change confidence here.
4. Given the project's fail-safe policy (shape features may only escalate,
   never suppress, a far-side alert), the modest lift these features provide
   is consistent with "usable as an escalation signal", which is a lower bar
   than "usable as a classifier" — worth explicitly confirming that's the
   intended reading before treating gate 2 as passed.

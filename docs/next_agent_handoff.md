# Next-agent handoff: detection and event policy

Updated 2026-09-19 on branch `feat/phase2-refactor`.

## Scope

Continue the detection, classification, event-policy, and related refactoring work.

The user will handle **Docker and Telegram integration**. Do not build the live Telegram watcher,
deployment image, container configuration, or notification orchestration unless the user later
asks for it explicitly. The reusable production logic belongs in `src/`; `scripts/` is for
offline analysis and operator tooling.

## Current state

- Working tree was clean when this handoff was written.
- Latest commits:
  - `3155f2d` — keep camera artifacts orthogonal to alerts.
  - `e33ee6d` — recover warmup-only guard evidence.
  - `f3702aa` — repository cleanup and Cam12 trace work; its original middle-era dating was
    corrected by `e33ee6d`.
- `EXTRACTOR_VERSION` is `motion-features-v6`.
- Full verification: **759 tests pass**, Ruff clean.
- Labelled backtest: **708 clips**, TP 29 / FP 26 / FN 18 / TN 635, precision 0.527,
  recall 0.617, F1 0.569.
- The increase from FP 25 to FP 26 is intentional: cam12/9162 is labelled `unknown`, but the
  user's latest review says it may contain a small animal as well as a camera artifact, so it is
  allowed to alert as `animal_candidate`.

Run the normal checks with:

```bash
uv run ruff check .
uv run pytest -q
uv run python scripts/backtest.py \
  --out /tmp/perimeter-labelled.csv --labelled-only --no-record
```

Use `--no-record` for experiments. Bump `src.motion.EXTRACTOR_VERSION` whenever cached feature
semantics change; classifier-only changes do not require an extractor bump.

## Next implementation target

Measure and implement a **completed-sibling event resolver**, offline and independently of
Telegram.

Many camera events produce a short `(Initial)` preview followed by a longer `(Stopped)` or
`(Timeout)` clip. The short preview can lose a guard during IR warmup, then classify a later bush,
pole, noise patch, or camera artifact as outside motion. The longer sibling often identifies the
guard correctly. Confirmed examples:

| short clip | fuller sibling | short result | fuller result |
| --- | --- | --- | --- |
| cam01b/17500 | cam01b/17501 | `animal_candidate` | `guard_candidate` |
| cam07/18641 | cam07/18642 | `incident_candidate` | `guard_candidate` |
| cam03/20521 | cam03/20522 | `incident_candidate` | `guard_candidate` |

Do not simply make “guard wins” or “longest clip wins” the production rule. First measure every
labelled sibling event, including animal and incident events where the short clip may hold the only
useful evidence.

The experiment should compare at least:

1. current `classify_event` behavior, where any animal/incident sibling wins;
2. the fuller/completed sibling alone;
3. a conservative resolver that may let a clearly completed guard result override a weak Initial
   alert while preserving corroborated animal/incident evidence.

Report event-level TP/FP/FN/TN and list every animal or incident event changed by the proposed
policy. Only after that evidence should reusable, transport-independent resolution logic be added
under `src/`. Keep caption parsing and report generation in `scripts/` unless they are truly part
of the production-domain API.

Definition of done:

- the policy is measured on all labelled sibling events;
- every protected animal/incident change is manually inspectable;
- tests cover event order, missing siblings, duplicate Initial clips, and an Initial alert whose
  fuller sibling is guard;
- the per-clip classifier remains usable unchanged;
- no Docker or Telegram work is included.

## Detection behavior that is already settled

### Warmup motion and flashlight

`src/scoring.py` now measures adjacent, photometrically corrected warmup frames through
`warmup_dynamic_frame_fraction` and `warmup_dynamic_outside_fraction`. The flashlight reading is
also independent of the settled-track summary. Five labelled guards that previously returned
`no_motion` are recovered:

- 5464 and 17501 through `warmup_dynamic_inside`;
- 18105, 21491, and 21506 through `warmup_flashlight`.

The no-settled-motion condition is deliberate. Onside warmup activity can coexist with a genuine
outside subject, so warmup evidence must not globally override a later scored track.

### Camera artifacts

Black/white rectangular artifacts are exposed as evidence, not a blanket veto.

- `rectangular_black_white_balance` scans rectangular contours even when they did not win the
  main track.
- A terminal reverse seed plus clipped pixels only routes to guard when there is independent,
  persistent onside warmup movement (`warmup_inside_terminal_artifact`, currently cam03/5465).
- cam12/9162 has artifact evidence but no such warmup corroboration, so it remains alert-worthy.
- cam07/4487 is unclear but explicitly approved to alert.

Do not reintroduce a rule that suppresses a clip merely because an artifact is present.

### Camera/fence movement

`global_camera_shift_score` is reporting-only. It rises on the reviewed wind/camera examples
(cam15/16208 = 2.323, cam15/16564 = 1.319, cam05/18679 = 1.494), but it also rises when a large
real subject dominates the image (guard cam03/20522 = 6.826). It is useful in the debug HUD but is
not safe as an environment gate.

### Cam12 geometry

There are two evidence-backed Cam12 eras, not three:

1. the user's improved trace applies to the entire pre-2026-remount period;
2. the remounted geometry starts at `2026-03-02T16:13:54Z`.

Do not restore the unsupported `2024-11-23` middle era. Clips 8982 and 9161/9162 show the same
pose, and earlier registration put 3457 and 9162 within roughly one pixel.

### Alert-worthy ambiguous footage

Do not tune these away without new user review:

- cam05/9694 and 9695 — possible very small daylight animal;
- cam07/19288 — unclear, but approved to alert;
- cam12/9162 — possible animal/unknown plus artifact;
- cam13/7778 — looks artifact-like, but approved as an incident alert;
- cam07/4487 — unclear, but approved to alert.

cam07/19145 is a bag covering much of the camera. Its incident category and maintenance flag are
intentionally orthogonal.

## Review data

- Refreshed debug set: `data/reports/review_false_alerts_2026-09-19-v2/`
- Label backup before the review:
  `data/backups/labels_pre_false_alert_review_20260919.jsonl`
- Label backup after the review:
  `data/backups/labels_after_false_alert_review_20260919.jsonl`
- Live labels/database: `data/perimeter_watch.db`
- Reference backgrounds: `data/reference_bg/`

These paths are gitignored operational data. Do not delete or reorganize them merely because they
do not appear in `git status`.

## Architecture map

- `src/motion.py` — detector and cached extraction identity.
- `src/scoring.py` — feature calculation from a `ClipDetection`.
- `src/classify.py` — deterministic per-clip classification and reason codes.
- `src/config.py`, `config/thresholds.yaml` — typed classification thresholds.
- `config/cameras.yaml` — dated geometry and camera configuration.
- `scripts/backtest.py` — labelled/full-corpus evaluation harness.
- `scripts/render_debug.py` — visual detector inspection; its HUD shows classification, warmup
  evidence, rectangular artifact evidence, and global shift evidence.
- `scripts/label.py` — label workflow and the current offline `_event_key` helper.
- `docs/handoff.md` — full historical investigation log. Start with session #22; older sections
  are retained for provenance and sometimes explicitly superseded.
- `docs/detection_capability_map.md` — feature inventory and scope.

`tests/test_architecture.py` enforces that `src` does not import `scripts`.

## Optional work, not a blocker

The persistent cache currently stores final feature mappings rather than fully zone-independent raw
tracks. Extracting a compact raw-track payload would allow fence geometry to be rescored without
rerunning detection. The user accepts reprocessing after detector changes, so this refactor is not
required before the event-policy work and should not displace it.

Likewise, do not create global suppression thresholds for storms, bushes, spider webs, rectangular
artifacts, or camera movement without protected-class margin. Their evidence currently overlaps
real animal/incident footage.

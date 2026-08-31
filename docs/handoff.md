# Handoff: gate-2 pass/fail is still open; Phase 1 foundation is now built

Written 2026-08-28, updated 2026-08-29, updated again 2026-08-30 for an agent picking this up
fresh, updated again 2026-08-31 (detection exploratory tools, see below). `docs/plan.md` is the
full plan and stays authoritative; this is the short version of where things actually stand and
what to do next.

## Where the project is

On branch `feat/phase1-foundation` (branched off `main` at the `phase0-checkpoint` tag;
`feat/phase0-foundations` is retired but kept). Working tree clean, 305 tests passing
(`uv run ruff check . && uv run pytest -q`).

Phase 0 is merged to `main` as a checkpoint, not a clean sign-off — see `docs/plan.md`'s
"Checkpoint (2026-08-30)" section. Gate 2 pass/fail is still the one open decision affecting
calibration/threshold work (Phase 3). That did not block Phase 1's foundation scaffolding (steps
15-19: project scaffolding, config/db, label JSONL durability, promoting Phase 0 scripts into
`src/backfill.py`/`src/sequence.py`), which is now done — see "Phase 1 foundation — done
below.

- **Phase 0a (metadata backfill) — done.** 16,887 clips in `data/perimeter_watch.db` with camera,
  timestamp, caption. Camera roster (`cam01`-`cam16`, plus `cam01a`/`cam01b`) is populated in
  `config/cameras.yaml` from real captions.
- **Phase 0b / gate 1 (camera-order inference) — resolved without inference (2026-08-28).**
  User decided fence order is just numerical/alphabetical by camera id; `order` is set directly
  in `config/cameras.yaml` (cam01=0 ... cam16=17). `scripts/infer_camera_order.py` still exists
  for optional later validation but is off the critical path.
- **Phase 0c / gate 2 (CV feasibility spike) — all 6 steps done, gate decision pending user
  sign-off.** Fence geometry was extended from the original 3 spike cameras to all 18 labelled
  cameras (2026-08-28), and the spike was rerun on that wider geometry plus a corrected
  background-subtraction detector (see "Things that will bite you" below). 281 labels rows as of
  2026-08-29 across all 18 cameras, not just the original 3 (see Phase 0c status table).

## Phase 0c status

| Step | State |
| --- | --- |
| 1. Pick spike cameras | done — `cam06` (crawl incident), `cam08` (probe + dusk animal), `cam05` (dusk animal) |
| 2. Download clips | done — 142+ clips on disk across the spike cameras, later downloads extended to more cameras |
| 3. Fence polylines | done — extended from 3 to all 18 labelled cameras (2026-08-28) |
| 4. Hand-label ~150 clips | done, and ongoing past the original milestone — 281 labels rows as of 2026-08-29: 174 with a known event class (guard 126, environment 22, unknown 11, incident 10, animal 5), 121 with only a `startup_state` so far (event class pending its partner clip's review — see schema v5 note below) |
| 5. Run `scripts/spike.py` | done — rerun 2026-08-30 with the corrected zone geometry (see below); `data/reports/spike_all_2026-08-30.csv` is current |
| 6. Read the CSV, decide gate 2 | **written finding done (`docs/gate2_separability_finding.md`), pass/fail call is the user's, not yet made — see staleness note below before signing off** |

## The next task: gate 2 pass/fail

`docs/gate2_separability_finding.md` has the full writeup. Summary of what it found:

**Staleness warning:** the doc's main body (label counts table, green-flashlight rule, shape/motion
table, the 53%/58% flashlight-false-positive count) was written 2026-08-28 on 110 guard clips with
the *old*, direction-of-travel-relative geometry -- it predates both a day of extra labelling
(guard is now 194 rows) and the 2026-08-30 `zones.py` fix. Only "Addendum 2 (2026-08-30)" is
current. A quick re-check on today's data found the coarse outside-majority rate barely moved (51%
vs. the old 53%), but that's not a controlled comparison -- label volume changed too. Rerun the
whole main-body analysis on `spike_all_2026-08-30.csv` before treating any of its specific numbers
(other than Addendum 2's) as the basis for the gate 2 sign-off.

- **Green flashlight is a strong guard identifier:** `green_light_ratio > 0.05 OR
  green_light_flicker > 0.02` catches 61% of guard clips, 2% of non-guard, 0/12 animal+incident.
- **Both predicted failure modes are confirmed and quantified:** 53% of guard clips have an
  outside-majority pixel fraction from geometry alone (flashlight beam crossing the fence line);
  55% of `environment` clips match the IR-insect signature.
- **Shape/motion separates animal/incident from guard/environment in the right direction**
  (aspect ratio, jitter, path length) but only 12 positive (`animal`+`incident`) clips exist
  against 132 negatives — too few to certify precision/recall. Best rule found: 75% recall / 22%
  precision (~2.7x the 8.3% base rate).
- **Recommendation in the doc:** usable as an escalation signal layered on outside geometry (the
  design `docs/plan.md` already specifies), not usable as a standalone classifier. Whether that
  clears the bar for "gate 2 passes" is an explicit call the finding leaves to the user.

Next step for whoever picks this up: get the user's pass/fail call on gate 2. If it passes (as an
escalation signal, per the recommendation), Phase 1 work can start. If not, the plan needs
revisiting before any further implementation.

## Things that will bite you

- **Do not read fence coordinates off an image by eye.** These are 320x240 IR night frames. Three
  separate attempts on `cam05`/`cam06` locked onto a bright diagonal cable instead of the fence,
  including after zooming and overlaying a coordinate grid. What worked: give the user a clean
  upscaled frame, they trace the fence in red in a paint tool, then colour-threshold the red
  pixels back out. Full recipe is in `README.md` section 4 step 3. Reuse it for any further
  cameras.
- **`outside` is plain screen position (fixed 2026-08-30).** `src.zones.side_name` used to be
  direction-of-travel-relative (depended on which way the polyline ran, flipping `left`/`right`
  if point order reversed) -- confusing enough that it was mistaken for a config bug once. It's
  now a plain point-x vs. fence-x-at-that-row comparison; point order no longer matters. Verified
  against real footage that this alone fixed cam06/cam09/cam10's known crawling intruders reading
  as "inside" with zero `cameras.yaml` edits. Still recompute with `src.zones.side_name` against
  a known-exterior point after any edit to the fence points, as a sanity check.
- **Guard/environment clips on cam06/cam09/cam10, and cam01b's 6 resident clips, still read
  `outside_pixel_fraction` inconsistently even after the above fix** -- same clip label, same
  camera, split between reading "inside" and "outside". That's a separate, still-open issue, most
  likely the fence line traced too high (top rail vs. base) rather than anything left/right --
  see the crawl incident's own description below (crouched silhouette at the base of the rail,
  not the rail itself).
- **Timing patterns alone do not identify incidents.** Every purely timing-based incident
  candidate in this repo's history turned out to be a false positive on visual review (see the
  retractions in `README.md` section 4). Always confirm visually or against independent ground
  truth.
- **No migration framework by default.** The documented fallback is: export labels to JSONL,
  delete the DB, reimport. In practice, once `clips` holds real backfilled volume (16,887 rows),
  that's disproportionate for a change scoped to one table -- `scripts/migrate_schema_v5.py` did
  an in-place rebuild instead (rename, recreate from the current schema, copy+transform, bump
  `schema_version`, keep the old table rather than dropping it), run via a raw `sqlite3`
  connection since `db.connect()` enforces an exact version match. Still export JSONL and
  file-copy the live `.db` first regardless, and stop every writer (background downloader AND
  any idle interactive `label.py` session sitting at a prompt) before migrating -- an idle
  session still holds an open db connection.
- **`cam01b`/`cam16` face the opposite way** to the other perimeter cameras: the open ground in
  frame is the interior, not outside. (Corrected 2026-08-28: `cam01`/`cam01a` actually face the
  *same* way as most cameras — the earlier note blaming the whole cam01 family was wrong.) This
  caused a misidentified incident once.
- **`cam15` has real fence geometry and is reversed-mount**, same as `cam01b`/`cam16`: open
  ground in frame is the interior side, not exterior (`config/cameras.yaml` is correct on this —
  `outside: left`). Rough terrain, confirmed animal sighting (2025-07-15, see
  `/memories/repo/incident-findings.md`), no human ever seen there in the sampled history.

## Schema v5 (2026-08-29): labels split into class + startup_state

`labels.label` used to conflate two different things: what the event was, and whether a
short/early pre-alert clip's own content was visible on its own. That collided for real — a
short clip auto-labeled `startup` (or hand-labeled `startup_clear`/`startup_blank`) silently
discarded the event's actual class whenever the paired, decisive clip hadn't been reviewed yet.
Found via a live-db audit: 123 startup-family rows, 107 with their partner still unlabeled.

Now `label` (nullable, 5 real classes) and `startup_state` (`clear`/`blank`/`duplicate`) are
separate columns. Labeling any clip in an event propagates its class to every sibling sharing
the same embedded alert timestamp that doesn't have one yet. `docs/plan.md`'s Ground truth
labels section has the full column semantics; `scripts/migrate_schema_v5.py` is the migration
that ran on the live db (158 rows unchanged, 16 startup-family rows inherited an already-known
partner class, 107 left `label=NULL` pending — those events' partner clips still need review,
which is the bulk of what's left in the current `--message-ids-file` shortlist session).

The frame-match rule for auto-detecting `startup_state=duplicate` was also tightened the same
day: the last decoded frame of a short/early clip is frequently a torn/truncated write from the
recording being cut off (not real content difference) — tolerated now, but only when every
earlier frame is a tight match, checked against a confirmed real incident (cam02#8699) that
fails the same way but with looser earlier-frame differences (moving subject, not a torn frame).

## Known open items, not yet scheduled

- **Fence polyline versioning.** Cameras can shift on their mounts, which moves the fence line in
  frame. `docs/plan.md` (Geometry model, and step 24) documents that `fence`/`outside`/
  `depth_cutoff` need to become a dated per-camera history keyed off clip timestamp. Not needed
  for the spike; required before Phase 2's zone editor and live classification are trustworthy.
- **Compass orientation.** The exterior/outside of the fence is compass south property-wide,
  noted at the top of `config/cameras.yaml`. Captured as documentation only — the per-camera
  `left`/`right` values still do the actual geometry work.
- **Gate 1 (camera order)** resolved 2026-08-28 — numerical/alphabetical by id, no inference run.
- **121 labels rows have only a `startup_state`, no event class yet** (schema v5 migration, see
  above) — their partner clip needs review. `scripts/label.py --message-ids-file` now pulls a
  shortlisted clip's partner into the queue automatically when this applies.
- **Real-time pairing logic is designed but not built.** For the live pipeline (not this Phase 0
  spike): pair a new clip with the last one for the same camera when their embedded alert
  timestamps match exactly (not a fuzzy time guess — the timestamp string is the real key), with
  a ~15min gap as a sanity bound only. Real pair gaps in the db: min 5s, median 3.2min, p95
  4.75min, max 8.6min across 8,274 pairs.

## Phase 1 foundation — done (2026-08-30)

Steps 15-19 of `docs/plan.md`. `src/config.py`/`src/db.py` (config loading, SQLite schema incl.
`blob_tracks`/`backtest_runs`/`backtest_results`, label JSONL export/import) already existed from
the Phase 0 spike, so there was nothing to add there. What was actually built:

- `src/logging_setup.py`: a `JsonFormatter` + `configure_logging()` on the root logger, replacing
  the old per-call-site `logger.info(json.dumps({...}))` pattern in `scripts/meta_backfill.py` and
  `scripts/download_clips.py` (now `logger.info(event, extra={...})`).
- `src/backfill.py`: `run_backfill`/`extract_primitives`/`RawMessage` promoted out of
  `scripts/meta_backfill.py`, which is now a thin Telethon-wiring wrapper around it — matching the
  existing `src/sequence.py` + `scripts/infer_camera_order.py` split (algorithm in `src/`, script
  keeps only I/O glue). `tests/test_meta_backfill.py` renamed to `tests/test_backfill.py`.
- "Console entry points" (from step 15) deliberately not added as real `[project.scripts]`: this
  repo is `uv init --app --no-package` (no `[build-system]`), so installed entry points aren't
  meaningful without converting to a packaged layout, which would contradict the flat-layout
  decision. Each script's own `main()` + `if __name__ == "__main__"` is treated as satisfying it.

Next up is Phase 2 (corpus and CV: camera-ID parsing hardening, resumable full backfill, cached
blob tracks, zone application, rule-engine classifier) — still gated on the user's gate 2
pass/fail call for anything threshold-dependent.

## Detection exploratory tools (2026-08-30/31)

Off the phased plan — a debugging/tuning pass on the spike-2 detector prompted by specific hard
clips (a camouflaged rooikat, a man whose accomplice's hand was drowning him out, a barely-visible
dassie). `scripts/motion_heatmap.py` (new) and `scripts/spike.py` both changed; 305 tests passing.

- **`scripts/motion_heatmap.py`** — standalone CLI, not wired into `spike.py`. Accumulates a
  per-pixel change signal across a whole clip and colorizes it, for judging where a marginal
  subject moved when per-frame detection only shows a dot. Normalises by the 90th percentile of
  *active* pixels, not the frame max — on the two-person clip a hand near the camera peaked ~4x
  the second man's whole body, so max-normalisation discarded 85% of the body's genuinely
  detected signal under the final display gate.
- **`detect_clip` anchor exemplar pass** (`_anchor_exemplar_index`/`_anchor_trace`, on by default
  via `anchor_refine=True`) — picks the real (non-appearance-recovered) detection closest to the
  clip's median tracked size and sweeps that one fixed crop forward and backward, filling frames
  the existing forward/backward passes leave with no box at all. 60 labelled clips: boxless
  frames 36 → 12, box-size jitter and centre-path smoothness unchanged.
- **Checked and rejected, with numbers recorded in the code so they aren't retried:**
  multi-scale template matching (box flickers between per-frame best scales); overwriting
  already-recovered boxes with the anchor exemplar (worse than leaving the drifting per-frame
  template alone); using the motion heatmap as a detection search prior (60-68% of frame lights
  up on vegetation-heavy clips — it's real wind motion, not signal); growing the tracked contour
  via a sliding-window "corroborated by neighbouring frames" test (only 2.2% of motion pixels sit
  within 20px of the tracked box across 60 clips, 47.3% sit >60px away — the existing
  `fragment_close_kernel_size` MORPH_CLOSE already recovers what's reachable).
- Debug clips re-rendered via `scripts/render_debug.py` after all of the above (2026-08-31):
  cam08/7360 (rooikat) and cam08/4053 (man) both track correctly; cam05/18270 (dassie) still only
  a 9px recovered box, expected given how faint its signal is; cam10/4042 correctly shows an IR
  flare frame with 12 spurious multi-tracks scattered across a bush, not a real subject — this is
  the same clip that lit up the motion heatmap and is why it was rejected as a prior.
- Full findings and every rejected variant's measurements: `/memories/repo/nature-ridge-
  conventions.md`.

## Conventions

- `uv` for everything: `uv run ruff check . --fix && uv run pytest -q` after every module change
  and before every commit.
- Commit logically per change, stay on `feat/phase1-foundation`, never auto-merge.
- `src/` is flat, no package. `scripts/` are standalone entry points that `sys.path`-insert the
  repo root.
- DB timestamps are UTC; site local time is UTC+2 (SAST).
- Secrets live in `.env` — never grep or cat it, and never scope a search broadly enough to
  match it.

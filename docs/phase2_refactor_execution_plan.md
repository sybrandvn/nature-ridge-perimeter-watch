# Phase 2 refactor: step-by-step execution plan

Written 2026-09-09 for an agent executing `docs/plan.md`'s "Phase 2 refactor brief". Scope is
deliberately narrow: **plan steps 25 (`classify.py`) and 28 (`backtester.py`), plus wiring the
classification thresholds into `config/thresholds.yaml` (user-directed, 2026-09-09).** Steps 22
(motion caching), 24 (browser zone editor) and the 20/21 backfill audit are OUT OF SCOPE — see
"Out of scope" at the bottom, and do not start them.

Branch: `feat/phase2-refactor`. Stay on it. Never merge to `main`.

**The one rule: classification behaviour must not change.** Every threshold in
`scripts/backtest.py::classify()` was measured against real labelled footage over thirteen
sessions of work (`docs/handoff.md`). This task moves that code and moves those numbers into
config; it does not re-derive them. If you find yourself reasoning about whether a threshold is
*correct*, stop — that is not this task.

---

## Read before starting

1. `docs/plan.md` → "Phase 2 refactor brief (2026-09-09)" — the per-step reality check.
2. `docs/handoff.md` → "Handoff for a new agent (2026-09-09, session close #14)" — current state.
3. `scripts/backtest.py` lines 1-250 — the module docstring. This is the provenance record for
   every threshold you are about to move. Read it, do not summarise it, do not reflow it.

Baseline as measured 2026-09-09: **577 tests passing**, working tree clean, `data/history/` holds
16,886 clips across 18 cameras, 678 of which carry a real label. Extraction runs at ~0.29 s/clip.

### Why the steps are in this order

The extraction (Step 2) is verified by a property that no other step can reproduce: **all 577 tests
pass with zero test edits.** That proof only holds while the code is byte-for-byte unchanged, so
the extraction must happen before `classify()`'s internals are rewired. The config work is
therefore split — the schema and loader are built first (Step 1, a pure addition that nothing
consumes yet, so it cannot change behaviour), and `classify()` only starts reading from them in
Step 5, after it has reached its final home. That way the wiring edit lands once, and a non-empty
corpus diff always points at exactly one change.

---

## DO NOT — read this list twice

1. **The direction of truth is code → YAML, never the reverse.** `config/thresholds.yaml`'s
   current `classification` values are stale and wrong: it says
   `outside_pixel_fraction.alert_min: 0.5` where the real, measured rule is `0.6`, and it carries a
   whole `aspect_ratio` section whose rule was retired 2026-09-04. The hardcoded constants and
   inline numbers in `scripts/backtest.py::classify()` are the validated source of truth. When
   Step 1 writes the new YAML, every value comes from the code. Do **not** "reconcile" a
   disagreement by moving toward the YAML's existing number.
2. **Do not touch `thresholds.yaml`'s `motion:` section.** It feeds
   `ThresholdsConfig.motion_fingerprint()`, which is the cache key for future motion caching
   (plan step 22). Editing it would invalidate caches that do not exist yet and desync a design
   that is deliberately staged. Only the `classification:` section changes.
3. **Do not change any threshold value, comparison operator (`>` vs `>=`), or rule order.** Rule
   order is load-bearing and documented as such (guard before environment before shape). Moving a
   number into YAML must not alter which side of a comparison it sits on.
4. **Do not change `features[...]` to `features.get(...)` or the reverse.** The mix is
   deliberate — `tests/test_backtest.py::test_classify_raises_on_missing_feature_key` asserts that
   a features dict missing a core key raises `KeyError` rather than silently miscategorising.
5. **Move docstrings verbatim.** The `classify()` docstring and the module docstring's rule
   derivations are the only record of how each number was measured. Copy them character-for-
   character. Do not shorten, re-indent, or "tidy" them.
6. **Do not touch** `scripts/spike.py`, `src/features.py`, `src/zones.py`, `detect_clip`, or
   `extract_clip_features`. Not one line of code. (Step 4 updates a few *docstring* cross-
   references in these files; that is the only permitted edit, and none in `spike.py` beyond prose.)
7. **`src/` must never import from `scripts/`.** `src/classify.py` may import `src.config`; it must
   have zero `scripts.*` imports. Verify this holds when you are done.
8. **Never `cat`, `grep`, or otherwise read `.env`**, and never scope a search broadly enough to
   match it. It holds live credentials.
9. **Back up the database before running any migration** (Step 8). Use the repo's existing
   convention: `data/backups/legacy_root/perimeter_watch.db.bak-2026-09-09-pre-v6`.
10. **Do not invent a mapping from `*_candidate` categories to
    `guard_side`/`outside_alert`/`outside_priority`/`ambiguous`.** See Step 8 — this is a routing
    policy decision tied to an unset business criterion, not a refactor.

---

## Conventions (from `docs/handoff.md`)

- `uv` for everything. After every module change and before every commit:
  `uv run ruff check . --fix && uv run pytest -q`
- Commit logically, one commit per numbered step below. Never auto-merge.
- `src/` is flat, no package. `scripts/` are standalone entry points that `sys.path`-insert the
  repo root.
- Scratch output goes in `data/reports/scratch/<name>_<date>/` (gitignored).

---

## Step 0 — Capture the baseline (do this FIRST, before editing anything)

You cannot prove the refactor changed nothing unless you record what "nothing" looked like. This
must run against the *unmodified* code.

```bash
BASE=data/reports/scratch/refactor_baseline_2026-09-09
mkdir -p $BASE
uv run pytest -q | tail -1 | tee $BASE/tests_before.txt
uv run python scripts/backtest.py --labelled-only --out $BASE/labelled_before.csv
uv run python scripts/check_incident_regression.py | tee $BASE/incident_before.txt
```

Expect ~678 rows and roughly 3-5 minutes for the backtest run. Confirm `tests_before.txt` says
`577 passed` and `incident_before.txt` ends in `PASS`. If either is not true, stop and report —
do not refactor on top of a red baseline.

Nothing to commit (`data/reports/` is gitignored).

---

## Step 1 — Config scaffolding: real thresholds in YAML, plus a typed loader

Pure addition. Nothing reads the new config in this step, so behaviour cannot change — which is
exactly why it is safe to do first.

### 1a. Inventory the thresholds

`classify()` holds 6 named constants and 9 inline magic numbers; `is_blinding_foreground()` holds
2 more inline, one of which duplicates `BLINDING_BLOB_WHITE_FRACTION` on purpose. That is **17
comparison sites but 16 distinct keys**, with the operator each one uses — **the operator is not
encoded in the key name, so use this table, not the name, when wiring Step 5:**

| YAML key (under `classification:`) | value | operator and feature as written today |
| --- | --- | --- |
| `guard.green_light_ratio_min` | `0.02` | `features["green_light_ratio"] > x` |
| `guard.green_light_flicker_min` | `0.02` | `features["green_light_flicker"] > x` |
| `guard.warmup_flashlight_ratio_min` | `0.002` | `features.get("warmup_flashlight_ratio", 0.0) > x` |
| `environment.blob_count_peak_min` | `10` | `features["blob_count"] > x` |
| `environment.blob_count_median_min` | `4.0` | `features.get("blob_count_median", 0.0) > x` |
| `environment.implausible_height_fraction_min` | `0.5` | `features.get("implausible_height_fraction", 0.0) > x` |
| `environment.motion_pixel_fraction_median_min` | `0.12` | `features.get("motion_pixel_fraction_median", 0.0) > x` |
| `outside.pixel_fraction_min` | `0.6` | `features["outside_pixel_fraction"] > x` |
| `outside.median_fence_distance_min` | `0.1` | `features["median_fence_distance"] > x` |
| `outside.median_fence_distance_max` | `0.40` | `features["median_fence_distance"] < x` |
| `outside.color_fraction_min` | `0.15` | `features["color_fraction"] > x` |
| `animal.row_normalised_area_max` | `3000.0` | `features.get("row_normalised_area", 0.0) > x` |
| `insect.jitter_min` | `50` | `features["jitter"] > x` |
| `insect.solidity_max` | `0.85` | `features["solidity"] < x` |
| `blinding.blob_white_fraction_min` | `0.4` | `features.get("blob_white_fraction", 0.0) >= x` — **used at two sites** |
| `blinding.long_flare_frames_min` | `18` | `features.get("long_flare_frames", 0.0) >= x` |

Two things this table encodes deliberately:

- `blinding.blob_white_fraction_min` is **one key referenced twice** — by `classify()`'s
  outside-geometry branch and by `is_blinding_foreground()`. The existing code comment is explicit
  that this is intentional ("Same threshold as `is_blinding_foreground()`'s own, deliberately --
  one obstruction definition"). Do not split it into two keys; that would let the two drift.
- `guard.green_light_ratio_min` and `guard.green_light_flicker_min` are both `0.02` but are
  **separate keys** — they are different features that happen to share a value, unlike the case
  above where a single definition is shared on purpose.

These stay hardcoded in `src/classify.py` and must **not** go in YAML — they are structural
presence/flag checks, not tuned thresholds, and putting them in config would invite someone to
"tune" a value that isn't one: `features.get("uncalibrated", 1.0) == 0.0`,
`features.get("zone_classifiable_fraction", 0.0) > 0.0`, `features["outside_pixel_fraction"] == 0.0`.

### 1b. Rewrite `config/thresholds.yaml`'s `classification:` section

Replace the whole `classification:` block with the 16 keys above, in rule-chain order so the file
reads like the classifier. Leave `motion:` untouched (DO-NOT #2). Fix the file's header comment,
which currently claims a state that does not exist yet — after Step 5 it will be true for
`classification`, so word it accordingly.

Every key needs a short comment pointing at where its derivation is recorded, e.g.
`# see src/classify.py's module docstring, "environment_candidate (sustained scattered motion)"`.
Do not copy the full derivation prose into YAML — one pointer per key; the prose stays in one place.

`_validate_numeric_tree` requires every leaf to be int/float/bool, so all 16 values are valid and
nesting is fine.

### 1c. Add the typed loader to `src/config.py`

Add a frozen dataclass and a strict constructor:

```python
@dataclass(frozen=True)
class ClassificationThresholds:
    """Every tuned number classify() compares against, loaded from
    config/thresholds.yaml. Flat on purpose: the YAML nests for readability,
    but a flat record keeps classify()'s call sites short."""
    green_light_ratio_min: float
    green_light_flicker_min: float
    warmup_flashlight_ratio_min: float
    blob_count_peak_min: float
    blob_count_median_min: float
    implausible_height_fraction_min: float
    motion_pixel_fraction_median_min: float
    outside_pixel_fraction_min: float
    median_fence_distance_min: float
    median_fence_distance_max: float
    color_fraction_min: float
    row_normalised_area_max: float
    jitter_min: float
    solidity_max: float
    blob_white_fraction_min: float
    long_flare_frames_min: float
```

Requirements:

- A function that reads the nested YAML shape and **raises `ConfigError` naming the missing path**
  on any absent key. No defaults, no `.get` fallbacks — a typo'd key must fail loudly. This mirrors
  the `KeyError` contract in DO-NOT #4. Note this check necessarily runs when
  `classification_thresholds()` is called, not inside `load_thresholds_config()` itself —
  `load_thresholds_config` has other callers (existing `motion_fingerprint` tests) that load
  minimal YAML with no `classification` section at all, so it cannot require these 16 keys to be
  present. The failure is still loud; it just surfaces at first use of the typed accessor rather
  than at file read.
- Raise `ConfigError` on unrecognised keys under `classification:` too, so a stale key left behind
  from the old file shape is caught rather than silently ignored.
- Add `ThresholdsConfig.classification_thresholds()` returning it, and
  `ThresholdsConfig.classification_fingerprint()` (reusing `_stable_hash`) — Step 9 needs the
  latter for the run record's `thresholds_hash`.

Add to `tests/test_config.py`:

- loads the repo's real `config/thresholds.yaml` and asserts all 16 fields;
- `ConfigError` on a missing key, naming the path;
- `ConfigError` on an unknown key;
- **golden-values test**: the 16 literal values from the table above, hardcoded in the test, with a
  comment that they were measured against real footage and may only be changed alongside a
  re-derivation recorded in `docs/handoff.md`. This is the transcription-error net for Step 5 and
  the thing that stops a future casual YAML edit from quietly moving a validated threshold.
- **coupling test**: `src.features.FLASHLIGHT_CANDIDATE_MIN_RATIO == green_light_ratio_min`.
  `src/features.py:167` sets that constant to `0.02` and its comment says it is deliberately the
  same bar as `GREEN_LIGHT_RATIO_MIN`, "so a candidate is never held to a different standard than a
  finished clip's own `best_contour`". Once the classify side lives in YAML that coupling is
  invisible; this test makes a desync fail immediately. Do not edit `features.py` to import the
  config — that file is out of scope (DO-NOT #6); assert the equality instead.
- `motion_fingerprint()` is **unchanged** by the new `classification` content. Assert this
  explicitly: it is the documented guarantee that "a threshold-only change never invalidates cached
  motion features", and it is what keeps this step safe for plan step 22 later.

Verify (`577 + ~6 passed`) and commit: `feat: real classification thresholds in config, with a typed loader`

---

## Step 2 — Extract `src/classify.py` verbatim

**Success criterion: all tests pass with ZERO edits to any test file.** That is the proof the move
was faithful. If you need to touch a test to make this step pass, you changed behaviour — revert
and redo. Note the constants stay hardcoded here; Step 5 replaces them.

Move these into `src/classify.py`, unchanged:

| what | current location in `scripts/backtest.py` |
| --- | --- |
| module docstring rule derivations (lines 1-250) | see note below |
| `MEDIAN_FENCE_DISTANCE_MAX = 0.40` | ~line 303 |
| `ANIMAL_ROW_AREA_MAX = 3000.0` | ~line 306 |
| `BLOB_COUNT_MEDIAN_MAX = 4.0` | ~line 308 |
| `MOTION_PIXEL_FRACTION_MEDIAN_MAX = 0.12` | ~line 310 |
| `BLINDING_BLOB_WHITE_FRACTION = 0.4` | ~line 313 |
| `GREEN_LIGHT_RATIO_MIN = 0.02` | ~line 322 |
| `classify()` | line 325 |
| `is_blinding_foreground()` | line 378 |
| `_EVENT_CATEGORY_PRIORITY` | line 403 |
| `classify_event()` | line 417 |

On the module docstring: `scripts/backtest.py`'s docstring currently mixes what the *script* does
(screening tool, CLI usage) with how each *classification rule* was derived. Move the rule
derivations to `src/classify.py`'s module docstring verbatim; leave the script/CLI framing in
`scripts/backtest.py` and have it point at `src.classify` for rule provenance.

Everything else stays in `scripts/backtest.py`: `ExtractFn`, `REPORT_COLUMNS`, `_IDENTITY_COLUMNS`,
`_NON_NUMERIC`, `iter_clips_with_files`, `_reference_background`, `run_backtest`, `write_csv`,
`main`.

In this step `src/classify.py`'s only imports are `from __future__ import annotations` and
`from collections.abc import Iterable`.

In `scripts/backtest.py`, replace the moved definitions with a re-export so every existing caller
keeps working untouched. **The `F401` suppression is mandatory, not optional:** after the move
`scripts/backtest.py` calls only `classify` and `is_blinding_foreground` in its own code — the six
constants and `classify_event` are re-exported purely for existing callers, so without the
suppression `uv run ruff check . --fix` will *silently delete them* and the tests will fail
somewhere confusing. Verified against this repo's ruff config 2026-09-09.

```python
# Temporary re-export shim so existing callers keep working; removed in Step 4.
from src.classify import (  # noqa: E402, F401
    ANIMAL_ROW_AREA_MAX,
    BLINDING_BLOB_WHITE_FRACTION,
    BLOB_COUNT_MEDIAN_MAX,
    GREEN_LIGHT_RATIO_MIN,
    MEDIAN_FENCE_DISTANCE_MAX,
    MOTION_PIXEL_FRACTION_MEDIAN_MAX,
    classify,
    classify_event,
    is_blinding_foreground,
)
```

Verify and commit:

```bash
uv run ruff check . --fix && uv run pytest -q
git diff --stat tests/                               # must be EMPTY
git diff scripts/backtest.py | grep -c 'classify'    # sanity: the shim survived --fix
```

Commit: `refactor: extract classify() and friends into src/classify.py`

---

## Step 3 — Prove it byte-identical on the labelled corpus

```bash
BASE=data/reports/scratch/refactor_baseline_2026-09-09
uv run python scripts/backtest.py --labelled-only --out $BASE/labelled_after_step2.csv
diff <(sort $BASE/labelled_before.csv) <(sort $BASE/labelled_after_step2.csv) \
  && echo "IDENTICAL" || echo "DIFFERS -- STOP"
uv run python scripts/check_incident_regression.py | tee $BASE/incident_after_step2.txt
diff $BASE/incident_before.txt $BASE/incident_after_step2.txt && echo "REGRESSION CHECK IDENTICAL"
```

**If the diff is not empty, stop.** Do not proceed, do not "explain" the difference, do not adjust
a threshold to make it match. Report exactly which columns and which clips differ. A non-empty
diff here means the extraction was not faithful and the mistake must be found, not worked around.

Nothing to commit. Record the result in your session notes.

---

## Step 4 — Migrate call sites, split the tests, drop the shim

Update these import sites:

- `scripts/check_incident_regression.py:40` — `from scripts.backtest import classify`
- `scripts/find_storm_events.py:38` — `from scripts.backtest import ExtractFn, classify, iter_clips_with_files`
  (keep `ExtractFn` and `iter_clips_with_files` from `scripts.backtest`; take `classify` from `src.classify`)
- `scripts/rank_candidates.py:48` — `from scripts.backtest import is_blinding_foreground`

Then remove the re-export block from `scripts/backtest.py` (including its `F401` suppression) and
import only what it actually calls — which, verified 2026-09-09, is exactly `classify` and
`is_blinding_foreground`. It calls none of the six constants and does not call `classify_event`;
those appear in its docstrings only, so leave the prose references and drop the imports. Once the
suppression is gone, `ruff check . --fix` deleting an unused import is correct behaviour again.

Split the tests. Move these from `tests/test_backtest.py` into a new `tests/test_classify.py`,
along with the `_features` helper (copy it; `tests/test_backtest.py` still needs it):

- every `test_classify_*` test
- every `test_is_blinding_foreground_*` test
- every `test_classify_event_*` test

Leave in `tests/test_backtest.py`: `test_iter_clips_with_files*`,
`test_run_backtest_assembles_rows_and_skips_unknown_camera`, and the `write_csv` test.

The move is a mechanical rename. **Do not change a single assertion or expected value.** The total
test count must not drop: same tests, different files.

Also update the docstring cross-references that name the old path. Each mentions `scripts.backtest`
where it now means `src.classify`:

- `src/storm_events.py:3`
- `src/features.py:162` ("Same value as `scripts.backtest`'s `GREEN_LIGHT_RATIO_MIN`") — prose only
- `src/zones.py:487`
- `scripts/spike.py:1274` — docstring text only; **do not touch any code in this file**
- `scripts/rank_candidates.py:131, 407, 458, 554`
- `scripts/find_storm_events.py:3`
- `scripts/check_incident_regression.py:3, 16`

Verify, and re-run the Step 3 diff — it must still be IDENTICAL.

Commit: `refactor: point every caller at src.classify, split its tests out`

---

## Step 5 — Config consumption: `classify()` reads its thresholds from the loader

This is the wiring. It lands once, in `src/classify.py`, now that the code is in its final home.

Change the two public entry points to take an optional thresholds object:

```python
def classify(
    features: dict[str, float] | None,
    thresholds: ClassificationThresholds | None = None,
) -> str:
```

and the same for `is_blinding_foreground`. When `thresholds is None`, fall back to a memoised
default loaded from the repo's `config/thresholds.yaml`:

```python
@lru_cache(maxsize=1)
def default_thresholds() -> ClassificationThresholds:
    path = Path(__file__).resolve().parents[1] / "config" / "thresholds.yaml"
    return load_thresholds_config(path).classification_thresholds()
```

**Resolve that path from `__file__`, not from the current working directory.** `scripts/` entry
points `sys.path`-insert the repo root but do not chdir, so a bare relative
`"config/thresholds.yaml"` breaks the moment anything runs from another directory.

The optional-parameter-with-default shape is what keeps every existing test passing unmodified —
`classify(features)` still works — while letting new tests pass an explicit object to vary a
threshold without touching the real file.

Then, branch by branch, replace each hardcoded constant and inline number with the corresponding
field from the table in Step 1a. Work through the table row by row and **keep each comparison
operator exactly as it was** (DO-NOT #3) — the key names say `_min`/`_max`, which does not tell you
whether the site uses `>` or `>=`; only the table does.

Delete the six module-level constants from `src/classify.py` once nothing references them. Keep
their explanatory comments by folding each into the YAML key comment written in Step 1b or into
the module docstring — the derivation prose must survive somewhere, and it must not be duplicated
in two places that can drift.

Add to `tests/test_classify.py`:

- an explicit-thresholds test per rule family, constructing a `ClassificationThresholds` with one
  field moved and asserting the category changes as expected — this is what proves the wiring is
  live rather than the code still using a stale literal;
- a test that `classify(features)` and `classify(features, default_thresholds())` agree.

Verify: full suite green, then **re-run the Step 3 diff — it must still be IDENTICAL.** This is the
step where a single mistyped digit changes real classifications, and that diff plus Step 1c's
golden-values test are the two nets under it. A non-empty diff means a transcription error; find
it, do not accommodate it.

Commit: `refactor: classify() takes its thresholds from config, not hardcoded constants`

---

## Step 6 — Add reason codes

The actual deliverable of plan step 25: today `classify()` returns a bare category string, so
nothing downstream can say *which* rule fired.

In `src/classify.py`:

```python
@dataclass(frozen=True)
class ClassificationResult:
    """One clip's category plus which rule produced it and the feature values
    that rule compared. `category` is byte-identical to what classify() has
    always returned; `reason` and `contributing` are additive."""
    category: str
    reason: str
    contributing: Mapping[str, float]
```

Add `classify_detailed(features, thresholds=None) -> ClassificationResult` holding the if-chain,
and reduce `classify()` to `return classify_detailed(features, thresholds).category`.

**Use exactly these reason codes** — one per return point in the existing chain, in the existing
order. Do not rename, add, or merge any:

| # | reason code | category | `contributing` keys |
| --- | --- | --- | --- |
| 1 | `no_features` | `no_motion` | (empty) |
| 2 | `green_light` | `guard_candidate` | `green_light_ratio`, `green_light_flicker` |
| 3 | `warmup_flashlight` | `guard_candidate` | `warmup_flashlight_ratio` |
| 4 | `blob_count_peak` | `environment_candidate` | `blob_count` |
| 5 | `blob_count_sustained` | `environment_candidate` | `blob_count_median` |
| 6 | `implausible_height` | `environment_candidate` | `uncalibrated`, `implausible_height_fraction` |
| 7 | `blinding_blob_white` | `environment_candidate` | `outside_pixel_fraction`, `median_fence_distance`, `blob_white_fraction` |
| 8 | `motion_pixel_sustained` | `environment_candidate` | `outside_pixel_fraction`, `median_fence_distance`, `motion_pixel_fraction_median` |
| 9 | `animal_row_area` | `environment_candidate` | `outside_pixel_fraction`, `median_fence_distance`, `color_fraction`, `row_normalised_area` |
| 10 | `outside_colour` | `animal_candidate` | `outside_pixel_fraction`, `median_fence_distance`, `color_fraction`, `row_normalised_area` |
| 11 | `outside_no_colour` | `incident_candidate` | `outside_pixel_fraction`, `median_fence_distance`, `color_fraction` |
| 12 | `jitter_solidity` | `insect_candidate` | `jitter`, `solidity` |
| 13 | `inside_only_daylight` | `resident_candidate` | `zone_classifiable_fraction`, `outside_pixel_fraction`, `is_daylight` |
| 14 | `inside_only_night` | `guard_candidate` | `zone_classifiable_fraction`, `outside_pixel_fraction`, `is_daylight` |
| 15 | `no_rule_matched` | `unclassified` | (empty) |

Read `contributing` values with the *same* accessor the branch's condition uses — direct `[...]`
where the condition uses `[...]`, `.get(..., default)` where it uses `.get`. This preserves the
`KeyError` contract (DO-NOT #4).

Add tests covering each of the 15 reason codes and asserting
`classify_detailed(f).category == classify(f)`. Existing tests stay untouched and green.

Verify, re-run the Step 3 diff (still IDENTICAL — `classify()`'s output is unchanged), commit:
`feat: classify_detailed() reports which rule fired and on what`

---

## Step 7 — Surface the reason in backtest rows and the CSV

In `run_backtest`, call `classify_detailed` once per clip and put both values on the row:

```python
result = classify_detailed(features)
row = {..., "category": result.category, "reason": result.reason, ...}
```

Add `"reason"` to `_IDENTITY_COLUMNS` immediately after `"category"`.

This is the one step whose output legitimately differs from the baseline: the CSV gains a column.
Verify by comparing every *pre-existing* column instead:

```bash
BASE=data/reports/scratch/refactor_baseline_2026-09-09
uv run python scripts/backtest.py --labelled-only --out $BASE/labelled_after_step7.csv
uv run python - <<'PY'
import csv
base = "data/reports/scratch/refactor_baseline_2026-09-09"
before = {(r["channel_id"], r["message_id"]): r for r in csv.DictReader(open(f"{base}/labelled_before.csv"))}
after = {(r["channel_id"], r["message_id"]): r for r in csv.DictReader(open(f"{base}/labelled_after_step7.csv"))}
assert before.keys() == after.keys(), "row set changed -- STOP"
cols = [c for c in next(iter(before.values())) if c != "reason"]
bad = [(k, c, before[k][c], after[k][c]) for k in before for c in cols if before[k][c] != after[k][c]]
print("MISMATCHES:", len(bad))
for row in bad[:20]:
    print(row)
PY
```

`MISMATCHES: 0` is required. Anything else, stop and report.

Commit: `feat: record which rule fired in the backtest CSV`

---

## Step 8 — Schema v6: let `backtest_results` hold the real categories

**Read this whole step before touching anything. The brief calls step 28 "the cheapest win"; that
assessment is incomplete and this is why.**

`src/db.py` already has the `backtest_runs`/`backtest_results` schema plus `create_run`,
`finish_run`, `add_backtest_result`, `iter_backtest_results`, `iter_backtest_runs` — written,
unit-tested, and called by nothing (all three tables are empty; verified 2026-09-09). But
`backtest_results.predicted_class` carries a CHECK constraint accepting only:

```
'guard_side', 'outside_alert', 'outside_priority', 'ambiguous'
```

`classify()` emits none of those. It emits `guard_candidate`, `animal_candidate`,
`incident_candidate`, `environment_candidate`, `insect_candidate`, `resident_candidate`,
`unclassified`, `no_motion`. The four-class vocabulary is the *planned* Phase 3 routing output
(`docs/plan.md` step 25, `README.md:251`) which was never built; the eight `*_candidate` categories
are the empirically-derived vocabulary that actually exists.

**Widen the constraint; do not map the categories.** Collapsing eight measured categories into
four routing classes would encode an alerting policy nobody has validated — and choosing it
depends on Ship readiness criterion #3 in `docs/plan.md`, which is explicitly still unset and is a
business call, not an engineering one. Storing the real category verbatim keeps the run records
honest and leaves that decision where it belongs.

1. In `src/db.py`: extend `VALID_PREDICTIONS` to include the eight real categories *in addition
   to* the existing four (keep the four — `tests/test_db.py` uses them and the Phase 3 vocabulary
   may still arrive). Extend the `CHECK (predicted_class IN (...))` list in `_SCHEMA_SQL` to match,
   exactly. Bump `SCHEMA_VERSION` from `5` to `6`.
2. Write `scripts/migrate_schema_v6.py`, following `scripts/migrate_schema_v5.py`'s structure and
   docstring style. SQLite cannot alter a CHECK constraint in place, and `backtest_results` is
   **empty**, so the migration is a drop-and-recreate of that one table plus a `schema_version`
   update. It must:
   - refuse to run if `backtest_results` has any rows (print how many, exit non-zero) — that
     assumption is what makes drop-and-recreate safe, so verify it rather than trusting this doc;
   - touch nothing else — not `clips`, `labels`, `system_events`, `blob_tracks`, `backtest_runs`;
   - be idempotent: if `schema_version` is already 6, print that and exit 0.
3. Add a `tests/test_db.py` case asserting a `*_candidate` value is accepted and a junk value still
   raises `DbError`.
4. Back up and migrate the live database:

```bash
cp data/perimeter_watch.db data/backups/legacy_root/perimeter_watch.db.bak-2026-09-09-pre-v6
uv run python scripts/migrate_schema_v6.py
uv run python scripts/check_incident_regression.py | tail -3   # proves the db still opens
```

`db.connect` hard-errors on a `schema_version` mismatch, so until the migration is run every tool
in the repo will refuse to open the database. That is the version guard working as designed — but
it means this step is not finished until the migration has actually been applied.

Commit: `feat: schema v6 -- backtest_results accepts the real categories`

---

## Step 9 — `src/backtester.py`: immutable run records

Create `src/backtester.py` wiring the existing `src/db.py` functions in. No new DB functions.

```python
def record_run(
    conn,
    rows: Iterable[dict],
    *,
    cameras_path: str = "config/cameras.yaml",
    thresholds_path: str = "config/thresholds.yaml",
    run_id: str | None = None,
    notes: str | None = None,
) -> str:
    """Persist one backtest pass as an immutable run. Returns the run_id."""
```

Behaviour:

- `run_id` defaults to `datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")`. Raise a clear error rather
  than silently colliding if that id already exists.
- `git_revision`: `git rev-parse HEAD` via `subprocess.run`, `None` on any failure (missing git, not
  a repo, non-zero exit). Never let this raise — a run record must not depend on git being present.
- `cameras_hash`: `sha256` of the file's bytes, truncated to 16 hex chars (matching
  `src/config.py::_stable_hash`'s width).
- `thresholds_hash`: use `ThresholdsConfig.classification_fingerprint()` from Step 1c. Because
  Step 5 made `config/thresholds.yaml` the real source of the classifier's numbers, hashing it is
  now honest — a threshold change genuinely changes the hash. Use the `classification` fingerprint
  specifically, not a whole-file hash: a `motion:` edit does not affect classification results, and
  conflating the two would make identical classification runs look different.
- Wrap the insert loop: on success `finish_run(status="completed")`; on any exception
  `finish_run(status="failed")` and re-raise. A crashed run must leave a `failed` record, not a
  dangling `running` one.
- Per row: `predicted_class=row["category"]`, `reason_codes=[row["reason"]]` (the column is
  `reason_codes_json`, so a list — this is why Step 6 comes first), and `features` = the row minus
  the identity columns, keeping `label` and `blinding_foreground` (they are what makes a stored run
  analysable later).

Wire it into `scripts/backtest.py::main()`: after `write_csv`, record the run and print the
`run_id`. Add a `--no-record` flag to skip it. Record by default — a run nobody can find again is
the exact problem step 28 exists to fix. Keep the CSV output unchanged; this is additive.

Add `tests/test_backtester.py` against a `tmp_path` database: a completed run's row count and
`status`, a failing insert leaving `status='failed'`, `git_revision=None` tolerated, and a
duplicate `run_id` rejected.

Commit: `feat: src/backtester.py records immutable backtest runs`

---

## Step 10 — Final verification and documentation

```bash
uv run ruff check . --fix && uv run pytest -q
uv run python scripts/check_incident_regression.py
uv run python scripts/backtest.py --labelled-only \
  --out data/reports/scratch/refactor_baseline_2026-09-09/labelled_final.csv
uv run python -c "
from src import db; from src.config import load_app_config
conn = db.connect(load_app_config(require_telegram=False).db_path)
for r in db.iter_backtest_runs(conn):
    print(dict(r))
"
```

Required: tests green; regression check `PASS` with the same event count as `incident_before.txt`;
the Step 7 column comparison still `MISMATCHES: 0`; exactly one `completed` run with a populated
`git_revision` and both config hashes.

Optional final gate, ~80 minutes at 0.29 s/clip — worth it given that Step 5 moved every threshold:
the same comparison across all 16,886 clips, without `--labelled-only`.

Then update the docs, following the existing style in each:

- `docs/plan.md` — mark steps 25 and 28 done in the refactor-brief table. Record two decisions and
  their reasons: `predicted_class` was widened rather than mapped, and the `classification`
  thresholds now live in `config/thresholds.yaml` (so step 25's "classify.py never hardcodes a
  number" is finally true, while `motion:` is still unconsumed and waits on step 22).
- `docs/handoff.md` — add `## Handoff for a new agent (<date>, session close #15)` above the other
  handoff sections and update the top-of-file pointer to it. Record: what moved; that the
  labelled-corpus output was verified byte-identical at Steps 3, 5 and 6; that the old
  `thresholds.yaml` values were stale and were replaced *from the code*, not reconciled toward;
  the `features.py` coupling and the test that now guards it; the schema v6 migration and that it
  must be run on any other copy of the database; and that step 22 is next and is the risky one.

Commit: `docs: session #15 -- Phase 2 steps 25 and 28 landed, thresholds moved to config`

---

## Out of scope — do not start these

- **Step 22, `motion.py` caching.** The riskiest item in the brief, deliberately left last.
  `detect_clip` takes 22 tuning keyword arguments and carries several sessions of hard-won
  correctness fixes; a cache key that misses even one will silently serve stale results. Two
  findings for whoever picks it up: `ThresholdsConfig.motion_fingerprint()` and the `blob_tracks`
  table (with its `extractor_version` + `motion_fingerprint` composite key) already exist and are
  tested, so the cache *machinery* is further along than the brief implies — but nothing defines an
  `EXTRACTOR_VERSION`, and `motion_fingerprint()` hashes `thresholds.yaml`'s `motion:` section,
  which is **not** where `detect_clip`'s real parameters live (they are defaults in its signature).
  This plan deliberately did not touch that: wiring `motion:` up is the same class of work Step 5
  did for `classification:`, but against a function with 22 parameters and a silent-staleness
  failure mode instead of 16 values and a loud one. It needs its own session and its own
  full-corpus verification.
- **Step 24, browser zone editor.** Standalone feature work, not a refactor. Lowest priority.
- **Steps 20/21 backfill/parsing audit.** A separate verification track; nothing has broken.
- **Setting Ship readiness criterion #3** or turning `prefer_flashlight_candidate` on by default.
  Business decisions for the user.

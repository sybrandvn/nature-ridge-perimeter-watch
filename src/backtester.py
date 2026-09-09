"""Wires scripts/backtest.py's screening pass into the immutable run records
`src/db.py` already has -- `backtest_runs`/`backtest_results` plus
create_run/finish_run/add_backtest_result/iter_backtest_results/
iter_backtest_runs, written and unit-tested but called by nothing until this
module (plan.md step 28, 2026-09-09). No new DB functions; this file is purely
the plumbing between one `run_backtest()` pass and those existing functions.

Before this, every real backtest run was a throwaway CSV -- nothing recorded
which config produced it, so two runs could never be reliably compared or
reproduced. `record_run()` persists one pass as a single immutable row plus
one result row per clip.
"""

from __future__ import annotations

import hashlib
import sqlite3
import subprocess
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src import db
from src.config import load_thresholds_config
from src.errors import DbError

# Columns scripts.backtest.run_backtest() always puts on a row that are NOT
# part of the feature vector -- everything else in the row is a feature,
# including "label" and "blinding_foreground" (kept in `features` on purpose:
# they are what makes a stored run analysable later, see record_run()).
_NON_FEATURE_COLUMNS = frozenset({"channel_id", "message_id", "camera_id", "category", "reason"})

# Resolved from __file__, not a bare relative string: this is an internal
# detail of what record_run() hashes, not a caller-supplied parameter, and
# should not depend on the caller's working directory the way cameras_path
# (a real parameter, matching scripts/backtest.py's own cwd-relative
# defaults) is allowed to.
_REPO_THRESHOLDS_PATH = Path(__file__).resolve().parents[1] / "config" / "thresholds.yaml"


def _git_revision() -> str | None:
    """`git rev-parse HEAD`, or None on any failure. A run record must never
    depend on git being present -- missing git, not a repo, a non-zero exit,
    all fall through to None rather than raising."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def _file_hash(path: str) -> str:
    """sha256 of the file's bytes, truncated to 16 hex chars -- same width as
    src.config's own _stable_hash, so a cameras_hash and a
    classification_fingerprint read the same way in a run record."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def record_run(
    conn: Any,
    rows: Iterable[dict[str, Any]],
    *,
    cameras_path: str = "config/cameras.yaml",
    thresholds_path: str = "src/classify.py",
    run_id: str | None = None,
    notes: str | None = None,
) -> str:
    """Persist one backtest pass (`scripts.backtest.run_backtest()`'s output)
    as an immutable run. Returns the run_id.

    `thresholds_path` defaults to `src/classify.py`, not
    `config/thresholds.yaml`, and `thresholds_hash` is computed from
    `ThresholdsConfig.classification_fingerprint()` rather than a whole-file
    hash of that path -- deliberately: `config/thresholds.yaml`'s
    `classification` section is where the numbers actually live (plan.md step
    25, wired 2026-09-09), but `src/classify.py` is where the rule chain that
    reads them lives, and `thresholds_path` here means "what governs this
    run's classification behaviour", which is the code, not just its config.
    Using the classification-only fingerprint (not a hash of the whole yaml
    file) also means a `motion:` edit -- which does not affect classification
    at all -- never changes what this run records, matching
    `classification_fingerprint()`'s own contract.

    `run_id` defaults to a UTC timestamp; passing one that collides with an
    existing run raises DbError rather than silently overwriting.

    On any exception partway through, the run is marked `failed` (not left
    `running`) and the exception re-raised.
    """
    if run_id is None:
        run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")

    cameras_hash = _file_hash(cameras_path)
    thresholds_hash = load_thresholds_config(_REPO_THRESHOLDS_PATH).classification_fingerprint()

    try:
        db.create_run(
            conn,
            run_id=run_id,
            thresholds_path=thresholds_path,
            thresholds_hash=thresholds_hash,
            cameras_path=cameras_path,
            cameras_hash=cameras_hash,
            git_revision=_git_revision(),
            notes=notes,
        )
    except sqlite3.IntegrityError as exc:
        raise DbError(f"backtest run {run_id!r} already exists") from exc

    try:
        for row in rows:
            features = {k: v for k, v in row.items() if k not in _NON_FEATURE_COLUMNS}
            db.add_backtest_result(
                conn,
                run_id=run_id,
                channel_id=row["channel_id"],
                message_id=row["message_id"],
                camera_id=row["camera_id"],
                predicted_class=row["category"],
                reason_codes=[row["reason"]],
                features=features,
            )
    except Exception:
        db.finish_run(conn, run_id, status="failed")
        raise
    db.finish_run(conn, run_id, status="completed")
    return run_id

"""Scored predictions to a comparable metrics record on disk.

One JSON per scored run, identical in shape every time. That is the whole
feature: Phase 03 writes the rules baseline and logistic regression, Phase 05
the model, Phase 06 adds USD — and comparing them is a read rather than a
reconciliation.

A run also leaves its **score vector** on disk, because a summary cannot be
re-plotted: a PR curve needs every threshold, not three capacities. ``write_run``
writes both from one frame, so the figure and the record are two views of the
same numbers rather than two computations that happen to agree.

No plotting here. Figures live in ``plots.py``, which owns every matplotlib
import, so this module stays importable anywhere the package is installed —
including the Phase 08 container, which has no reason to carry a plotting
library.
"""

from __future__ import annotations

import json
import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from fraud_engine.evaluation.metrics import evaluate

# TEST is not here, and that is the point. The invariant is not "never score
# test" — measurement is unlimited — but nothing may *change* because of what it
# shows. Scoring it has to be a deliberate keystroke rather than a default, so
# that "we were still iterating" and "test was scored" cannot quietly overlap.
# The keys of the written report record which splits a run actually touched.
DEFAULT_SPLITS = ("val_fit", "val_cal")

REQUIRED_COLUMNS = ("isFraud", "score", "day", "split")

# What a predictions parquet carries. TransactionID is the join key across runs —
# without it two runs cannot be checked for having scored the same rows, and a
# comparison figure would silently overlay curves drawn on different populations.
PREDICTION_COLUMNS = ("TransactionID", "split", "day", "isFraud", "score")

# Names become filenames. Anything outside this cannot escape metrics_dir.
_SAFE_NAME = re.compile(r"^[A-Za-z0-9_-]+$")


def git_revision() -> str | None:
    """The commit these numbers came from, marked if the tree was dirty.

    Phase 09 compares runs across weeks; "which code produced this" stops being
    obvious well before then. A ``-dirty`` suffix matters more than the sha: a
    number produced from uncommitted code cannot be reproduced from the history.

    ``reports/`` is excluded from the dirty check. It is where this function's
    own caller writes, so including it makes every stage mark itself dirty for
    having an output — the artifact is untracked until the run that produces it
    is committed, which can only happen after the run. The question the suffix
    answers is whether the *code* is in history; the state of the output
    directory has no bearing on that. Untracked source still flags dirty.

    Returns:
        ``"<sha>"``, ``"<sha>-dirty"``, or None outside a git checkout.
    """
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--", ":!reports"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None

    return f"{sha}-dirty" if dirty else sha


def load_capacities(cost_matrix: dict) -> list[float]:
    """Review capacities to report, from the parsed ``cost_matrix.yaml``.

    Takes the parsed mapping rather than a path so the policy is testable
    without a file.

    The headline capacity must appear in the sweep. It is the one §5 states the
    success criteria against, and a report that omits it cannot answer the
    question the project was set up to ask.

    Args:
        cost_matrix: Parsed ``config/cost_matrix.yaml``.

    Returns:
        The sweep values, ascending.

    Raises:
        ValueError: If the headline capacity is absent from the sweep.
    """
    headline = cost_matrix["constraints"]["review_capacity"]["value"]
    values = [float(value) for value in cost_matrix["sensitivity"]["review_capacity"]["values"]]

    if float(headline) not in values:
        raise ValueError(
            f"the headline review capacity ({headline}) is not among the sweep values "
            f"({values}). Every report would omit the operating point the success "
            f"criteria are stated against."
        )

    return sorted(values)


def load_operating_capacity(cost_matrix: dict) -> float:
    """The committed review capacity — the one operating point, not the sweep.

    ``load_capacities`` returns the range Phase 06 tests the conclusion across;
    this is the single value §3.2 commits to and the figures mark. Separate
    functions because confusing the two turns a sensitivity range into a claim
    about what the team can actually staff.

    Args:
        cost_matrix: Parsed ``config/cost_matrix.yaml``.

    Returns:
        The capacity, as a fraction of daily transaction volume.
    """
    return float(cost_matrix["constraints"]["review_capacity"]["value"])


def evaluate_splits(
    frame: pd.DataFrame,
    capacities: list[float],
    splits: tuple[str, ...] = DEFAULT_SPLITS,
) -> dict[str, dict]:
    """Run the full metric set over each requested split.

    Args:
        frame: Scored rows, carrying ``isFraud``, ``score``, ``day`` and
            ``split``. Rows outside ``splits`` — the purge gap, and test unless
            asked for — are ignored.
        capacities: Review capacities, as fractions of daily volume.
        splits: Which splits to score. Defaults to validation only.

    Returns:
        ``{split_name: evaluate(...)}``, in the order requested.

    Raises:
        ValueError: If a column is missing, or a requested split holds no rows.
    """
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"frame is missing {missing}; needs {list(REQUIRED_COLUMNS)}.")

    if not splits:
        raise ValueError("no splits requested; there would be nothing to report.")

    scored = {}
    for split in splits:
        # An empty slice means a typo'd split name, and would otherwise produce
        # a report that is structurally valid and silently about nothing.
        rows = frame[frame["split"] == split]
        if rows.empty:
            raise ValueError(
                f"split {split!r} holds no rows. Present in the frame: "
                f"{sorted(value for value in frame['split'].dropna().unique())}."
            )

        rows = rows.reset_index(drop=True)
        scored[split] = evaluate(rows["isFraud"], rows["score"], rows["day"], capacities)

    return scored


def build_report(
    name: str,
    frame: pd.DataFrame,
    capacities: list[float],
    splits: tuple[str, ...] = DEFAULT_SPLITS,
) -> dict:
    """Assemble one run's metrics record.

    Args:
        name: Identifier for this run — becomes the filename. Letters, digits,
            underscores and hyphens only.
        frame: Scored rows. See ``evaluate_splits``.
        capacities: Review capacities to report.
        splits: Which splits to score. Defaults to validation only.

    Returns:
        A JSON-serialisable record: ``name``, ``created``, ``git_revision``,
        ``capacities``, and ``splits``. The keys of ``splits`` are the audit
        trail of what this run touched.

    Raises:
        ValueError: If ``name`` is unusable as a filename, or per
            ``evaluate_splits``.
    """
    if not _SAFE_NAME.match(name):
        raise ValueError(
            f"name {name!r} must be letters, digits, underscores or hyphens — it "
            f"becomes a filename under metrics_dir."
        )

    return {
        "name": name,
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_revision": git_revision(),
        "capacities": list(capacities),
        "splits": evaluate_splits(frame, capacities, splits),
    }


def write_report(report: dict, metrics_dir: Path | str) -> Path:
    """Write a report to ``metrics_dir/<name>.json``.

    Indented and newline-terminated because ``reports/`` is tracked: these
    land in git history, and a one-line JSON blob diffs uselessly.

    Args:
        report: From ``build_report``.
        metrics_dir: Directory to write into. Created if absent.

    Returns:
        The path written.
    """
    path = Path(metrics_dir) / f"{report['name']}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    return path


def write_predictions(
    name: str,
    frame: pd.DataFrame,
    predictions_dir: Path | str,
    splits: tuple[str, ...] = DEFAULT_SPLITS,
) -> Path:
    """Write a run's score vector to ``predictions_dir/<name>.parquet``.

    The metrics record answers "how good was this run"; this answers "what did
    it actually say", which is what a PR curve needs — every threshold, not
    three capacities. Derived data, so it lives under ``data/`` and is
    gitignored: the tracked artifacts are the record and the figures drawn from
    this.

    **Filtered to ``splits``, exactly as the report is.** The same gate that
    keeps test out of the metrics record keeps it out of this file, so a stage
    reading predictions cannot plot a split the run was not asked to score. It
    is the invariant made structural rather than repeated.

    Args:
        name: Run identifier, matching the metrics record. Becomes the filename.
        frame: Scored rows, carrying ``PREDICTION_COLUMNS``.
        predictions_dir: Directory to write into. Created if absent.
        splits: Which splits to keep. Defaults to validation only.

    Returns:
        The path written.

    Raises:
        ValueError: If ``name`` is unusable as a filename, a column is missing,
            or no row survives the split filter.
    """
    if not _SAFE_NAME.match(name):
        raise ValueError(
            f"name {name!r} must be letters, digits, underscores or hyphens — it "
            f"becomes a filename under predictions_dir."
        )

    missing = [column for column in PREDICTION_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"frame is missing {missing}; needs {list(PREDICTION_COLUMNS)}.")

    kept = frame.loc[frame["split"].isin(splits), list(PREDICTION_COLUMNS)]
    if kept.empty:
        raise ValueError(
            f"no rows in splits {list(splits)}. Present in the frame: "
            f"{sorted(value for value in frame['split'].dropna().unique())}."
        )

    path = Path(predictions_dir) / f"{name}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    kept.reset_index(drop=True).to_parquet(path, index=False)
    return path


def write_run(
    name: str,
    frame: pd.DataFrame,
    capacities: list[float],
    metrics_dir: Path | str,
    predictions_dir: Path | str,
    splits: tuple[str, ...] = DEFAULT_SPLITS,
) -> tuple[Path, Path]:
    """Write both of a run's artifacts from one scored frame.

    One call rather than two so the record and the scores it summarises cannot
    come from different computations. A caller that writes only the report
    leaves a run that can be compared numerically but never plotted; one that
    writes them separately can let them drift. Neither is reachable from here.

    Args:
        name: Run identifier. Names both files.
        frame: Scored rows, carrying ``REQUIRED_COLUMNS`` and
            ``PREDICTION_COLUMNS``.
        capacities: Review capacities to report.
        metrics_dir: Where the JSON record goes.
        predictions_dir: Where the score vector goes.
        splits: Which splits to score and keep. Defaults to validation only.

    Returns:
        ``(metrics_path, predictions_path)``.

    Raises:
        ValueError: Per ``build_report`` and ``write_predictions``.
    """
    metrics_path = write_report(build_report(name, frame, capacities, splits), metrics_dir)
    predictions_path = write_predictions(name, frame, predictions_dir, splits)
    return metrics_path, predictions_path

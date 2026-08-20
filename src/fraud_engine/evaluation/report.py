"""Scored predictions to a comparable metrics record on disk.

One JSON per scored run, identical in shape every time. That is the whole
feature: Phase 03 writes the rules baseline and logistic regression, Phase 05
the model, Phase 06 adds USD — and comparing them is a read rather than a
reconciliation.

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

# Names become filenames. Anything outside this cannot escape metrics_dir.
_SAFE_NAME = re.compile(r"^[A-Za-z0-9_-]+$")


def git_revision() -> str | None:
    """The commit these numbers came from, marked if the tree was dirty.

    Phase 09 compares runs across weeks; "which code produced this" stops being
    obvious well before then. A ``-dirty`` suffix matters more than the sha: a
    number produced from uncommitted code cannot be reproduced from the history.

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
            ["git", "status", "--porcelain"],
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

"""What the label-maturity gap costs — E1, at three arms.

The purge is the most defensible decision in this project and the one with no
number behind it. Removing it hands a model two advantages production never has,
and E1 measures them apart rather than together: a run trained up to the
validation boundary has *recency*, and one trained over the vacated days has
recency plus *volume*. A retrain cadence can partly buy the first. Nothing buys
the second, because those labels could not exist yet.

**Arms are built beside the shipped pipeline, never over it.** Each runs the
split and feature stages against a config whose outputs are redirected into a
working directory. Editing the shipped config in place and restoring it after
would leave every downstream stage one interruption away from reading a split
nobody chose.
"""

from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger(__name__)

# Every path the split and feature stages write. Named rather than derived from
# the config, because the failure this guards is an arm overwriting the shipped
# artifact it is being compared against — and a stage that starts writing
# somewhere new has to be added here deliberately for that to keep holding.
REDIRECTED = ("splits", "split_summary", "features_dir", "encoders", "amount_stats", "vblock")

# Read and never written, and deliberately left where it is. It is pre-split, so
# every arm reads the same rows and differs only in how they are labelled —
# which is what makes the arms comparable at all.
SHARED = "interim"


def redirect(config: dict, directory: Path | str, *, train_start: int, gap_days: int) -> dict:
    """The shipped config, pointed at one arm's window and one arm's output.

    Two blocks move and nothing else. `splits` gets the arm's window, which is
    the whole experimental variable — `resolve_boundaries` derives every other
    boundary from it, so an arm is two integers rather than a table of days.
    `paths` gets every written location rewritten under `directory`, keeping
    each file's own name so an arm's directory reads like a small copy of the
    project.

    The input is not modified. A caller holding the shipped config after
    building three arms still holds the shipped config.

    Args:
        config: The loaded `config.yaml`.
        directory: Where this arm's artifacts go. Not created here — the stages
            that write into it do that.
        train_start: First training day.
        gap_days: Purged days between train and `VAL-FIT`. Zero runs unpurged.

    Returns:
        A new config, shallow-copied except for the two blocks that change.

    Raises:
        ValueError: If a redirected path is missing from config, or if two of
            them would land on the same name. Either one leaves a stage writing
            where the shipped pipeline writes, and the arm would be measured
            against an artifact it had just overwritten.
    """
    directory = Path(directory)

    missing = [key for key in REDIRECTED if key not in config["paths"]]
    if missing:
        raise ValueError(
            f"paths to redirect are absent from config: {missing}; "
            "an arm would write where the shipped pipeline writes"
        )

    paths = {**config["paths"]}
    for key in REDIRECTED:
        paths[key] = str(directory / Path(paths[key]).name)

    landed = [paths[key] for key in REDIRECTED]
    if len(set(landed)) != len(landed):
        raise ValueError(
            f"redirected paths collide under {directory}: {sorted(landed)}; "
            "two stages would write the same file"
        )

    return {
        **config,
        "paths": paths,
        "splits": {**config["splits"], "train_start": train_start, "gap_days": gap_days},
    }

"""The monitoring window — one definition for PSI, PR-AUC and the score distribution.

`docs/monitoring.md` §2. Those three are read side by side, and a PSI figure beside a
PR-AUC figure means nothing if the two were cut on different spans.

A window is a calendar span of fixed width, anchored at the first day of its horizon.
A trailing span shorter than the width is kept and marked partial rather than dropped:
it holds the most recent days, which a decay reading needs most, and the retraining
trigger must not fire on it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _checked(days: pd.Series | np.ndarray, width: int, anchor: int | None) -> tuple:
    """Days as integers, and the anchor they count from.

    Raises:
        ValueError: On a width below one, no days, a null or fractional day, or an
            anchor after the first day — a row before the anchor has no window.
    """
    if width < 1:
        raise ValueError(f"a window needs a width of at least one day, got {width}")

    values = np.asarray(days, dtype="float64")
    if not values.size:
        raise ValueError("no days to cut into windows")
    if np.isnan(values).any():
        raise ValueError("a row has no day; it cannot be placed in a window")
    if (values != np.floor(values)).any():
        raise ValueError("days must be whole: a window boundary falls between days, not inside one")

    values = values.astype("int64")
    first = int(values.min())
    anchor = first if anchor is None else anchor
    if anchor > first:
        raise ValueError(f"anchor {anchor} is after the first day {first}; those rows fall outside")
    return values, anchor


def window_index(days: pd.Series | np.ndarray, width: int, anchor: int | None = None) -> np.ndarray:
    """Which window each row falls in, counting from zero at the anchor.

    Args:
        days: One day per row.
        width: Days per window.
        anchor: The first day of window zero. Defaults to the earliest day present,
            which is the first day of the horizon the rows belong to.

    Returns:
        An integer window per row, in the input's order.
    """
    values, anchor = _checked(days, width, anchor)
    return (values - anchor) // width


def describe(days: pd.Series | np.ndarray, width: int, anchor: int | None = None) -> pd.DataFrame:
    """One row per window, in order, whether or not any transaction fell in it.

    A window is the span of the calendar, not of the data: a day with no rows does not
    shorten it. Only the trailing window can be partial, and it ends on the last day
    present, because that is where the horizon ends.

    Args:
        days: One day per row.
        width: Days per window.
        anchor: As for `window_index`.

    Returns:
        Columns `window`, `first_day`, `last_day`, `days` (calendar days spanned),
        `rows` and `partial`.
    """
    values, anchor = _checked(days, width, anchor)
    last = int(values.max())
    count = (last - anchor) // width + 1

    first_days = anchor + width * np.arange(count)
    last_days = np.minimum(first_days + width - 1, last)
    spans = last_days - first_days + 1

    return pd.DataFrame(
        {
            "window": np.arange(count),
            "first_day": first_days,
            "last_day": last_days,
            "days": spans,
            "rows": np.bincount((values - anchor) // width, minlength=count),
            "partial": spans < width,
        }
    )

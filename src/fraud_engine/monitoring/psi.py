"""Population stability index, and the bins it is counted in.

`docs/monitoring.md` §3. Bins are cut once, on the reference, and every window is
counted into the same ones; a window binned on its own quantiles would be measured
with a different ruler each time.

**Missing is always a bin of its own.** Filling first would turn the most common real
drift — a field that stops arriving — into no drift at all.

**Categorical columns are not binned here.** The caller routes them through
`train.apply_categories`, the function the booster's inputs pass through, so the bins
are the distinctions the model can make; this module only counts them. Importing it
here would hand every stage that computes a PSI the config sections training reads.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def quantile_edges(reference: pd.Series, bins: int) -> np.ndarray:
    """Interior cut points from the reference's non-null values, with ties merged.

    Value bins are right-closed — `(-inf, e0]`, `(e0, e1]`, …, `(e_last, inf)` — so
    there are `len(edges) + 1` of them. A column whose mass sits on a few values
    shares quantiles, and merging them leaves it fewer bins than asked for.

    Every edge is a value the reference holds: the empirical quantile, the smallest
    value whose share at or below it reaches `k / bins`. Interpolating would put an
    edge between two tied values of a count column — 0.4, between a run of zeros and
    a run of ones — and open a bin nothing can ever land in. Positions are found in
    integers, because a level like 0.3 held as a float lands one row late.

    Args:
        reference: The column in the reference window.
        bins: Value bins wanted.

    Returns:
        Sorted, unique edges. Empty when the reference has no values at all, which
        leaves one value bin.

    Raises:
        ValueError: If `bins` is below one, the column is categorical, or the
            reference holds an infinity — a quantile interpolated across one is
            `nan`, and every comparison against `nan` is false.
    """
    if bins < 1:
        raise ValueError(f"need at least one bin, got {bins}")
    if isinstance(reference.dtype, pd.CategoricalDtype):
        raise ValueError(f"{reference.name!r} is categorical; bin it with apply_categories")

    values = reference.dropna().to_numpy(dtype="float64")
    if not values.size:
        return np.empty(0)
    if not np.isfinite(values).all():
        raise ValueError(f"{reference.name!r} holds an infinity; its quantiles are undefined")

    ordered = np.sort(values)
    k = np.arange(1, bins)
    return np.unique(ordered[-(-k * len(ordered) // bins) - 1])  # ceil(k·n / bins) - 1


def numeric_counts(values: pd.Series, edges: np.ndarray) -> np.ndarray:
    """Rows per value bin, then the missing rows, last.

    The missing bin is always present, empty or not, so a reference with no nulls and
    a window full of them are counted on the same bins.

    Args:
        values: The column in one window.
        edges: From `quantile_edges` on the reference.

    Returns:
        `len(edges) + 2` counts.

    Raises:
        ValueError: If the column is categorical.
    """
    if isinstance(values.dtype, pd.CategoricalDtype):
        raise ValueError(f"{values.name!r} is categorical; bin it with apply_categories")

    missing = values.isna().to_numpy()
    present = values.to_numpy(dtype="float64", na_value=np.nan)[~missing]
    # side="left" puts a value equal to an edge in the bin that edge closes.
    counts = np.bincount(np.searchsorted(edges, present, side="left"), minlength=len(edges) + 1)
    return np.append(counts, missing.sum())


def categorical_counts(values: pd.Series) -> np.ndarray:
    """Rows per level, in the vocabulary's order, zeros included.

    Args:
        values: The column after `apply_categories`, carrying the shipped vocabulary
            as its categories.

    Returns:
        One count per level, sentinels included.

    Raises:
        ValueError: If the column is not categorical, or holds a null — which means
            it never passed through `apply_categories`, where a null becomes
            `MISSING` before the membership test.
    """
    if not isinstance(values.dtype, pd.CategoricalDtype):
        raise ValueError(f"{values.name!r} is not categorical; route it through apply_categories")
    if values.isna().any():
        raise ValueError(
            f"{values.name!r} holds nulls, so it was not routed through apply_categories; "
            "counted as is, its missing rows would vanish from every bin"
        )
    return np.bincount(values.cat.codes.to_numpy(), minlength=len(values.cat.categories))


def psi(reference: np.ndarray, window: np.ndarray, epsilon: float) -> float:
    """The population stability index of a window against its reference.

    `sum((a - e) * ln(a / e))` over bins, on shares. `epsilon` is added to both
    shares of every bin, so an empty bin has a finite logarithm and a bin empty on
    both sides contributes exactly nothing. Every term is non-negative, because
    `a - e` and `ln(a / e)` share a sign.

    Args:
        reference: Counts per bin in the reference.
        window: Counts per bin in the window, on the same bins.
        epsilon: `monitoring.psi.epsilon`.

    Returns:
        The index; zero when the two distributions agree bin for bin.

    Raises:
        ValueError: On bins that do not line up, a negative count, an empty side,
            or an epsilon that is not positive.
    """
    reference = np.asarray(reference, dtype="float64")
    window = np.asarray(window, dtype="float64")

    if reference.shape != window.shape:
        raise ValueError(f"bins do not line up: {reference.shape} against {window.shape}")
    if (reference < 0).any() or (window < 0).any():
        raise ValueError("a bin holds a negative count")
    if not reference.sum() or not window.sum():
        raise ValueError("one side holds no rows; its shares are undefined")
    if epsilon <= 0:
        raise ValueError(f"epsilon must be positive, got {epsilon}")

    expected = reference / reference.sum() + epsilon
    actual = window / window.sum() + epsilon
    return float(np.sum((actual - expected) * np.log(actual / expected)))

"""The drift figure, in the house style `evaluation/plots.py` defines.

Its own module rather than a function in `monitoring/plots.py`, which the decay chart
depends on: adding a figure there would mark a recorded stage stale for a figure it does
not draw — the reason that module was split from `evaluation/plots.py` in the first place.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from fraud_engine.evaluation.plots import INK, INK_MUTED, SERIES_COLOURS, SURFACE, style_axes

# Days are plotted from the end of the training window, so the thirty the files do not
# cover stay visible as a gap: points only, never a line across it.
TRAINING_END = 90
HORIZONS = {
    "labelled": ("labelled, days 121-182", SERIES_COLOURS[0]),
    "horizon": ("unlabelled horizon, days 213-395", SERIES_COLOURS[1]),
}


def _x(entry: dict) -> float:
    return (entry["first_day"] + entry["last_day"]) / 2 - TRAINING_END


def _scatter(axes, entries: list[dict], value, colour: str) -> None:
    for entry in entries:
        axes.plot(
            _x(entry),
            value(entry),
            "o",
            color=colour,
            markerfacecolor=SURFACE if entry["partial"] else colour,
            markeredgecolor=colour,
            markersize=5,
        )


def _reference(axes, value: float, text: str, style: str = "--") -> None:
    axes.axhline(value, color=INK_MUTED, linewidth=1, linestyle=style)
    axes.annotate(
        text,
        xy=(1.0, value),
        xycoords=("axes fraction", "data"),
        xytext=(4, 0),
        textcoords="offset points",
        va="center",
        ha="left",
        fontsize=8,
        color=INK_MUTED,
    )


def plot_drift(record: dict) -> plt.Figure:
    """Feature drift on both horizons, and the prediction drift the unlabelled one shows.

    Four panels on one horizontal axis, never a second y-scale: the contribution-weighted
    PSI §7's first condition reads, with the convention's bands; the score PSI against
    test; the mean calibrated probability; and the frozen policy's block rate — the last
    two against test's own value.

    Args:
        record: `drift.json`, as `drift.main` writes it.

    Returns:
        The figure, unsaved. Use `save_figure`.
    """
    psi_cfg, anchor = record["psi"], record["score_reference"]
    labelled, horizon = record["labelled"], record["horizon"]

    figure, (weighted, scores, mean, blocked) = plt.subplots(
        4, 1, figsize=(7.0, 9.0), dpi=150, sharex=True, height_ratios=(3, 2, 1.6, 1.6)
    )
    figure.patch.set_facecolor(SURFACE)
    for axes in (weighted, scores, mean, blocked):
        style_axes(axes)

    for name, entries in (("labelled", labelled), ("horizon", horizon)):
        _scatter(weighted, entries, lambda e: e["weighted_psi"], HORIZONS[name][1])
    for axes in (weighted, scores):
        _reference(axes, psi_cfg["moderate"], "moderate", ":")
        _reference(axes, psi_cfg["significant"], "significant", ":")
    low, high = 0, weighted.get_ylim()[1]
    weighted.set_ylim(low, high * 1.15)
    for name, entries in (("labelled", labelled), ("horizon", horizon)):
        centre = sum(_x(entry) for entry in entries) / len(entries)
        weighted.annotate(
            HORIZONS[name][0].split(",")[0],
            xy=(centre, 0.97),
            xycoords=("data", "axes fraction"),
            ha="center",
            va="top",
            fontsize=9,
            color=INK_MUTED,
        )
    weighted.set_ylabel("weighted PSI\nvs train", color=INK_MUTED, fontsize=9)
    weighted.set_title(
        "Drift without labels: inputs against train, scores against test",
        color=INK,
        fontsize=11,
        loc="left",
        pad=12,
    )

    colour = HORIZONS["horizon"][1]
    _scatter(scores, horizon, lambda e: e["prediction"]["score_psi"], colour)
    scores.set_ylim(0, None)
    scores.set_ylabel("score PSI\nvs test", color=INK_MUTED, fontsize=9)

    _scatter(mean, horizon, lambda e: 100 * e["prediction"]["mean_calibrated"], colour)
    _reference(mean, 100 * anchor["mean_calibrated"], "test")
    mean.set_ylabel("mean predicted\nfraud, %", color=INK_MUTED, fontsize=9)

    _scatter(blocked, horizon, lambda e: 100 * e["prediction"]["block_rate"], colour)
    _reference(blocked, 100 * anchor["block_rate"], "test")
    blocked.set_ylabel("EV policy\nblocks, %", color=INK_MUTED, fontsize=9)
    blocked.set_xlabel("days since the training window ended", color=INK_MUTED, fontsize=9)

    handles = [
        Line2D([], [], marker="o", linestyle="", color=c, markersize=5) for _, c in HORIZONS.values()
    ] + [
        Line2D(
            [], [], marker="o", linestyle="", markerfacecolor=SURFACE,
            markeredgecolor=INK_MUTED, markersize=5,
        )
    ]  # fmt: skip
    figure.tight_layout(rect=(0, 0.035, 1, 1))
    figure.legend(
        handles,
        [label for label, _ in HORIZONS.values()] + ["partial window"],
        loc="lower center",
        ncol=len(handles),
        fontsize=8,
        frameon=False,
        labelcolor=INK_MUTED,
    )
    return figure

"""The reliability diagram, drawn from the out-of-fold probabilities on disk.

A stage of its own, like `figures.py`: figures are redrawn from persisted
vectors, never from a fit, and `calibrate.py` stays free of matplotlib because
Phase 08 applies the calibrator it defines.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from fraud_engine.data.load import DEFAULT_CONFIG_PATH, load_config
from fraud_engine.evaluation.plots import plot_reliability, save_figure
from fraud_engine.models.calibrate import METHODS, OUT_OF_FOLD, reliability_bins
from fraud_engine.models.train import LABEL

log = logging.getLogger(__name__)

# Named in the README once it is written; changing it here means changing it there.
FIGURE = "reliability_val_cal.png"


def main(config_path: Path = DEFAULT_CONFIG_PATH) -> None:
    """Draw uncalibrated and both out-of-fold calibrations on one diagram.

    Wiring only. Invoked by `make calibrate` as `python -m fraud_engine.evaluation.reliability`.

    Args:
        config_path: Path to `config.yaml`.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    config = load_config(config_path)
    paths = config["paths"]
    bins = config["calibration"]["ece_bins"]

    frame = pd.read_parquet(Path(paths["predictions_dir"]) / f"{OUT_OF_FOLD}.parquet")
    selected = json.loads(Path(paths["calibration"]).read_text())["selected"]

    y = frame[LABEL].to_numpy()
    tables = {"uncalibrated": reliability_bins(y, frame["score"].to_numpy(), bins)}
    for method in METHODS:
        label = f"{method}, out-of-fold" + (" (selected)" if method == selected else "")
        tables[label] = reliability_bins(y, frame[method].to_numpy(), bins)

    path = save_figure(
        plot_reliability(tables, title="Reliability — VAL-CAL"),
        Path(paths["figures_dir"]) / FIGURE,
    )
    log.info("wrote %s", path)


if __name__ == "__main__":
    main()

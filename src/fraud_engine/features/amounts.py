"""Amount transforms — the row-local feature family.

Every column here is a function of one transaction's own amount and product. No
entity, no history, no training window, and nothing that a single scoring
request cannot supply.

The family exists to **test H2**, not to add features. `docs/hypotheses.md`
predicts that a "$50 multiple within $150-$500" feature earns importance while a
plain "is a round number" feature does not, and names `ProductCD` H as where the
effect concentrates. All three columns ship so that prediction can fail: drop the
falsifier and the claim stops being falsifiable.

Two things are deliberately absent. A log of the amount, because the probe's own
`amount` branch already applies one — adding it here would measure nothing. And
any decimal-places feature: on train days 1-90 the 3-decimal group carries a
3.17x lift, but 100% of those rows are `ProductCD` C and the lift within C is
1.01x. It is the same proxy H3 found sitting behind address-presence, and the
probe already carries `ProductCD`.
"""

from __future__ import annotations

import pandas as pd

# H2's falsifier and H2's claim, plus the interaction H2 names. The falsifier is
# expected to be worthless; that is the result, not a reason to leave it out.
WHOLE_DOLLAR = "amt_whole_dollar"
ROUND_BAND = "amt_round_band"
ROUND_BAND_PRODUCT = "amt_round_band_product"

COLUMNS = (WHOLE_DOLLAR, ROUND_BAND, ROUND_BAND_PRODUCT)

# The smallest unit TransactionAmt carries. Every test in this module is a
# divisibility question, and floats cannot answer those.
UNITS_PER_DOLLAR = 1000


def thousandths(amount: pd.Series) -> pd.Series:
    """``TransactionAmt`` as a whole number of thousandths of a dollar.

    Asking a float whether it divides evenly is asking the wrong question:
    ``49.99`` is stored as ``49.989999999999995`` and ``0.1 + 0.2`` is not
    ``0.3``. Converting once to the smallest unit the data actually carries
    makes every test below integer arithmetic, rather than leaving each call
    site to pick its own tolerance.

    Safe to cast without a null guard: ``validate.py`` declares
    ``TransactionAmt`` non-nullable and positive, and that schema raises inside
    the pipeline rather than being eyeballed.

    Args:
        amount: ``TransactionAmt``, in USD.

    Returns:
        The amount in thousandths, as ``int64``.
    """
    return (amount * UNITS_PER_DOLLAR).round().astype("int64")


def is_whole_dollar(amount: pd.Series) -> pd.Series:
    """Roundness with no band — H2's falsifier.

    H2 predicts this carries nothing on its own, and measurement on train agrees:
    54.1% of rows, 1.02x lift. A plain round-number feature that *did* carry the
    signal would make the band incidental and H2 wrong, which is exactly why it
    ships rather than being left out for scoring poorly.
    """
    return thousandths(amount) % UNITS_PER_DOLLAR == 0


def is_round_in_band(amount: pd.Series, step: int, low: int, high: int) -> pd.Series:
    """A multiple of ``step`` dollars, between ``low`` and ``high`` inclusive.

    H2's claim. On train: 5.3% of rows at 2.26x lift, against 1.20x for "$50
    multiple" with no band and 1.02x for roundness alone — the band is doing the
    work, not the roundness.

    Args:
        amount: ``TransactionAmt``, in USD.
        step: Multiple to test for, in whole dollars.
        low: Lower bound, inclusive, in whole dollars.
        high: Upper bound, inclusive, in whole dollars.
    """
    units = thousandths(amount)
    return (
        (units % (step * UNITS_PER_DOLLAR) == 0)
        & (units >= low * UNITS_PER_DOLLAR)
        & (units <= high * UNITS_PER_DOLLAR)
    )


def add_amount_features(frame: pd.DataFrame, amounts_cfg: dict) -> pd.DataFrame:
    """The amount family, appended to a copy of ``frame``.

    Null-free by construction: every column is a divisibility test on a
    non-nullable amount, so the probe's median imputer never fires on them. That
    matters more than it looks — a family that arrives with nulls inherits a
    fill decision made by the evaluation harness rather than by whoever knows
    what "missing" means for that feature.

    ``int8`` rather than ``bool``: both reach the numeric branch fine, but a
    boolean column that later acquires a null silently becomes ``object``.

    Args:
        frame: Rows carrying ``TransactionAmt`` and ``ProductCD``.
        amounts_cfg: The ``features.amounts`` config block.

    Returns:
        A copy of ``frame`` with ``COLUMNS`` added.
    """
    amount = frame["TransactionAmt"]
    round_band = is_round_in_band(
        amount,
        amounts_cfg["round_step"],
        amounts_cfg["round_min"],
        amounts_cfg["round_max"],
    )
    # The interaction H2 actually claims. A linear probe reads roundness and
    # product as separate terms and cannot see their product on its own.
    in_product = frame["ProductCD"] == amounts_cfg["round_product"]

    return frame.assign(
        **{
            WHOLE_DOLLAR: is_whole_dollar(amount).astype("int8"),
            ROUND_BAND: round_band.astype("int8"),
            ROUND_BAND_PRODUCT: (round_band & in_product).astype("int8"),
        }
    )

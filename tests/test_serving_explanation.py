"""Tests for the notice a declined customer is read, and where it is produced.

`docs/serving.md` §3. The endpoint exists apart from `/score` because of an arithmetic
result, not a preference — so what is tested here is that it stays apart, that it refuses
rather than inventing when it cannot answer, and that the serving path still imports
nothing it promised not to.
"""

import subprocess
import sys

import numpy as np
import pytest

from fraud_engine.serving.explanation import ADDITIVITY_TOLERANCE, check_additive

# Imported for their side effect on `sys.modules` inside a fresh interpreter — see below.
FORBIDDEN = ("shap", "matplotlib", "fraud_engine.features.registry")


def test_contributions_that_decompose_the_prediction_are_accepted():
    contributions = np.array([0.5, -0.2, 0.1, -1.4])

    check_additive(contributions, contributions.sum())


def test_an_explanation_of_the_wrong_row_is_refused():
    """What this catches is a true sentence about somebody else's transaction."""
    contributions = np.array([0.5, -0.2, 0.1, -1.4])

    with pytest.raises(ValueError, match="not an explanation"):
        check_additive(contributions, contributions.sum() + 1.0)


def test_the_tolerance_is_relative_to_the_mass_being_summed():
    """A wrong-object detector, not a precision claim — `explainability.md` §2."""
    large = np.array([50.0, -49.0, 3.0])
    drift = float(np.abs(large).sum()) * ADDITIVITY_TOLERANCE / 2

    check_additive(large, large.sum() + drift)


def test_the_serving_path_imports_none_of_what_it_promised_not_to():
    """In a fresh interpreter: another test having imported `shap` would hide this.

    The registry is on the list for the reason `explain/codes.py` gives — the reason-code
    dictionary is the tier boundary at request time, and the registry guards the dictionary
    in the tests instead, where the built matrices exist.
    """
    program = (
        "import sys; import fraud_engine.serving.app;"
        f"print([name for name in {FORBIDDEN!r} if name in sys.modules])"
    )
    result = subprocess.run(
        [sys.executable, "-c", program], capture_output=True, text=True, check=True
    )

    assert result.stdout.strip() == "[]", f"the serving path pulled in {result.stdout.strip()}"

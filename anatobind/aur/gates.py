"""Pure model-training gates shared by training scripts and tests."""

import math
from numbers import Real


def _finite_metric(value):
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def q25_step1000_gate(readout, a_dice):
    """Return the fixed Q25 step-1000 decision; invalid metrics fail closed."""
    readout_value = _finite_metric(readout)
    a_dice_value = _finite_metric(a_dice)

    readout_passed = readout_value is not None and readout_value >= 0.65
    a_dice_passed = a_dice_value is not None and a_dice_value > 0.045
    failures = []
    if readout_value is None:
        failures.append("readout_missing_or_non_finite")
    elif not readout_passed:
        failures.append("readout_below_minimum")
    if a_dice_value is None:
        failures.append("a_dice_missing_or_non_finite")
    elif not a_dice_passed:
        failures.append("a_dice_not_above_minimum")

    return {
        "passed": bool(readout_passed and a_dice_passed),
        "readout_passed": bool(readout_passed),
        "a_dice_passed": bool(a_dice_passed),
        "failures": failures,
        "thresholds": {
            "readout_min": 0.65,
            "a_dice_min_exclusive": 0.045,
        },
    }

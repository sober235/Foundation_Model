"""Pure model-training gates shared by training scripts and tests."""

import json
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


def build_q25_step1000_report(
    g1_report,
    val_jsonl,
    *,
    checkpoint_exists,
    probe_error=None,
    input_errors=(),
):
    """Build a JSON-safe durable report from the Q25 probe and validation artifacts."""
    failures = []
    readout = None
    a_dice = None

    if not checkpoint_exists:
        failures.append("resume_checkpoint_missing")
    if probe_error:
        failures.append("probe_failed")
    failures.extend(str(error) for error in input_errors)

    if not isinstance(g1_report, dict):
        failures.append("invalid_g1_report")
    else:
        g1 = g1_report.get("g1")
        if not isinstance(g1, dict) or "host_macro_ssl" not in g1:
            failures.append("invalid_g1_report")
        else:
            readout = g1["host_macro_ssl"]

    rows = []
    if not isinstance(val_jsonl, str):
        failures.append("validation_jsonl_missing")
    else:
        try:
            rows = [json.loads(line) for line in val_jsonl.splitlines() if line.strip()]
            if any(not isinstance(row, dict) for row in rows):
                raise ValueError("validation rows must be objects")
        except (json.JSONDecodeError, ValueError):
            rows = []
            failures.append("validation_jsonl_invalid")

    step_rows = [row for row in rows if row.get("step") == 1000]
    if not step_rows:
        failures.append("validation_step_1000_missing")
    elif len(step_rows) > 1:
        failures.append("validation_step_1000_ambiguous")
    elif "a_dice" not in step_rows[0]:
        failures.append("validation_a_dice_missing")
    else:
        a_dice = step_rows[0]["a_dice"]

    gate = q25_step1000_gate(readout, a_dice)
    failures.extend(gate["failures"])
    # Preserve first occurrence while avoiding duplicate gate/input failures.
    failures = list(dict.fromkeys(failures))

    return {
        "schema_version": 1,
        "passed": bool(gate["passed"] and not failures),
        "readout_passed": gate["readout_passed"],
        "a_dice_passed": gate["a_dice_passed"],
        "metrics": {
            "frozen_readout": _finite_metric(readout),
            "a_dice": _finite_metric(a_dice),
        },
        "thresholds": gate["thresholds"],
        "failures": failures,
    }

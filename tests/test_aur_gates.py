import math

import pytest

from anatobind.aur.gates import q25_step1000_gate


def test_q25_gate_passes_inclusive_readout_and_strict_a_dice_boundary():
    result = q25_step1000_gate(0.65, 0.0451)

    assert result["passed"] is True
    assert result["readout_passed"] is True
    assert result["a_dice_passed"] is True
    assert result["failures"] == []
    assert result["thresholds"] == {
        "readout_min": 0.65,
        "a_dice_min_exclusive": 0.045,
    }


@pytest.mark.parametrize(
    ("readout", "a_dice", "expected_failures"),
    [
        (0.65, 0.045, {"a_dice"}),
        (0.6499, 0.046, {"readout"}),
        (None, 0.046, {"readout"}),
        (float("nan"), 0.046, {"readout"}),
        (0.7, float("inf"), {"a_dice"}),
        ("bad", 0.046, {"readout"}),
    ],
)
def test_q25_gate_fails_closed_for_boundary_and_invalid_metrics(readout, a_dice, expected_failures):
    result = q25_step1000_gate(readout, a_dice)

    assert result["passed"] is False
    assert result["failures"]
    if "readout" in expected_failures:
        assert result["readout_passed"] is False
    if "a_dice" in expected_failures:
        assert result["a_dice_passed"] is False


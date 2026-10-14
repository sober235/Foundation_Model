import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

from anatobind.aur.gates import build_q25_step1000_report, q25_step1000_gate


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


def test_q25_report_serializes_a_valid_pass():
    report = build_q25_step1000_report(
        {"g1": {"host_macro_ssl": 0.65}},
        '{"step":1000,"a_dice":0.0451}\n',
        checkpoint_exists=True,
    )

    assert report["passed"] is True
    assert report["metrics"] == {"frozen_readout": 0.65, "a_dice": 0.0451}
    assert report["failures"] == []
    assert report["thresholds"] == {
        "readout_min": 0.65,
        "a_dice_min_exclusive": 0.045,
    }
    assert isinstance(json.dumps(report), str)


def test_q25_report_fails_when_required_probe_input_is_missing():
    report = build_q25_step1000_report(
        {"g1": {"host_macro_ssl": 0.65}},
        '{"step":1000,"a_dice":0.046}\n',
        checkpoint_exists=True,
        input_errors=["validation_manifest_missing"],
    )

    assert report["passed"] is False
    assert "validation_manifest_missing" in report["failures"]


@pytest.mark.parametrize(
    ("g1_report", "val_jsonl", "checkpoint_exists", "probe_error", "failure"),
    [
        ({"g1": {"host_macro_ssl": 0.65}}, '{"step":1000,"a_dice":0.046}\n', False, None, "resume_checkpoint_missing"),
        ({"g1": {"host_macro_ssl": 0.65}}, '{"step":500,"a_dice":0.046}\n', True, None, "validation_step_1000_missing"),
        ({"wrong": {}}, '{"step":1000,"a_dice":0.046}\n', True, None, "invalid_g1_report"),
        ({"g1": {"host_macro_ssl": math.nan}}, '{"step":1000,"a_dice":Infinity}\n', True, None, "readout_missing_or_non_finite"),
        ({"g1": {"host_macro_ssl": 0.65}}, '{"step":1000,"a_dice":0.046}\n', True, "probe failed", "probe_failed"),
    ],
)
def test_q25_report_serializes_every_invalid_state_as_failure(
    g1_report, val_jsonl, checkpoint_exists, probe_error, failure
):
    report = build_q25_step1000_report(
        g1_report,
        val_jsonl,
        checkpoint_exists=checkpoint_exists,
        probe_error=probe_error,
    )

    assert report["passed"] is False
    assert failure in report["failures"]
    assert isinstance(json.dumps(report), str)


# The checker script itself (Task 2): every failure path exits non-zero, writes its verdict under the run, and never
# reaches the GPU probe. The PASS path needs the real probe; its decision is the pure function tested above.
CHECKER = Path(__file__).resolve().parents[1] / "docs/verification/2026-10-09/anatobind_brain_ssl_first/ops/c2v2_check.sh"


def _stage2_run(tmp_path):
    run = tmp_path / "stage2"
    run.mkdir()
    (run / "val.jsonl").write_text(json.dumps({"step": 1000, "a_dice": 0.05}) + "\n")
    (tmp_path / "samples.json").write_text("[]")
    (tmp_path / "val_patients.json").write_text("[]")
    env = {"STAGE2_DIR": str(run), "SAMPLES": str(tmp_path / "samples.json"),
           "VAL_PATIENTS": str(tmp_path / "val_patients.json"), "PROBE_GPU": "none"}
    return run, env


def _run_checker(env, drop=()):
    full = {k: v for k, v in os.environ.items() if k not in ("STAGE2_DIR", "SAMPLES", "VAL_PATIENTS", "PROBE_GPU")}
    full.update(env, PY=sys.executable, CUDA_VISIBLE_DEVICES="")
    for key in drop:
        full.pop(key)
    return subprocess.run(["bash", str(CHECKER)], env=full, capture_output=True, text=True, timeout=120)


def _report(run):
    return json.loads((run / "gate_step1000" / "report.json").read_text())


def test_q25_checker_fails_closed_without_the_step1000_checkpoint(tmp_path):
    run, env = _stage2_run(tmp_path)

    result = _run_checker(env)

    assert result.returncode != 0
    assert _report(run)["passed"] is False
    assert "resume_checkpoint_missing" in _report(run)["failures"]
    assert not (run / "gate_step1000" / "probe").exists()


def test_q25_checker_fails_closed_when_an_input_is_missing(tmp_path):
    run, env = _stage2_run(tmp_path)
    (run / "resume_step1000.pt").write_bytes(b"placeholder")

    result = _run_checker(env, drop=("VAL_PATIENTS",))

    assert result.returncode != 0
    assert "env_missing:VAL_PATIENTS" in _report(run)["failures"]
    assert not (run / "gate_step1000" / "probe").exists()


def test_q25_checker_fails_closed_when_the_checkpoint_cannot_be_read(tmp_path):
    run, env = _stage2_run(tmp_path)
    (run / "resume_step1000.pt").write_bytes(b"not a torch checkpoint")

    result = _run_checker(env)

    assert result.returncode != 0
    assert "probe_failed" in _report(run)["failures"]
    assert not (run / "gate_step1000" / "probe").exists()


def test_q25_checker_never_overwrites_an_earlier_verdict(tmp_path):
    run, env = _stage2_run(tmp_path)
    (run / "gate_step1000").mkdir()
    (run / "gate_step1000" / "report.json").write_text('{"passed": true}')

    result = _run_checker(env)

    assert result.returncode != 0
    assert _report(run) == {"passed": True}

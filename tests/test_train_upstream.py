import json

import pytest

torch = pytest.importorskip("torch")

from anatobind.train.train_upstream import lr_lambda, main


def test_the_schedule_warms_up_linearly_then_decays_to_zero():
    f = lr_lambda(warmup=10, total=110)
    assert f(0) == pytest.approx(0.1) and f(9) == pytest.approx(1.0)
    assert f(10) == pytest.approx(1.0) and f(60) == pytest.approx(0.5) and f(110) == pytest.approx(0.0)


def _args(export, cache, out, steps):
    return ["--export-root", str(export), "--cache-root", str(cache), "--fold", "0", "--steps", str(steps),
            "--tiny", "--cpu", "--workers", "0", "--ckpt-every", "1", "--warmup", "1", "--out", str(out)]


def test_a_tiny_run_writes_metrics_and_a_last_checkpoint(synthetic_m1r, tmp_path):
    export, cache, _ = synthetic_m1r
    main(_args(export, cache, tmp_path / "run", 2))
    rows = [json.loads(l) for l in open(tmp_path / "run/metrics.jsonl")]
    assert [r["step"] for r in rows] == [1, 2]
    last = torch.load(tmp_path / "run/last.pt", map_location="cpu", weights_only=False)
    assert last["step"] == 2 and last["config"]["model"]["M"] == 4
    assert len(rows[0]["scans"]) == 4                     # four whole volumes per step by default


def test_resume_continues_from_the_checkpoint(synthetic_m1r, tmp_path):
    export, cache, _ = synthetic_m1r
    main(_args(export, cache, tmp_path / "run", 2))
    main(_args(export, cache, tmp_path / "run", 3) + ["--resume"])
    rows = [json.loads(l) for l in open(tmp_path / "run/metrics.jsonl")]
    assert [r["step"] for r in rows] == [1, 2, 3]

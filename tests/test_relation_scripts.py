import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("sklearn")
pytest.importorskip("torch")

from anatobind.level_r.admin import FINAL_COLUMNS, seal
from anatobind.relation.cv import read_preds
from anatobind.relation.table import write_table
from synth_relation import synthetic_table


def _script(name):
    """scripts/ is not a package; load it the way tests/test_eval_knee_folds_script.py does."""
    path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


run_main = _script("run_relation_baselines").main
eval_main = _script("eval_relation_baselines").main


@pytest.fixture
def table_dir(tmp_path):
    t, p = synthetic_table(10, 3)
    write_table(tmp_path / "v1", t.rows, p, {"version": "v1"})
    return tmp_path / "v1", t


def test_run_then_eval_on_c1_and_on_synthetic_sealed_labels(tmp_path, table_dir):
    tdir, t = table_dir
    run = tmp_path / "run"
    run_main(["--table", str(tdir), "--out", str(run), "--labels", "C1", "--device", "cpu", "--epochs", "1", "--patience", "1", "--inner-k", "2"])
    names = {"b0", "bprior_majority", "bprior_type", "bprior_type_side", "bprior_type_side_location", "bgeo_lr", "bgeo_hgb", "bgeo_mlp", "b1", "b2"}
    assert {p.stem for p in (run / "preds").glob("*.csv")} == names
    assert sorted(read_preds(run / "preds" / "b1.csv")["lesion_id"].tolist()) == list(range(30))
    meta = json.loads((run / "run.json").read_text())
    assert meta["bgeo_best"] in ("bgeo_lr", "bgeo_hgb", "bgeo_mlp") and meta["labels"] == "C1" and set(meta["records"]) == names
    assert (run / "models" / "b1_fold0.pt").exists()
    with pytest.raises(FileExistsError):
        run_main(["--table", str(tdir), "--out", str(run), "--labels", "C1", "--device", "cpu", "--epochs", "1"])

    out = tmp_path / "eval_c1"
    eval_main(["--run", str(run), "--table", str(tdir), "--labels", "C1", "--out", str(out), "--n-boot", "50"])
    report = (out / "REPORT.md").read_text()
    for needle in ("NOT_EVIDENCE", "Gate R1", "agreement with C1", "tie", "bgeo_", "## Strata", "delta_d"):
        assert needle in report
    rows = list(csv.DictReader(open(out / "tables.csv")))
    assert {r["arm"] for r in rows} >= names and (out / "output.txt").exists()

    final = tmp_path / "final.csv"
    with open(final, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FINAL_COLUMNS)
        w.writeheader()
        for r in t.rows:
            w.writerow({"lesion_id": r["lesion_id"], "status": "agreed", "primary_host": r["c1_class"],
                        "acceptable_hosts": json.dumps([r["c1_class"]]), "not_a_lesion": r["lesion_id"] == 0,
                        "lesion_type": None, "side": None, "lobe": None})
    seal(final, {r["lesion_id"]: r["fold"] for r in t.rows}, tmp_path / "sealed", tmp_path / "manifest.json", k=5, now="2026-10-01T00:00:00+00:00")
    with pytest.raises(SystemExit) as e:
        eval_main(["--run", str(run), "--table", str(tdir), "--labels", "R", "--sealed-dir", str(tmp_path / "sealed"),
                   "--manifest", str(tmp_path / "manifest.json"), "--out", str(tmp_path / "eval_r")])
    assert e.value.code == 2
    eval_main(["--run", str(run), "--table", str(tdir), "--labels", "R", "--unblind", "--sealed-dir", str(tmp_path / "sealed"),
               "--manifest", str(tmp_path / "manifest.json"), "--out", str(tmp_path / "eval_r"), "--n-boot", "50"])
    report_r = (tmp_path / "eval_r" / "REPORT.md").read_text()
    assert "NOT_EVIDENCE" not in report_r and "excluded (not_a_lesion): 1" in report_r


def test_eval_missing_required_arms_exits_2(tmp_path, table_dir):
    tdir, t = table_dir
    run = tmp_path / "run_b1_only"
    run_main(["--table", str(tdir), "--out", str(run), "--labels", "C1", "--arms", "b1", "--device", "cpu",
              "--epochs", "1", "--patience", "1", "--inner-k", "2"])
    with pytest.raises(SystemExit) as e:
        eval_main(["--run", str(run), "--table", str(tdir), "--labels", "C1", "--out", str(tmp_path / "eval_b1_only")])
    assert e.value.code == 2


def test_stage_two_init_from_stage_one(tmp_path, table_dir):
    tdir, t = table_dir
    settings = ["--labels", "C1", "--arms", "b1", "--device", "cpu", "--epochs", "1", "--patience", "1", "--inner-k", "2"]
    run1 = tmp_path / "run1"
    run_main(["--table", str(tdir), "--out", str(run1)] + settings)
    run2 = tmp_path / "run2"
    run_main(["--table", str(tdir), "--out", str(run2)] + settings + ["--init-from", str(run1)])
    preds2 = read_preds(run2 / "preds" / "b1.csv")
    assert sorted(preds2["lesion_id"].tolist()) == list(range(30))
    meta1 = json.loads((run1 / "run.json").read_text())
    meta2 = json.loads((run2 / "run.json").read_text())
    assert meta2["init_from"] == str(run1)
    for k, f1 in meta1["records"]["b1"]["folds"].items():
        assert meta2["records"]["b1"]["folds"][k]["chosen"] == f1["chosen"]

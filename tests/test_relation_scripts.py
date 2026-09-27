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
from anatobind.relation.table import SLOTS, write_table
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


def test_stage_two_on_sealed_r_labels_trains_on_training_folds_only(tmp_path):
    """The stage-2 path end to end on 160 lesions: sealed R labels with sets outside the candidates, a two-element set,
    the reader host "other" and one not_a_lesion; the run reads training folds only and the final eval unblinds once."""
    t, p = synthetic_table(40, 4)
    tdir = tmp_path / "v1"
    write_table(tdir, t.rows, p, {"version": "v1"})
    cand = t.candidates()
    outside = 0
    with open(tmp_path / "final.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FINAL_COLUMNS)
        w.writeheader()
        for i, r in enumerate(t.rows):
            reader = lambda s: "other" if s == "other_deep_grey" else s                        # noqa: E731
            not_cand = [s for j, s in enumerate(SLOTS) if not cand[i, j]]
            if i % 4 == 0 and not_cand:
                hosts = [reader(not_cand[0])]                                                   # truth outside the candidates
                outside += 1
            elif i % 4 == 1:
                hosts = [r["c1_class"], "cortex" if r["c1_class"] != "cortex" else "white_matter"]
            elif i % 8 == 2:
                hosts = ["other"]
            else:
                hosts = [r["c1_class"]]
            w.writerow({"lesion_id": r["lesion_id"], "status": "agreed", "primary_host": hosts[0], "acceptable_hosts": json.dumps(hosts),
                        "not_a_lesion": i == 5, "lesion_type": None, "side": None, "lobe": None})
    assert outside > 0
    sealed, manifest = tmp_path / "sealed", tmp_path / "manifest.json"
    seal(tmp_path / "final.csv", {r["lesion_id"]: r["fold"] for r in t.rows}, sealed, manifest, k=5, now="2026-10-01T00:00:00+00:00")
    small = ["--arms", "b0,bprior,bgeo,b1,b2", "--device", "cpu", "--epochs", "1", "--patience", "1", "--inner-k", "2"]
    run1, run2 = tmp_path / "run1", tmp_path / "run2"
    run_main(["--table", str(tdir), "--out", str(run1), "--labels", "C1"] + small)
    run_main(["--table", str(tdir), "--out", str(run2), "--labels", "R", "--sealed-dir", str(sealed), "--manifest", str(manifest),
              "--init-from", str(run1)] + small)
    assert not (sealed / "access_log.txt").exists()                                            # no test fold was ever read
    meta1 = json.loads((run1 / "run.json").read_text())
    meta2 = json.loads((run2 / "run.json").read_text())
    for pf in (run2 / "preds").glob("*.csv"):
        assert sorted(read_preds(pf)["lesion_id"].tolist()) == list(range(160)), pf.stem
    rec = meta2["records"]
    per_arm = {"bprior_majority": rec["bprior_majority"]["n_untrainable"]}
    per_arm.update({a: {k: f["n_untrainable"] for k, f in rec[a]["folds"].items()} for a in ("bgeo_lr", "bgeo_hgb", "bgeo_mlp", "b1", "b2")})
    for v in ("type", "type_side", "type_side_location"):
        per_arm[f"bprior_{v}"] = rec[f"bprior_{v}"]["n_untrainable"]
    ref = per_arm["b1"]
    assert set(ref) == {"0", "1", "2", "3", "4"} and all(n > 0 for n in ref.values())
    assert all(v == ref for v in per_arm.values()), per_arm
    for arm in ("b1", "b2"):
        for k, f1 in meta1["records"][arm]["folds"].items():
            assert rec[arm]["folds"][k]["chosen"] == f1["chosen"]
    out = tmp_path / "eval_r"
    eval_main(["--run", str(run2), "--table", str(tdir), "--labels", "R", "--unblind", "--sealed-dir", str(sealed),
               "--manifest", str(manifest), "--out", str(out), "--n-boot", "50"])
    assert "excluded (not_a_lesion): 1" in (out / "REPORT.md").read_text()


def test_eval_refuses_an_existing_out_and_a_table_other_than_the_run_s(tmp_path, table_dir, capsys):
    tdir, t = table_dir
    run = tmp_path / "run_sklearn"
    run_main(["--table", str(tdir), "--out", str(run), "--labels", "C1", "--arms", "b0,bprior,bgeo", "--inner-k", "2"])
    out = tmp_path / "eval"
    eval_main(["--run", str(run), "--table", str(tdir), "--labels", "C1", "--out", str(out), "--n-boot", "20"])
    with pytest.raises(FileExistsError):
        eval_main(["--run", str(run), "--table", str(tdir), "--labels", "C1", "--out", str(out), "--n-boot", "20"])
    other = tmp_path / "v2"
    write_table(other, t.rows, synthetic_table(10, 3)[1], {"version": "v2"})           # same rows, another manifest
    capsys.readouterr()
    with pytest.raises(SystemExit) as e:
        eval_main(["--run", str(run), "--table", str(other), "--labels", "C1", "--out", str(tmp_path / "eval_v2"), "--n-boot", "20"])
    assert e.value.code == 2 and "table_manifest_sha256" in capsys.readouterr().err
    assert not (tmp_path / "eval_v2").exists()

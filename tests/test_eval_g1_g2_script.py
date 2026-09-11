import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from anatobind.eval.predict import save_prediction
from anatobind.train.cache import VIEWS, load_array
from anatobind.train.dataset_v2 import read_all_boxes


def _load_script():
    path = Path(__file__).resolve().parents[1] / "scripts/eval_g1_g2.py"
    spec = importlib.util.spec_from_file_location("eval_g1_g2", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _perfect(export, cache, pred, scans):
    for scan in scans:
        seg = np.asarray(load_array(cache / scan, "seg"))
        rows = read_all_boxes(export / scan / "boxes.csv")
        cls_prob = np.zeros((6, 5), np.float32)
        cls_prob[:, 4] = 1.0
        boxes = np.zeros((6, 6), np.float32)
        for q, r in enumerate(rows):
            cls_prob[q] = 0.0
            cls_prob[q, r["cls"]] = 1.0
            boxes[q] = r["box"]
        p = {"label_map": seg.astype(np.uint8), "presence": np.ones(6, np.float32), "cls_prob": cls_prob,
             "boxes_vox": boxes, "a_embed": np.zeros((6, 4), np.float16), "u_embed": np.zeros((6, 4), np.float16)}
        for fold in range(5):
            (pred / "ours" / f"fold{fold}").mkdir(parents=True, exist_ok=True)
            for view in VIEWS:
                save_prediction(pred / "ours" / f"fold{fold}" / f"{scan}__{view}.npz", p)
        (pred / "nnunet").mkdir(parents=True, exist_ok=True)
        for view in VIEWS:
            np.savez_compressed(pred / "nnunet" / f"{scan}__{view}.npz", label_map=seg.astype(np.uint8))


def test_perfect_predictions_give_all_correct_and_a_failed_gate(synthetic_m1r, tmp_path):
    export, cache, scans = synthetic_m1r
    _perfect(export, cache, tmp_path / "pred", scans)
    out = tmp_path / "report"
    _load_script().main(["--out", str(out), "--export-root", str(export), "--cache-root", str(cache),
                         "--pred-root", str(tmp_path / "pred"), "--figs", str(tmp_path / "figs")])
    with open(out / "records.csv", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 5 * 7 * 3 * 4
    in_seg = [r for r in rows if r["layer"] == "in_seg" and r["thr"] == "0.1"]
    assert {r["bucket_nnunet"] for r in in_seg} == {"correct"} and {r["bucket_ours"] for r in in_seg} == {"correct"}
    summary = json.loads((out / "summary.json").read_text())
    assert summary["g2"]["nnunet"]["pass"] is False and summary["g2"]["nnunet"]["noise_q3"]["delta"] == pytest.approx(0.0)
    assert (tmp_path / "figs/buckets_by_view.png").exists() and (tmp_path / "figs/paired_wrong_host.png").exists()

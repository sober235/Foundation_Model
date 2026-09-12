import importlib.util
import json
from pathlib import Path

import numpy as np

from anatobind.eval.predict import save_prediction
from anatobind.train.cache import VIEWS, load_array
from anatobind.train.dataset_v2 import read_all_boxes


def _load():
    path = Path(__file__).resolve().parents[1] / "scripts/peek_fold.py"
    spec = importlib.util.spec_from_file_location("peek_fold", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_perfect_fold_predictions_score_all_correct(synthetic_m1r, tmp_path):
    export, cache, scans = synthetic_m1r
    pred = tmp_path / "pred" / "ours" / "fold0"
    pred.mkdir(parents=True)
    for scan in scans:
        seg = np.asarray(load_array(cache / scan, "seg")).astype(np.uint8)
        rows = read_all_boxes(export / scan / "boxes.csv")
        cls_prob = np.zeros((6, 5), np.float32)
        cls_prob[:, 4] = 1.0
        boxes = np.zeros((6, 6), np.float32)
        for q, r in enumerate(rows):
            cls_prob[q] = 0.0
            cls_prob[q, r["cls"]] = 1.0
            boxes[q] = r["box"]
        p = {"label_map": seg, "presence": np.ones(6, np.float32), "cls_prob": cls_prob, "boxes_vox": boxes,
             "a_embed": np.zeros((6, 4), np.float16), "u_embed": np.zeros((6, 4), np.float16)}
        for view in VIEWS:
            save_prediction(pred / f"{scan}__{view}.npz", p)
    out = _load().peek(fold=0, export_root=export, cache_root=cache, pred_root=tmp_path / "pred")
    assert out["held_out_scans"] == 1 and out["training_scans"] == 4
    for view in VIEWS:
        assert out["detection"][view]["recall"] == 1.0
        assert out["buckets"][view] == {"correct": 2}
        assert out["dice"][view]["held_out"] == 1.0 and out["dice"][view]["training"] == 1.0

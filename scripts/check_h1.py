#!/usr/bin/env python
# scripts/check_h1.py
"""H1 止损门（leg 2 spec 3.4）：折外检出是否多到、且对错是否均衡到，值得去训可靠性头。

阈值只在训练折上选：先在 folds 1-4 的折外检出上找到使扫描级正样本率落入 [0.2, 0.5] 的最小分数阈值，
再把该阈值固定，用它在全部五折上计算 H1 的两个条件。
"""
import argparse
import csv
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from anatobind.data_engine.fastmri_knee import EXPORT_ROOT, VIEWS  # noqa: E402
from anatobind.eval.detect3d import failure_labels  # noqa: E402
from anatobind.train.train_detector import load_fold  # noqa: E402


def gather(export_root, det_root, folds, score_min):
    per_all, scan_all = [], []
    with open(Path(export_root) / "lesions.csv", newline="") as fh:
        by_file = {}
        for r in csv.DictReader(fh):
            by_file.setdefault(r["file"], []).append(
                {**r, **{k: int(r[k]) for k in ("z0", "z1", "x0", "y0", "x1", "y1")}})
    for fold in folds:
        _, held, _ = load_fold(export_root, fold)
        for f in held:
            for view in VIEWS:
                p = Path(det_root) / f"fold{fold}" / f"{f}__{view}.pkl"
                if not p.exists():
                    continue
                dets = [d for d in pickle.load(open(p, "rb"))["dets"] if d["score"] >= score_min]
                per, scan = failure_labels(by_file.get(f, []), dets)
                for r in per:
                    per_all.append({**r, "file": f, "view": view, "fold": fold})
                scan_all.append({"file": f, "view": view, "fold": fold, "label": scan})
    return per_all, scan_all


def scan_positive_rate(scan_all):
    """扫描级正样本率；没有扫描时视为 1.0（不可采纳，逼阈值搜索继续往上走，绝不悄悄当成 0）。"""
    return float(np.mean([s["label"] for s in scan_all])) if scan_all else 1.0


def pick_threshold(thresholds, rate_of, lo=0.2, hi=0.5):
    """按传入顺序遍历阈值，返回第一个使 rate_of(thr) 落入 [lo, hi] 的阈值；全都不落入则 None（不悄悄挑一个凑数）。"""
    for thr in thresholds:
        if lo <= rate_of(thr) <= hi:
            return float(thr)
    return None


def h1_verdict(per, min_det=300, lo=0.2, hi=0.8):
    """H1 的两个条件：检出数 >= min_det，且逐病灶正确率落在 [lo, hi]（太高=没什么可学，太低=检测器本身不可用）。"""
    n_det = len(per)
    correct_rate = float(np.mean([r["correct"] for r in per])) if per else 0.0
    passed = n_det >= min_det and lo <= correct_rate <= hi
    return {"n_det": n_det, "correct_rate": correct_rate, "passed": passed}


TRAIN_FOLDS = (1, 2, 3, 4)
ALL_FOLDS = (0, 1, 2, 3, 4)


def choose_and_measure(gather_fn, thresholds, train_folds=TRAIN_FOLDS, all_folds=ALL_FOLDS, on_threshold=None):
    """阈值只在 train_folds 的折外检出上选；选定后用它在 all_folds 上算 H1 的两个条件。

    gather_fn(folds, score_min) -> (per, scan)，与 gather() 同型，测试时可换成假的，
    不用碰真实文件系统。找不到可采纳的阈值时返回 (None, [], [])，不悄悄挑一个凑数。
    """
    def rate_of(thr):
        rate = scan_positive_rate(gather_fn(train_folds, thr)[1])
        if on_threshold is not None:
            on_threshold(thr, rate)
        return rate

    chosen = pick_threshold(thresholds, rate_of)
    if chosen is None:
        return None, [], []
    per, scan = gather_fn(all_folds, chosen)
    return chosen, per, scan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export-root", type=Path, default=EXPORT_ROOT)
    ap.add_argument("--det-root", type=Path, default=EXPORT_ROOT / "detections")
    a = ap.parse_args()

    def log(thr, rate):
        print(f"threshold {thr:.2f}: scan-level positive rate on training folds {rate:.3f}")

    chosen, per, scan = choose_and_measure(
        lambda folds, thr: gather(a.export_root, a.det_root, folds, thr),
        np.arange(0.05, 0.96, 0.05), on_threshold=log)
    if chosen is None:
        print("H1 FAIL: no threshold puts the scan-level positive rate in [0.2, 0.5]")
        sys.exit(1)

    verdict = h1_verdict(per)
    print(f"\nSCORE_MIN = {chosen:.2f}")
    print(f"detections (all folds, all views): {verdict['n_det']}; "
          f"per-lesion correct rate: {verdict['correct_rate']:.3f}")
    print(f"scan-level positive rate: {scan_positive_rate(scan):.3f}")
    print(f"H1 {'PASS' if verdict['passed'] else 'FAIL'} (needs >= 300 detections and correct rate in [0.2, 0.8])")
    sys.exit(0 if verdict["passed"] else 1)


if __name__ == "__main__":
    main()

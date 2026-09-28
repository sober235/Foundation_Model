#!/usr/bin/env python
# scripts/nndet_runner.py
"""nnDetection side of the brain second arm (spec 2026-09-28 brain-nndet §5, §7, §8).

Runs ONLY in the nndet env (python 3.8, torch 1.11, nnDetection 97a58f3) and never imports anatobind. It writes
{"layout", "source", "cases"} JSON; boxes are nnDetection's layout on the ORIGINAL image array axes (slice, row, col),
[s_lo, r_lo, s_hi, r_hi, c_lo, c_hi], half-open floats, after undoing nnDetection's one-voxel margin.

  bash -c 'source scripts/nndet_env.sh && python scripts/nndet_runner.py extract --state DIR --params default --out J'
  ... extract --state DIR --params swept --train-dir DIR --out J
  ... gt --prep ${det_data}/Task903_FastMRIBrainSmallLesion/preprocessed --out J
  ... predict --image NII --train-dir DIR --work DIR --out J
"""
import argparse
import json
import pickle
import shutil
from pathlib import Path

import numpy as np

LAYOUT = "nndet [s_lo, r_lo, s_hi, r_hi, c_lo, c_hi] on original array axes (slice, row, col), half-open"
LOW_ENDS = [0, 1, 4]            # x1, y1, z1 of nnDetection's (x1, y1, x2, y2, z1, z2)


def undo_margin(boxes):
    """nnDetection boxes are [min - 1, max + 1] per axis (instances_to_boxes_np); shifting the low ends by +1 gives
    the half-open [min, max + 1). Applies in the preprocessed space, before restoring."""
    b = np.array(boxes, dtype=float).reshape(-1, 6)
    b[:, LOW_ENDS] += 1.0
    return b


def to_original(boxes_prep, properties, transpose_backward):
    b = undo_margin(boxes_prep)
    if len(b) == 0:
        return b
    from nndet.inference.restore import restore_detection
    return np.asarray(restore_detection(b, transpose_backward=list(transpose_backward),
                                        original_spacing=properties["original_spacing"],
                                        spacing_after_resampling=properties["spacing_after_resampling"],
                                        crop_bbox=properties["crop_bbox"]), float)


def _np(x):
    return np.asarray(x.detach().cpu().numpy() if hasattr(x, "detach") else x)


def default_parameters():
    from nndet.inference.ensembler.detection import BoxEnsemblerSelective
    return BoxEnsemblerSelective.get_default_parameters()


def extract_cases(state_dir, params):
    """Saved ensembler states (sweep_predictions, or predict's state dir) -> postprocess with params, restore=False,
    then undo the margin and restore with each case's own properties."""
    from nndet.inference.ensembler.detection import BoxEnsemblerSelective
    out = {}
    for case_id in sorted(BoxEnsemblerSelective.get_case_ids(state_dir)):
        ens = BoxEnsemblerSelective.from_checkpoint(base_dir=state_dir, case_id=case_id)
        ens.update_parameters(**params)
        res = ens.get_case_result(restore=False)
        boxes = _np(res["pred_boxes"]).reshape(-1, 6)
        p = ens.properties
        out[case_id] = {"boxes": to_original(boxes, p, p["transpose_backward"]).tolist(),
                        "scores": [float(s) for s in _np(res["pred_scores"]).reshape(-1)],
                        "labels": [int(v) for v in _np(res["pred_labels"]).reshape(-1)]}
    return out


def gt_cases(prep_dir, plan_id="D3V001_3d"):
    """Ground-truth boxes after nnDetection preprocessing, through the same conversion as predictions (spec §5 check
    two), with the instance ids still present after resampling."""
    prep_dir = Path(prep_dir)
    plan = pickle.loads((prep_dir / f"{plan_id}.pkl").read_bytes())
    stage = prep_dir / plan["data_identifier"] / "imagesTr"
    out = {}
    for f in sorted(stage.glob("*_boxes.pkl")):
        case_id = f.name[:-len("_boxes.pkl")]
        cand = pickle.loads(f.read_bytes())
        props = pickle.loads((stage / f"{case_id}.pkl").read_bytes())
        ids = [int(i) for i in cand["instances"]]
        boxes = np.asarray(cand["boxes"], float).reshape(-1, 6) if ids else np.zeros((0, 6))
        if len(boxes) != len(ids):
            raise ValueError(f"{case_id}: {len(boxes)} boxes for {len(ids)} instances")
        out[case_id] = {"boxes": to_original(boxes, props, plan["transpose_backward"]).tolist(),
                        "scores": [1.0] * len(ids), "instances": ids}
    return out


def predict_case(image, train_dir, work, num_processes=0):
    """One FLAIR NIfTI -> detections with model_last, 8 mirror TTA and default postprocessing (spec §8)."""
    from nndet.inference.helper import predict_dir
    from nndet.inference.loading import get_loader_fn
    from nndet.planning import PLANNER_REGISTRY
    from omegaconf import OmegaConf
    work, train_dir = Path(work), Path(train_dir)
    if work.exists():
        raise FileExistsError(f"{work} exists")
    raw = work / "raw_splitted" / "imagesTs"
    raw.mkdir(parents=True)
    shutil.copyfile(image, raw / "case_0000.nii.gz")
    plan = pickle.loads((train_dir / "plan.pkl").read_bytes())
    plan["inference_plan"] = {}                     # empty -> BoxEnsembler.from_case falls back to its defaults
    cfg = OmegaConf.load(str(train_dir / "config.yaml"))
    cfg.merge_with_dotlist(["host.parent_data=${oc.env:det_data}", "host.parent_results=${oc.env:det_models}"])
    cfg = OmegaConf.to_container(cfg, resolve=True)
    PLANNER_REGISTRY.get(plan["planner_id"]).run_preprocessing_test(
        preprocessed_output_dir=work / "preprocessed", splitted_4d_output_dir=work / "raw_splitted", plan=plan,
        num_processes=num_processes)
    state = work / "state"
    state.mkdir()
    predict_dir(source_dir=work / "preprocessed" / plan["data_identifier"] / "imagesTs", target_dir=state, cfg=cfg,
                plan=plan, source_models=train_dir, num_models=1, num_tta_transforms=None,
                model_fn=get_loader_fn(mode="last"), restore=False, save_state=True)
    return extract_cases(state, default_parameters())


def main(argv=None):
    ap = argparse.ArgumentParser(description="nnDetection runner for the brain second arm (nndet env only)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("--state", type=Path, required=True)
    e.add_argument("--params", choices=("default", "swept"), required=True)
    e.add_argument("--train-dir", type=Path)
    e.add_argument("--out", type=Path, required=True)
    g = sub.add_parser("gt")
    g.add_argument("--prep", type=Path, required=True)
    g.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("predict")
    p.add_argument("--image", type=Path, required=True)
    p.add_argument("--train-dir", type=Path, required=True)
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    if a.out.exists():
        raise FileExistsError(f"{a.out} exists")
    if a.cmd == "extract":
        if a.params == "default":
            params = default_parameters()
        else:
            if a.train_dir is None:
                ap.error("--params swept needs --train-dir")
            params = pickle.loads((a.train_dir / "plan_inference.pkl").read_bytes())["inference_plan"]
        cases, source = extract_cases(a.state, params), {"state": str(a.state), "params": a.params}
    elif a.cmd == "gt":
        cases, source = gt_cases(a.prep), {"prep": str(a.prep)}
    else:
        cases = predict_case(a.image, a.train_dir, a.work)
        source = {"image": str(a.image), "train_dir": str(a.train_dir)}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"layout": LAYOUT, "source": source, "cases": cases}))
    print(f"wrote {a.out}: {len(cases)} cases, {sum(len(c['boxes']) for c in cases.values())} boxes")


if __name__ == "__main__":
    main()

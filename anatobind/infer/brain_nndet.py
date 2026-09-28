"""Brain small-lesion inference with the nnDetection second arm, for S5 (spec 2026-09-28 brain-nndet §8): a fastMRI
FLAIR h5 -> RSS NIfTI -> the nndet runner (model_last, default postprocessing) -> S2-format lesion rows."""
import json
import shlex
import subprocess
from pathlib import Path

import nibabel as nib

from anatobind.data_engine.fastmri import rss_h5_to_nifti
from anatobind.nndet.boxes import detections, lesion_rows, load_runner_json

REPO = Path(__file__).resolve().parents[2]


def run_runner(image, train_dir, work, out_json, gpu):
    q = lambda p: shlex.quote(str(p))  # noqa: E731
    inner = (f"source {q(REPO / 'scripts/nndet_env.sh')} && CUDA_VISIBLE_DEVICES={int(gpu)} nice -n 19 python "
             f"{q(REPO / 'scripts/nndet_runner.py')} predict --image {q(image)} --train-dir {q(train_dir)} "
             f"--work {q(work)} --out {q(out_json)}")
    subprocess.run(["bash", "-c", inner], check=True)


def run(h5_path, out_dir, train_dir, gpu, predict=run_runner):
    out = Path(out_dir)
    if out.exists():
        raise FileExistsError(f"{out} exists")
    (out / "input").mkdir(parents=True)
    img = out / "input" / "case_0000.nii.gz"
    rss_h5_to_nifti(h5_path, img, pad_to_slices=0)
    predict(img, train_dir, out / "work", out / "pred.json", gpu)
    cases = load_runner_json(out / "pred.json")["cases"]
    if set(cases) != {"case"}:
        raise ValueError(f"runner returned cases {sorted(cases)}, expected ['case']")
    rows = lesion_rows(detections(cases["case"]), nib.load(str(img)).shape)
    (out / "lesions.json").write_text(json.dumps(rows, indent=1))
    return rows

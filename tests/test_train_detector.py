import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from anatobind.data_engine.fastmri_knee import VIEWS


def _export(root, files, slices=9, size=32):
    import csv
    rng = np.random.default_rng(0)
    rows = []
    for i, name in enumerate(files):
        d = root / name
        d.mkdir(parents=True)
        for v in VIEWS:
            np.save(d / f"{v}.npy", rng.normal(size=(slices, size, size)).astype(np.float16))
        (d / "meta.json").write_text(json.dumps({"slices": slices, "size": size, "patient_id": f"p{i}"}))
        rows.append({"lesion_id": i, "file": name, "family": "meniscus", "z0": 3, "z1": 5,
                     "x0": 4, "y0": 6, "x1": 14, "y1": 18, "n_boxes": 3})
    with open(root / "lesions.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (root / "folds.json").write_text(json.dumps({"folds": {f: i % 5 for i, f in enumerate(files)}}))


def test_training_runs_logs_and_writes_a_resumable_checkpoint(tmp_path):
    from anatobind.train.train_detector import main
    root = tmp_path / "leg2"
    root.mkdir()
    _export(root, [f"file{i}" for i in range(5)])
    out = tmp_path / "run"
    main(["--fold", "0", "--steps", "3", "--batch", "2", "--out", str(out), "--export-root", str(root),
          "--tiny", "--cpu", "--workers", "0", "--ckpt-every", "2", "--log-every", "1"])
    rows = [json.loads(l) for l in open(out / "metrics.jsonl")]
    assert len(rows) == 3 and all(np.isfinite(r["loss"]) for r in rows)
    assert (out / "last.pt").exists() and (out / "ckpt.pt").exists()
    cfg = json.loads((out / "config.json").read_text())
    assert cfg["fold"] == 0 and cfg["n_train"] == 4
    last = torch.load(out / "last.pt", map_location="cpu", weights_only=False)
    assert last["step"] == 3 and last["config"]["model"]["num_classes"] == 5


def test_resume_continues_from_the_checkpoint(tmp_path):
    from anatobind.train.train_detector import main
    root = tmp_path / "leg2"
    root.mkdir()
    _export(root, [f"file{i}" for i in range(5)])
    out = tmp_path / "run"
    args = ["--fold", "0", "--batch", "2", "--out", str(out), "--export-root", str(root),
            "--tiny", "--cpu", "--workers", "0", "--ckpt-every", "1", "--log-every", "1"]
    main(args + ["--steps", "2"])
    main(args + ["--steps", "4", "--resume"])
    rows = [json.loads(l) for l in open(out / "metrics.jsonl")]
    assert [r["step"] for r in rows] == [1, 2, 3, 4]
